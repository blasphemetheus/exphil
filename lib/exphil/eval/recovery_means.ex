defmodule ExPhil.Eval.RecoveryMeans do
  @moduledoc """
  Recovery-means scorecard (2026-10-05): does the player pick the recovery
  tool the expert picks from the same spot?

  The 10-04 SD review found the bot's deaths are mostly MIXUP stocks thrown
  by the wrong means for its height — side-B from below the ledge (illusion
  falls short), airdodge at the ledge with a jump in hand, laser/shine at the
  ledge. A death list is survivorship-biased, so this scores EVERY offstage
  episode: the situation at the decision frame (first actionable frame
  offstage) and the first means the player reached for, compared against the
  expert's distribution over the same situation bucket.

  Frames are `%{own: player, opp: player, controller: c}` like
  `ExPhil.Eval.PlayStats` so sim rollouts and parsed .slp games both feed it.

  Situation = `height` (relative to the stage surface: high / ledge / low /
  deep) × `dist` (beyond the edge: near / mid / far) × `jumps` (0 / 1+).
  Means (first of): `jump` (double jump spent), `side_b`, `up_b`, `airdodge`,
  `attack` (aerial, laser, shine), `drift` (back on stage with none of them),
  `none` (died with none of them).
  """

  @means ~w(jump side_b up_b airdodge attack drift none)a
  # a bucket thinner than this in the reference falls back to its height-band marginal
  @min_bucket 20

  @doc "Offstage episodes of one game: situation at the decision frame, means, outcome."
  def episodes(frames, edge) when is_list(frames) do
    init = %{off: false, actionable: nil, path: [], last_hit: -10_000, out: [], hist: []}

    r =
      frames
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index(1)
      |> Enum.reduce(init, fn {[f0, f1], i}, e ->
        p0 = f0.own
        p1 = f1.own
        hit? = (p1.percent || 0) > (p0.percent || 0) or stun?(p1)
        died? = (p1.stock || 0) < (p0.stock || 0)
        now_off = offstage?(p1, edge) and not died?

        actionable =
          cond do
            hit? -> nil
            now_off and not stun?(p1) and (stun?(p0) or not e.off) ->
              # a re-actionable frame inside the same trip keeps the trip's approach
              {p1, i, if(e.actionable == nil, do: Enum.take([f0 | e.hist], 90), else: elem(e.actionable, 2))}
            true -> e.actionable
          end

        path = if actionable == nil, do: [], else: [f1 | e.path]
        last_hit = if hit?, do: i, else: e.last_hit
        # the 90 frames before the decision frame, frozen when the trip starts
        # (the approach fields read the last 30; pre_trace keeps all 90 so the
        # jump spend before a carried-off trip is on the record — 10-10)
        hist = Enum.take([f0 | e.hist], 90)

        cond do
          died? and e.actionable != nil ->
            %{e | off: false, actionable: nil, path: [],
                  out: [episode(e.actionable, Enum.reverse(e.path), :died, i - last_hit > 90, edge) | e.out]}

          died? ->
            %{e | off: false, actionable: nil, path: []}

          e.off and not now_off and e.actionable != nil ->
            %{e | off: false, actionable: nil, path: [],
                  out: [episode(e.actionable, Enum.reverse(e.path), :returned, false, edge) | e.out]}

          true ->
            %{e | off: now_off, actionable: actionable, path: path, last_hit: last_hit, hist: hist}
        end
      end)

    Enum.reverse(r.out)
  end

  defp episode({p, _i, pre}, path, outcome, sd?, edge) do
    {means, first_at, first_y} = means_sequence(p, path)
    pre_all = Enum.reverse(pre)
    pre = Enum.take(pre_all, -30)
    sign = if (p.x || 0.0) >= 0, do: 1, else: -1
    toward = fn f -> f.controller != nil and (f.controller.main_stick.x - 0.5) * sign >= 0.33 end
    pre_speed = fn f -> abs((Map.get(f.own, :speed_ground_x_self) || 0.0) + (Map.get(f.own, :speed_air_x_self) || 0.0)) end
    first = List.first(means) || if(outcome == :died, do: :none, else: :drift)

    %{
      height: height_band(p.y || 0.0),
      dist: dist_band(abs(p.x || 0.0) - edge),
      jumps: if((p.jumps_left || 0) >= 1, do: 1, else: 0),
      x: Float.round((p.x || 0.0) * 1.0, 1),
      y: Float.round((p.y || 0.0) * 1.0, 1),
      first: first,
      # where and when the first means fired (the height band above is the
      # DECISION frame's; a side-B 40 frames later fires from far lower)
      first_at: first_at,
      first_y: first_y,
      first_height: if(first_y, do: height_band(first_y)),
      seq: Enum.join(Enum.take(means, 3), ">"),
      # the approach: what the player was doing in the 30 frames before the trip
      pre_actions: pre |> Enum.map(&(&1.own.action || 0)) |> Enum.dedup() |> Enum.take(-5),
      pre_stick_edge_share: if(pre == [], do: nil, else: Float.round(Enum.count(pre, toward) / length(pre), 2)),
      pre_speed: if(pre == [], do: nil, else: Float.round(Enum.sum(Enum.map(Enum.take(pre, -5), pre_speed)) / max(length(Enum.take(pre, -5)), 1), 2)),
      pre_facing_edge: (Map.get(p, :facing) || 1) * (p.x || 0.0) > 0,
      outcome: outcome,
      sd: sd?,
      frames: length(path),
      # the crime scene: every 3rd frame of the trip, so a died trip can be
      # read offline (drifting away? stick frozen? fast-falling?) without
      # re-rolling the sim
      trace: path |> Enum.take_every(3) |> Enum.map(&trace_frame/1),
      # the approach in the same row format, every 3rd of the 90 frames before
      # the decision frame (oldest first): where the double jump went before a
      # carried-off trip (scripts/recovery_jump_spend.js)
      pre_trace: pre_all |> Enum.reverse() |> Enum.take_every(3) |> Enum.reverse() |> Enum.map(&trace_frame/1)
    }
  end

  # [action, x, y, speed_y, stick_x, stick_y, b, jump, jumps_left]
  defp trace_frame(f) do
    c = f.controller
    r = fn v, d -> if v == nil, do: nil, else: Float.round(v * 1.0, d) end
    b = fn v -> if v, do: 1, else: 0 end

    [f.own.action || 0, r.(f.own.x, 1), r.(f.own.y, 1), r.(Map.get(f.own, :speed_y_self), 2),
     r.(c && c.main_stick.x, 2), r.(c && c.main_stick.y, 2),
     b.(c && c.button_b), b.(c && (c.button_x or c.button_y)), f.own.jumps_left || 0]
  end

  @doc "Bucket key for an episode (string, JSON-stable)."
  def bucket(%{height: h, dist: d, jumps: j}), do: "#{h}/#{d}/j#{j}"

  @doc "Counts by bucket and by height band: `%{\"bucket\" => %{\"jump\" => n, ...}}`."
  def table(episodes) do
    count = fn eps -> eps |> Enum.frequencies_by(&Atom.to_string(&1.first)) end
    buckets = episodes |> Enum.group_by(&bucket/1) |> Map.new(fn {k, v} -> {k, count.(v)} end)
    heights = episodes |> Enum.group_by(&Atom.to_string(&1.height)) |> Map.new(fn {k, v} -> {k, count.(v)} end)
    returns =
      episodes
      |> Enum.group_by(&bucket/1)
      |> Map.new(fn {k, v} -> {k, %{"n" => length(v), "returned" => Enum.count(v, &(&1.outcome == :returned))}} end)

    %{"buckets" => buckets, "heights" => heights, "returns" => returns, "episodes" => length(episodes)}
  end

  @doc """
  Score episodes against a reference table.

  - `mismatch_rate`: share of episodes whose first means the reference picks
    < 10 % of the time from the same bucket (thin buckets fall back to the
    height-band marginal).
  - `means_js`: Jensen-Shannon distance (base 2, 0..1) between the means
    distributions per bucket, weighted by the scored side's bucket counts.
  - named defects, as rates over the episodes where they can occur:
    `side_b_low` (side-B first from low/deep), `airdodge_with_jump`
    (airdodge first with a jump in hand), `nothing_died` (died with no means).
  """
  def score(episodes, ref) do
    probs = fn counts ->
      total = counts |> Map.values() |> Enum.sum()
      Map.new(@means, fn m -> {m, if(total == 0, do: 0.0, else: Map.get(counts, Atom.to_string(m), 0) / total)} end)
    end

    ref_probs = fn ep ->
      b = ref["buckets"][bucket(ep)] || %{}
      if Enum.sum(Map.values(b)) >= @min_bucket, do: probs.(b), else: probs.(ref["heights"][Atom.to_string(ep.height)] || %{})
    end

    n = length(episodes)
    mismatches = Enum.count(episodes, fn ep -> ref_probs.(ep)[ep.first] < 0.10 end)

    own = table(episodes)
    js =
      own["buckets"]
      |> Enum.map(fn {k, counts} ->
        m = Enum.sum(Map.values(counts))
        ep = hd(Enum.filter(episodes, &(bucket(&1) == k)))
        {m, js_distance(probs.(counts), ref_probs.(ep))}
      end)

    weighted = if n == 0, do: nil, else: Float.round(Enum.sum(Enum.map(js, fn {m, d} -> m * d end)) / n, 3)

    rate = fn pred, denom_pred ->
      d = Enum.count(episodes, denom_pred)
      if d == 0, do: nil, else: Float.round(Enum.count(episodes, &(denom_pred.(&1) and pred.(&1))) / d, 3)
    end

    %{
      "episodes" => n,
      "mismatch_rate" => if(n == 0, do: nil, else: Float.round(mismatches / n, 3)),
      "means_js" => weighted,
      "return_rate" => rate.(&(&1.outcome == :returned), fn _ -> true end),
      "side_b_low" => rate.(&(&1.first == :side_b), &(&1.height in [:low, :deep])),
      "airdodge_with_jump" => rate.(&(&1.first == :airdodge), &(&1.jumps == 1)),
      "nothing_died" => rate.(&(&1.first == :none), &(&1.outcome == :died)),
      # side-B that FIRED from low/deep, over all side-B-first episodes; and
      # the latency (frames from the decision frame to the first means)
      "side_b_fired_low" => rate.(&(&1.first_height in [:low, :deep]), &(&1.first == :side_b)),
      "first_means_latency_median" => median(episodes |> Enum.map(& &1.first_at) |> Enum.reject(&is_nil/1)),
      "side_b_latency_median" => median(episodes |> Enum.filter(&(&1.first == :side_b)) |> Enum.map(& &1.first_at)),
      # EDGE SELF-DESTRUCTS (10-05): a trip whose first means is already active
      # on the first offstage frame was carried off the stage by a move started
      # on it (illusion off the lip, wavedash/airdodge off, aerial/laser off).
      # Share of all trips, their death rate, and the per-move breakdown; the
      # DECIDED trips (launched, or acted offstage) scored separately.
      "carried_off_share" => rate.(&carried_off?/1, fn _ -> true end),
      "carried_off_died" => rate.(&(&1.outcome == :died), &carried_off?/1),
      "carried_off_by_move" =>
        episodes
        |> Enum.filter(&carried_off?/1)
        |> Enum.group_by(& &1.first)
        |> Map.new(fn {m, l} -> {Atom.to_string(m), %{"n" => length(l), "died" => Enum.count(l, &(&1.outcome == :died))}} end),
      "decided_n" => Enum.count(episodes, &(not carried_off?(&1))),
      "decided_return_rate" => rate.(&(&1.outcome == :returned), &(not carried_off?(&1))),
      "first_means" => Enum.frequencies_by(episodes, &Atom.to_string(&1.first)),
      "by_height" =>
        episodes
        |> Enum.group_by(& &1.height)
        |> Map.new(fn {h, eps} ->
          {Atom.to_string(h), %{"n" => length(eps), "return_rate" => Float.round(Enum.count(eps, &(&1.outcome == :returned)) / length(eps), 3),
             "first" => Enum.frequencies_by(eps, &Atom.to_string(&1.first))}}
        end)
    }
  end

  @doc "Split-half noise floor: score each half of the episodes against the other half's table."
  def split_half(episodes) do
    {a, b} = episodes |> Enum.with_index() |> Enum.split_with(fn {_, i} -> rem(i, 2) == 0 end)
    a = Enum.map(a, &elem(&1, 0))
    b = Enum.map(b, &elem(&1, 0))
    sa = score(a, table(b))
    sb = score(b, table(a))
    Map.new(~w(mismatch_rate means_js), fn k -> {k, Float.round(((sa[k] || 0.0) + (sb[k] || 0.0)) / 2, 3)} end)
  end

  # ---- means --------------------------------------------------------------------

  # first-to-last means on the way back, deduplicated; a jump is a spent double
  # jump. Also the frame offset and height at which the FIRST means fired.
  defp means_sequence(p0, path) do
    {_, acc, first_at, first_y, _} =
      Enum.reduce(path, {p0.jumps_left || 0, [], nil, nil, 0}, fn f, {jumps, acc, first_at, first_y, i} ->
        p = f.own
        a = p.action || 0
        j = p.jumps_left || 0

        m =
          cond do
            j < jumps -> :jump
            a in 350..352 -> :side_b
            a in 353..356 -> :up_b
            a == 236 -> :airdodge
            a in 65..69 or a in 341..348 or a in 360..368 -> :attack
            true -> nil
          end

        {first_at, first_y} =
          if m != nil and acc == [], do: {i, Float.round((p.y || 0.0) * 1.0, 1)}, else: {first_at, first_y}

        {min(jumps, j), if(m && List.first(acc) != m, do: [m | acc], else: acc), first_at, first_y, i + 1}
      end)

    {Enum.reverse(acc), first_at, first_y}
  end

  # y relative to the stage surface (ledge grab box reaches ~ -20 for Fox)
  defp height_band(y) when y > 0.0, do: :high
  defp height_band(y) when y > -20.0, do: :ledge
  defp height_band(y) when y > -60.0, do: :low
  defp height_band(_), do: :deep

  # horizontal distance beyond the edge
  defp dist_band(d) when d < 20.0, do: :near
  defp dist_band(d) when d < 60.0, do: :mid
  defp dist_band(_), do: :far

  # dead / rebirth action states (0..13) sit below the blast zone for ~59 frames
  # and then teleport to the revival platform: not an offstage trip
  defp carried_off?(ep), do: ep.first_at == 0

  defp median([]), do: nil
  defp median(l), do: l |> Enum.sort() |> Enum.at(div(length(l), 2))

  # y < -12 rather than PlayStats' -5: a wavedash's airdodge and an onstage
  # illusion dip to y ~ -5.5 on the surface and are not trips (seen 10-05)
  defp offstage?(p, edge),
    do: not (p.on_ground == true) and (p.action || 99) > 13 and (abs(p.x || 0.0) > edge or (p.y || 0.0) < -12.0)
  defp stun?(p), do: (Map.get(p, :hitstun_frames_left) || 0) > 0

  defp js_distance(p, q) do
    m = Map.new(p, fn {k, v} -> {k, (v + q[k]) / 2} end)
    kl = fn a, b -> Enum.reduce(a, 0.0, fn {k, v}, s -> if v > 0, do: s + v * :math.log2(v / b[k]), else: s end) end
    :math.sqrt(max(0.0, (kl.(p, m) + kl.(q, m)) / 2))
  end
end
