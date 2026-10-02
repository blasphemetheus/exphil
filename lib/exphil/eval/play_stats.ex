defmodule ExPhil.Eval.PlayStats do
  @moduledoc """
  Behaviour statistics of ONE player over contiguous games, computed the same
  way from expert replays and from a policy's sim rollouts, so the two can be
  compared as distributions ("does it play like what it imitates?") rather
  than by outcomes (2026-10-02, INPUT_COHERENCE doc).

  A game is a list of `%{own: Player, opp: Player, controller: ControllerState}`
  in frame order. `from_game/2` returns raw counts (mergeable with `merge/2`);
  `summarize/1` turns counts into per-minute rates and normalized histograms;
  `compare/2` gives the total-variation distance (0 = identical, 1 = disjoint)
  per histogram plus the scalar rates side by side.

  Action-state groups are coarse id ranges applied identically to both
  sources — the comparison is consistent even where a group boundary is
  approximate.
  """

  @buttons [a: :button_a, b: :button_b, x: :button_x, y: :button_y, z: :button_z, l: :button_l, r: :button_r]
  @run_edges [1, 2, 4, 8, 16, 32, 64]

  @type game :: [%{own: map(), opp: map(), controller: map()}]

  @doc "Raw counts for one game. `edge` = stage half-width (x of the ledge)."
  @spec from_game(game(), number()) :: map()
  def from_game([], _edge), do: empty()

  def from_game(frames, edge) do
    own = Enum.map(frames, & &1.own)
    opp = Enum.map(frames, & &1.opp)
    ctrl = Enum.map(frames, & &1.controller)
    acts = Enum.map(own, &(&1.action || 0))

    hists =
      %{}
      |> put_button_holds(ctrl)
      |> Map.put("stick_dwell", run_hist(Enum.map(ctrl, &stick_quant/1), fn _ -> true end))
      |> Map.put("stick_zone", freq(Enum.map(ctrl, &stick_zone/1)))
      |> Map.put("action_group", freq(Enum.map(acts, &action_group/1)))
      |> Map.put("position", freq(Enum.map(own, &position_bin(&1, edge))))
      |> Map.put("landing_lag", exact_run_hist(acts, &(&1 in 70..74), 30))
      |> Map.put("jump_peak", freq(jump_peaks(own, acts)))

    counts =
      %{"frames" => length(frames)}
      |> Map.merge(press_counts(ctrl))
      |> Map.merge(outcome_counts(own, opp, edge))
      |> Map.merge(tech_counts(acts))

    %{counts: counts, hists: hists}
  end

  def empty, do: %{counts: %{}, hists: %{}}

  @spec merge(map(), map()) :: map()
  def merge(a, b) do
    %{
      counts: Map.merge(a.counts, b.counts, fn _k, x, y -> x + y end),
      hists: Map.merge(a.hists, b.hists, fn _k, x, y -> Map.merge(x, y, fn _b, p, q -> p + q end) end)
    }
  end

  @doc "Per-minute rates, ratios, and normalized histograms."
  @spec summarize(map()) :: map()
  def summarize(%{counts: c, hists: h}) do
    minutes = max(Map.get(c, "frames", 0), 1) / 3600
    per_min = fn k -> Float.round(Map.get(c, k, 0) / minutes, 3) end
    ratio = fn n, d -> if d == 0, do: nil, else: Float.round(n / d, 3) end
    g = fn k -> Map.get(c, k, 0) end

    rates =
      %{
        "minutes" => Float.round(minutes, 1),
        "deaths_per_min" => per_min.("deaths"),
        "sd_per_min" => per_min.("sds"),
        "offstage_eps_per_min" => per_min.("off_eps"),
        "offstage_return_rate" => ratio.(g.("off_recovered"), g.("off_recovered") + g.("off_died")),
        "damage_dealt_per_min" => per_min.("dmg_dealt"),
        "damage_taken_per_min" => per_min.("dmg_taken"),
        "kills_per_min" => per_min.("kills"),
        "dashes_per_min" => per_min.("dashes"),
        "wavedashes_per_min" => per_min.("wavedashes"),
        "ground_jumps_per_min" => per_min.("ground_jumps"),
        "tech_rate" => ratio.(g.("techs"), g.("techs") + g.("missed_techs")),
        "tech_situations" => g.("techs") + g.("missed_techs"),
        "input_repeat_share" => ratio.(g.("input_repeats"), g.("frames")),
        "neutral_input_share" => ratio.(g.("neutral_inputs"), g.("frames"))
      }
      |> Map.merge(Map.new(@buttons, fn {b, _} -> {"press_#{b}_per_min", per_min.("press_#{b}")} end))

    %{rates: rates, hists: Map.new(h, fn {k, v} -> {k, normalize(v)} end)}
  end

  @doc """
  Total-variation distance per histogram between two summaries, plus
  `"hold_mean"` (mean over the seven button hold-length histograms).
  """
  @spec compare(map(), map()) :: map()
  def compare(%{hists: a}, %{hists: b}) do
    keys = (Map.keys(a) ++ Map.keys(b)) |> Enum.uniq()
    d = Map.new(keys, fn k -> {k, tv(Map.get(a, k, %{}), Map.get(b, k, %{}))} end)
    holds = for {b, _} <- @buttons, v = d["hold_#{b}"], v != nil, do: v
    if holds == [], do: d, else: Map.put(d, "hold_mean", Float.round(Enum.sum(holds) / length(holds), 3))
  end

  def tv(p, q) do
    keys = (Map.keys(p) ++ Map.keys(q)) |> Enum.uniq()
    Float.round(0.5 * Enum.sum(Enum.map(keys, fn k -> abs(Map.get(p, k, 0.0) - Map.get(q, k, 0.0)) end)), 3)
  end

  # ---- inputs ------------------------------------------------------------------

  defp put_button_holds(hists, ctrl) do
    Enum.reduce(@buttons, hists, fn {b, field}, acc ->
      Map.put(acc, "hold_#{b}", run_hist(Enum.map(ctrl, &(Map.get(&1, field) == true)), & &1))
    end)
  end

  defp press_counts(ctrl) do
    presses =
      Map.new(@buttons, fn {b, field} ->
        seq = Enum.map(ctrl, &(Map.get(&1, field) == true))
        {"press_#{b}", seq |> Enum.chunk_every(2, 1, :discard) |> Enum.count(fn [p, q] -> q and not p end)}
      end)

    keys = Enum.map(ctrl, &input_key/1)

    presses
    |> Map.put("input_repeats", Enum.zip(keys, tl(keys)) |> Enum.count(fn {p, q} -> p == q end))
    |> Map.put("neutral_inputs", Enum.count(ctrl, &neutral?/1))
  end

  defp input_key(c) do
    {for({_, f} <- @buttons, do: Map.get(c, f) == true), stick_quant(c), round(c.c_stick.x * 16), round(c.c_stick.y * 16)}
  end

  defp neutral?(c) do
    not Enum.any?(@buttons, fn {_, f} -> Map.get(c, f) == true end) and stick_zone(c) == "neutral"
  end

  # main stick on the 17-bucket grid (the policy's output resolution)
  defp stick_quant(c), do: {round(c.main_stick.x * 16), round(c.main_stick.y * 16)}

  defp stick_zone(c) do
    dx = c.main_stick.x - 0.5
    dy = c.main_stick.y - 0.5

    if :math.sqrt(dx * dx + dy * dy) < 0.14 do
      "neutral"
    else
      oct = round(:math.atan2(dy, dx) / (:math.pi() / 4))
      Enum.at(~w(left down_left down down_right right up_right up up_left left), oct + 4)
    end
  end

  # ---- outcomes ----------------------------------------------------------------

  defp offstage?(p, edge), do: not (p.on_ground == true) and (abs(p.x || 0.0) > edge or (p.y || 0.0) < -5.0)

  defp outcome_counts(own, opp, edge) do
    init = %{deaths: 0, sds: 0, kills: 0, off_eps: 0, off_recovered: 0, off_died: 0, dmg_dealt: 0.0, dmg_taken: 0.0,
             last_hit: -10_000, off: false}

    pairs = Enum.zip([own, tl(own), opp, tl(opp)]) |> Enum.with_index()

    r =
      Enum.reduce(pairs, init, fn {{p0, p1, o0, o1}, t}, a ->
        taken = max(0.0, ((p1.percent || 0) - (p0.percent || 0)) * 1.0)
        last_hit = if taken > 0, do: t, else: a.last_hit
        died? = (p1.stock || 0) < (p0.stock || 0)
        now_off = offstage?(p1, edge) and not died?

        %{a |
          deaths: a.deaths + b2i(died?),
          sds: a.sds + b2i(died? and t - a.last_hit > 90),
          kills: a.kills + b2i((o1.stock || 0) < (o0.stock || 0)),
          dmg_taken: a.dmg_taken + taken,
          dmg_dealt: a.dmg_dealt + max(0.0, ((o1.percent || 0) - (o0.percent || 0)) * 1.0),
          off_eps: a.off_eps + b2i(now_off and not a.off),
          off_recovered: a.off_recovered + b2i(a.off and not now_off and not died?),
          off_died: a.off_died + b2i(a.off and died?),
          last_hit: last_hit,
          off: now_off}
      end)

    r |> Map.drop([:last_hit, :off]) |> Map.new(fn {k, v} -> {Atom.to_string(k), v} end)
  end

  defp position_bin(p, edge) do
    cond do
      (p.action || 99) in 0..13 -> "dead"
      offstage?(p, edge) and (p.y || 0.0) < -5.0 -> "off_below"
      offstage?(p, edge) -> "off_side"
      true -> "#{if p.on_ground == true, do: "ground", else: "air"}_#{min(trunc(abs(p.x || 0.0) / edge * 4), 3)}"
    end
  end

  # ---- technique ---------------------------------------------------------------

  defp tech_counts(acts) do
    trans = Enum.zip(acts, tl(acts))
    entered = fn pred -> Enum.count(trans, fn {p, q} -> pred.(q) and not pred.(p) end) end
    runs = acts |> Enum.chunk_by(& &1) |> Enum.map(&{hd(&1), length(&1)})

    wavedashes =
      runs
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.count(fn [{a, n}, {b, _}] -> a == 236 and n <= 4 and b == 43 end)

    %{
      "techs" => entered.(&(&1 in 199..201)),
      "missed_techs" => entered.(&(&1 in [183, 191])),
      "dashes" => entered.(&(&1 == 20)),
      "ground_jumps" => entered.(&(&1 == 24)),
      "wavedashes" => wavedashes
    }
  end

  # Peak height above takeoff for every ground jump (jumpsquat 24 -> 25/26),
  # followed until the player is grounded again; 5-unit bins. Short hops and
  # full hops separate into two modes.
  defp jump_peaks(own, acts) do
    t = List.to_tuple(own)
    a = List.to_tuple(acts)
    n = tuple_size(t)

    for i <- 1..(n - 1)//1, elem(a, i) in [25, 26] and elem(a, i - 1) == 24 do
      y0 = elem(t, i - 1).y || 0.0

      peak =
        Enum.reduce_while(i..min(i + 90, n - 1), y0, fn j, m ->
          p = elem(t, j)
          if j > i and p.on_ground == true, do: {:halt, m}, else: {:cont, max(m, p.y || 0.0)}
        end)

      "#{min(trunc((peak - y0) / 5) * 5, 60)}"
    end
  end

  # ---- helpers -----------------------------------------------------------------

  # histogram of run lengths (bucketed) of consecutive equal values that satisfy `keep`
  defp run_hist(seq, keep) do
    seq
    |> Enum.chunk_by(& &1)
    |> Enum.filter(fn [v | _] -> keep.(v) end)
    |> Enum.map(&run_bucket(length(&1)))
    |> freq()
  end

  defp exact_run_hist(seq, pred, cap) do
    seq
    |> Enum.chunk_by(pred)
    |> Enum.filter(fn [v | _] -> pred.(v) end)
    |> Enum.map(&"#{min(length(&1), cap)}")
    |> freq()
  end

  defp run_bucket(n) do
    case Enum.find(@run_edges, &(n <= &1)) do
      nil -> "65+"
      1 -> "1"
      2 -> "2"
      e -> "#{div(e, 2) + 1}-#{e}"
    end
  end

  defp freq(list), do: Enum.frequencies(list)

  defp normalize(h) do
    total = h |> Map.values() |> Enum.sum()
    if total == 0, do: %{}, else: Map.new(h, fn {k, v} -> {k, Float.round(v / total, 4)} end)
  end

  defp b2i(true), do: 1
  defp b2i(_), do: 0

  @doc "Coarse action-state group for a Melee action id."
  def action_group(a) do
    cond do
      a in 0..13 -> "dead_respawn"
      a == 14 -> "stand"
      a in 15..19 -> "walk_turn"
      a in 20..23 -> "dash_run"
      a == 24 -> "jumpsquat"
      a in 25..28 -> "jump"
      a in 29..38 -> "fall"
      a in 39..41 -> "crouch"
      a in 42..43 -> "land"
      a in 44..49 -> "jab"
      a == 50 -> "dash_attack"
      a in 51..56 -> "tilt"
      a in 57..64 -> "smash"
      a in 65..69 -> "aerial"
      a in 70..74 -> "aerial_land"
      a in 75..91 -> "hitstun"
      a in 178..182 -> "shield"
      a in 183..198 -> "knockdown"
      a in 199..204 -> "tech"
      a in 212..232 -> "grab_throw"
      a in 233..236 -> "dodge_roll"
      a in 252..263 -> "ledge"
      a >= 341 -> "special"
      true -> "other"
    end
  end
end
