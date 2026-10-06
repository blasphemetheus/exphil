defmodule ExPhil.Eval.SilenceMap do
  @moduledoc """
  Where does the player let go — the input-change hazard by situation
  (2026-10-06, INPUT_COHERENCE doc "10-06").

  The silent fall (the bot releases a deflected stick offstage and dies)
  turned out to be one symptom of a general defect: the hold/change
  decision barely reads the game state, so the bot's silence budget is
  about the expert's overall but spent in the wrong states. This module
  measures that directly. For every frame `t` with a successor in the same
  game it classifies the input at `t` (active / silent) and the transition
  to `t + 1`:

    * active → **enter_silence** (all-neutral controller next frame)
    * active → **change** (still active, but a different stick zone or a
      button edge)
    * active → hold
    * silent → **resume** (any input next frame)

  and accumulates the counts in two bucket families:

    * `state`: universal physical bins computed here (grounded centre / edge,
      airborne onstage, offstage high / low / deep × jumps left, ledge hang,
      hitstun) — character-agnostic, no label dependency;
    * `situation`: every `ExPhil.Situations` label the frame carries (a frame
      counts in each of its labels);
    * `age`: onstage / offstage × how long the current input has been held
      (the hazard-by-age curve a semi-Markov stick head would learn).

  Same input shape as `ExPhil.Eval.PlayStats` (a game = a list of
  `%{own: Player, opp: Player, controller: ControllerState}`), so the fidelity
  rollout and the expert-replay pass feed it without a second producer.
  `summarize/1` turns counts into hazards; `compare/2` puts model and expert
  side by side with the ratio and a binomial standard error, worst first.
  """

  alias ExPhil.Bridge.GameState
  alias ExPhil.Training.SilentFallWeighting

  @buttons [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r]
  @cliff 252..263
  @first_actionable 14
  @edge_margin 20.0
  @low_y -60.0
  @keys ~w(frames active enter_silence change silent resume)

  @type game :: [%{own: map(), opp: map(), controller: map()}]

  @doc "Raw counts for one game. `opts`: `:stage` (id, default 32) and `:edge` (half-width)."
  @spec from_game(game(), keyword()) :: map()
  def from_game(frames, opts \\ [])
  def from_game([], _opts), do: empty()

  def from_game(frames, opts) do
    stage = Keyword.get(opts, :stage, 32)
    edge = Keyword.get(opts, :edge, 85.5656967163)

    {labels, _ctx} =
      frames
      |> Enum.with_index()
      |> Enum.map_reduce(ExPhil.Situations.new_context(), fn {f, i}, ctx ->
        gs = %GameState{frame: i, stage: stage, players: %{1 => f.own, 2 => f.opp}}
        {ctx, set} = ExPhil.Situations.fold(ctx, gs, 1)
        {set, ctx}
      end)

    # age = frames the current input (zone + buttons) has been held so far
    {ages, _} =
      frames
      |> Enum.chunk_every(2, 1)
      |> Enum.map_reduce(1, fn
        [f, g], age -> {age, if(same_input?(f.controller, g.controller), do: age + 1, else: 1)}
        [_last], age -> {age, age}
      end)

    [frames, labels, ages]
    |> Enum.zip()
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.reduce(empty(), fn [{f, set, age}, {g, _, _}], acc ->
      if countable?(f.own) and countable?(g.own) do
        outcome = transition(f.controller, g.controller)
        sb = state_bin(f.own, edge)
        side = if String.starts_with?(sb, "offstage"), do: "offstage", else: "onstage"

        buckets =
          [{"state", sb}, {"age", "#{side}:#{age_bin(age)}"} | Enum.map(set, &{"situation", Atom.to_string(&1)})]

        Enum.reduce(buckets, acc, fn b, a -> bump(a, b, outcome) end)
      else
        acc
      end
    end)
  end

  @doc "How long the current input has been held, binned."
  @spec age_bin(pos_integer()) :: String.t()
  def age_bin(a) when a <= 3, do: "a01-03"
  def age_bin(a) when a <= 7, do: "a04-07"
  def age_bin(a) when a <= 15, do: "a08-15"
  def age_bin(a) when a <= 31, do: "a16-31"
  def age_bin(_), do: "a32+"

  def empty, do: %{}

  @spec merge(map(), map()) :: map()
  def merge(a, b), do: Map.merge(a, b, fn _k, x, y -> Map.merge(x, y, fn _c, p, q -> p + q end) end)

  @doc "Hazards per bucket: enter_silence / change per active frame, resume per silent frame."
  @spec summarize(map()) :: map()
  def summarize(counts) do
    Map.new(counts, fn {{fam, name}, c} ->
      act = c["active"] || 0
      sil = c["silent"] || 0

      {"#{fam}:#{name}",
       %{
         frames: c["frames"] || 0,
         active: act,
         silent: sil,
         enter_silence: rate(c["enter_silence"], act),
         change: rate(c["change"], act),
         resume: rate(c["resume"], sil)
       }}
    end)
  end

  @doc """
  Model vs expert per bucket for one hazard (default `:enter_silence`): the
  two hazards, the ratio, the binomial SE of the model's hazard, and `z` =
  (model − expert) / SE. Buckets with fewer than `min_n` model frames in
  the denominator are dropped; sorted by ratio, worst first.
  """
  @spec compare(map(), map(), keyword()) :: [map()]
  def compare(model, expert, opts \\ []) do
    hazard = Keyword.get(opts, :hazard, :enter_silence)
    min_n = Keyword.get(opts, :min_n, 200)
    denom = if hazard == :resume, do: :silent, else: :active

    model
    |> Enum.flat_map(fn {k, m} ->
      e = expert[k]
      n = m[denom]
      p = m[hazard]
      q = e && e[hazard]

      if e == nil or n < min_n or p == nil or q == nil do
        []
      else
        se = :math.sqrt(max(p * (1 - p), 1.0e-9) / n)
        ratio = if q > 0, do: p / q, else: if(p > 0, do: :infinity, else: 1.0)

        [%{bucket: k, n: n, expert_n: e[denom], model: p, expert: q,
           ratio: round_ratio(ratio), z: Float.round((p - q) / se, 1)}]
      end
    end)
    |> Enum.sort_by(&sort_key(&1.ratio), :desc)
  end

  @doc "Universal physical bin for one player state."
  @spec state_bin(map(), number()) :: String.t()
  def state_bin(p, edge) do
    act = p.action || 0
    x = abs(p.x || 0.0)
    y = p.y || 0.0
    jumps = if (p.jumps_left || 0) > 0, do: "j1+", else: "j0"

    cond do
      act in @cliff -> "ledge_hang"
      (p.hitstun_frames_left || 0) > 0 -> "hitstun"
      p.on_ground and x > edge - @edge_margin -> "grounded_edge"
      p.on_ground -> "grounded_center"
      x <= edge -> "airborne_onstage"
      y >= 0 -> "offstage_high_#{jumps}"
      y > @low_y -> "offstage_low_#{jumps}"
      true -> "offstage_deep_#{jumps}"
    end
  end

  @doc "Transition class from the input at `t` to the input at `t + 1`."
  @spec transition(map(), map()) :: :enter_silence | :change | :hold | :resume | :stay
  def transition(c, n) do
    case {SilentFallWeighting.neutral?(c), SilentFallWeighting.neutral?(n)} do
      {false, true} -> :enter_silence
      {true, false} -> :resume
      {true, true} -> :stay
      {false, false} -> if same_input?(c, n), do: :hold, else: :change
    end
  end

  # -- internals -------------------------------------------------------------

  defp bump(acc, bucket, outcome) do
    c = Map.get(acc, bucket, Map.new(@keys, &{&1, 0}))
    c = Map.update!(c, "frames", &(&1 + 1))

    c =
      case outcome do
        :enter_silence -> c |> inc("active") |> inc("enter_silence")
        :change -> c |> inc("active") |> inc("change")
        :hold -> inc(c, "active")
        :resume -> c |> inc("silent") |> inc("resume")
        :stay -> inc(c, "silent")
      end

    Map.put(acc, bucket, c)
  end

  defp inc(c, k), do: Map.update!(c, k, &(&1 + 1))

  defp countable?(nil), do: false
  defp countable?(p), do: (p.action || 0) >= @first_actionable

  defp same_input?(c, n) do
    zone(c) == zone(n) and Enum.all?(@buttons, &(truthy(Map.get(c, &1)) == truthy(Map.get(n, &1))))
  end

  defp truthy(v), do: v == true or (is_number(v) and v > 0)

  # 8 octants + centre, same radius as PlayStats.stick_zone
  defp zone(c) do
    ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
    dx = (ms[:x] || 0.5) - 0.5
    dy = (ms[:y] || 0.5) - 0.5
    if :math.sqrt(dx * dx + dy * dy) < 0.14,
      do: :neutral,
      else: rem(round(:math.atan2(dy, dx) / (:math.pi() / 4)) + 8, 8)
  end

  defp rate(_num, 0), do: nil
  defp rate(nil, _), do: nil
  defp rate(num, den), do: Float.round(num / den, 4)

  defp round_ratio(:infinity), do: :infinity
  defp round_ratio(r), do: Float.round(r * 1.0, 2)

  defp sort_key(:infinity), do: 1.0e9
  defp sort_key(r), do: r
end
