defmodule ExPhil.Interp.LoopStats do
  @moduledoc """
  Repetition / taunt pathology metrics — the things a HUMAN flags that
  `coach_report` never scored.

  Motivation (2026-08-28): two decode brackets returned null because the
  scored metric (armed approaches/min) is a rare, bursty event — the
  baseline arm moved 7x across days at n=5 (`eval_runs/0828_buttons_temp/
  RESULTS.md`). Meanwhile the actual human complaints from the 08-28 play
  session — "taunts in neutral", "laser/shield-grab/pummel loops",
  "dithering" — were never measured at all.

  Every metric here is **frequent per game** by construction (button
  presses and frame-runs, not conversions), which is what gives it the
  statistical power the approach-rate metric lacks.

  ## The three pathologies

  * **Taunts.** The bot's only taunt-capable button is `d_up` (the rarest
    of the 8 legal buttons). We count both the EXECUTED taunt (action
    states 264/265 = `TAUNT_RIGHT`/`TAUNT_LEFT`) and the d_up PRESS
    (rising edge). Presses are far more numerous — most are eaten
    mid-action — so `dpad_per_min` is the higher-power signal for
    decode-arm comparisons, and `taunts_per_min` is the ground truth of
    what a human actually sees.
  * **Streaks.** Runs of an identical action state (holding one thing),
    and runs of a literally identical controller input (dithering /
    frozen decode). `long_frac` — the fraction of frames spent inside
    runs at or over the threshold — is the headline: it is a *fraction*,
    so it is bounded, dense, and comparable across games of any length.
  * **Cycles.** Repeated action n-grams — "laser, laser, laser" or
    "grab, pummel, grab, pummel". Detected over the TRANSITION sequence
    (consecutive duplicates collapsed) so that frame duration doesn't
    disguise a two-state loop as two long streaks.

  All entry points take a `.slp` path or pre-extracted lists, mirroring
  `ExPhil.Interp.ReplayStats`.
  """

  alias ExPhil.Data.Peppi

  # libmelee enums.py: TAUNT_RIGHT = 0x108, TAUNT_LEFT = 0x109
  @taunt_states MapSet.new([264, 265])

  @fps 60.0

  @legal_buttons [
    :button_a,
    :button_b,
    :button_x,
    :button_y,
    :button_z,
    :button_l,
    :button_r,
    :button_d_up
  ]

  # ---- loading ---------------------------------------------------------------

  @doc """
  Parse a replay into the bot's own action + controller streams.

  Unlike `ReplayStats.load/1` (which is port-1 canonical for controllers),
  this extracts frames with `player_port: bot_port`, so `:controllers` are
  always the BOT's inputs — `Peppi` only fills `frame.controller` for the
  requested player port (`peppi.ex:342`); per-port `controller_state`
  inside `game_state.players` is nil.

  Returns `%{actions: [int], controllers: [ControllerState | nil], n: int}`.
  """
  def load(path, opts \\ []) do
    case safe_load(path, opts) do
      {:ok, data} -> data
      {:error, reason} -> raise "LoopStats.load failed for #{path}: #{inspect(reason)}"
    end
  end

  @doc """
  Like `load/2` but returns `{:error, reason}` instead of raising on an
  unreadable replay.

  Truncated replays are COMMON and their loss is BIASED: the live-eval
  protocol copies the newest `.slp` out of `~/Slippi` without waiting for
  Dolphin to finalize it (`eval_live_protocol.sh:155-157`), so a run whose
  game ended early — i.e. a run where the bot DIED — is the most likely to
  produce a truncated, unparseable file. A scorer that crashes on the first
  bad replay silently makes the worst arm the least measurable one, so
  every batch entry point here skips and COUNTS them instead.
  """
  def safe_load(path, opts \\ []) do
    bot_port = Keyword.get(opts, :bot_port, 1)
    opp_port = if bot_port == 1, do: 2, else: 1

    with {:ok, replay} <- Peppi.parse(Path.expand(path)) do
      {:ok, build(replay, bot_port, opp_port)}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp build(replay, bot_port, opp_port) do
    frames =
      replay
      |> Peppi.to_training_frames(player_port: bot_port, opponent_port: opp_port)
      |> Enum.reject(&(&1.game_state.frame < 0))

    actions =
      Enum.map(frames, fn f ->
        case f.game_state.players[bot_port] do
          %{action: a} when is_number(a) -> trunc(a)
          _ -> 0
        end
      end)

    %{
      actions: actions,
      controllers: Enum.map(frames, & &1.controller),
      n: length(frames),
      bot_port: bot_port
    }
  end

  # ---- taunts ----------------------------------------------------------------

  @doc """
  Executed taunts: entries into (and frames spent in) action states
  264/265.
  """
  def taunt_stats(actions) do
    %{
      entries: count_entries(actions, &MapSet.member?(@taunt_states, &1)),
      frames: Enum.count(actions, &MapSet.member?(@taunt_states, &1))
    }
  end

  @doc """
  D-pad-up presses: rising edges on `button_d_up`, the taunt input.

  This is the DECODE-side cause of taunting; `taunt_stats/1` is the
  in-game effect. Presses outnumber executed taunts (most are eaten
  mid-action), which is exactly why this is the arm-separating metric.
  """
  def dpad_stats(controllers) do
    pressed = Enum.map(controllers, fn c -> !!(c && c.button_d_up) end)

    %{
      presses: count_entries(pressed, & &1),
      frames: Enum.count(pressed, & &1)
    }
  end

  # ---- streaks ---------------------------------------------------------------

  @doc """
  Runs of an identical action state.

  `:min_long` (default 60 frames = 1s) is the "held too long" threshold.
  """
  def action_streaks(actions, opts \\ []) do
    streaks(actions, Keyword.get(opts, :min_long, 60))
  end

  @doc """
  Runs of a literally identical controller input (quantized buttons +
  sticks).

  `:min_long` defaults to 10 frames. Stochastic decode resamples the
  sticks every frame, so a bit-identical input surviving even 10 frames
  already means the decode has locked; a 30-frame threshold never fires
  under sampling and reads as a constant 0.0.
  """
  def input_streaks(controllers, opts \\ []) do
    controllers
    |> Enum.map(&quantize/1)
    |> streaks(Keyword.get(opts, :min_long, 10))
  end

  defp streaks(values, min_long) do
    runs =
      values
      |> Enum.chunk_by(& &1)
      |> Enum.map(fn [h | _] = run -> {h, length(run)} end)

    lens = Enum.map(runs, &elem(&1, 1))
    long = Enum.filter(runs, fn {_, l} -> l >= min_long end)
    total = Enum.sum(lens)

    %{
      runs: length(runs),
      max: Enum.max(lens, fn -> 0 end),
      median: percentile(lens, 50),
      p90: percentile(lens, 90),
      long_runs: length(long),
      long_frac: safe_div(Enum.sum(Enum.map(long, &elem(&1, 1))), total),
      top: long |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(5),
      min_long: min_long
    }
  end

  # Collapse a controller into a comparable fingerprint. Sticks are
  # bucketed at 1/8 so that analog jitter below human perception doesn't
  # break a run that a human would call "frozen".
  defp quantize(nil), do: :none

  defp quantize(c) do
    buttons = for k <- @legal_buttons, do: if(Map.get(c, k), do: 1, else: 0)

    {buttons, stick_bucket(c.main_stick), stick_bucket(c.c_stick),
     round((c.l_shoulder || 0.0) * 4)}
  end

  defp stick_bucket(%{x: x, y: y}) when is_number(x) and is_number(y),
    do: {round(x * 8), round(y * 8)}

  defp stick_bucket(_), do: {4, 4}

  # ---- cycles ----------------------------------------------------------------

  @doc """
  Repeated action cycles over the TRANSITION sequence.

  Collapses consecutive duplicate actions, then finds places where a
  window of `period` states repeats back-to-back at least `:min_repeats`
  times. Catches "laser, laser, laser" (period 2 with the intervening
  Wait) and "grab, pummel, grab, pummel" (period 2+) — loops that
  per-frame streak metrics miss because no single state is held long.

  Options: `:min_repeats` (default 3), `:max_period` (default 6).
  Returns `%{episodes: [%{pattern:, period:, repeats:}], count:, max_repeats:}`.
  """
  def action_cycles(actions, opts \\ []) do
    min_repeats = Keyword.get(opts, :min_repeats, 3)
    max_period = Keyword.get(opts, :max_period, 6)

    seq =
      actions
      |> Enum.chunk_by(& &1)
      |> Enum.map(&hd/1)
      |> List.to_tuple()

    episodes = scan_cycles(seq, 0, tuple_size(seq), min_repeats, max_period, [])

    %{
      episodes: episodes,
      count: length(episodes),
      max_repeats: episodes |> Enum.map(& &1.repeats) |> Enum.max(fn -> 0 end)
    }
  end

  defp scan_cycles(_seq, i, n, _min_r, _max_p, acc) when i >= n, do: Enum.reverse(acc)

  defp scan_cycles(seq, i, n, min_r, max_p, acc) do
    best =
      2..max_p
      |> Enum.map(fn p -> {p, repeats_at(seq, i, p, n)} end)
      |> Enum.filter(fn {_p, r} -> r >= min_r end)
      # Prefer the most repeats; tie-break to the SHORTEST period, so a
      # 2-cycle isn't reported as its own 4-cycle.
      |> Enum.max_by(fn {p, r} -> {r, -p} end, fn -> nil end)

    case best do
      nil ->
        scan_cycles(seq, i + 1, n, min_r, max_p, acc)

      {p, r} ->
        pattern = for k <- 0..(p - 1), do: elem(seq, i + k)
        ep = %{pattern: pattern, period: p, repeats: r}
        scan_cycles(seq, i + p * r, n, min_r, max_p, [ep | acc])
    end
  end

  # How many times does the p-window at i repeat back-to-back?
  defp repeats_at(seq, i, p, n) do
    max_r = div(n - i, p)

    Enum.reduce_while(1..max(max_r, 1)//1, 0, fn r, _acc ->
      if r <= max_r and window_equal?(seq, i, i + (r - 1) * p, p) do
        {:cont, r}
      else
        {:halt, r - 1}
      end
    end)
  end

  defp window_equal?(seq, a, b, p) do
    Enum.all?(0..(p - 1)//1, fn k -> elem(seq, a + k) == elem(seq, b + k) end)
  end

  # ---- report ----------------------------------------------------------------

  @doc """
  Full per-game report. Accepts a path or a `load/2` map.

  Rates are per minute of ACTUAL scored frames, so short games and
  timeouts stay comparable.
  """
  def report(path, opts \\ [])

  def report(path, opts) when is_binary(path) do
    path |> load(opts) |> report(opts)
  end

  def report(%{actions: actions, controllers: controllers, n: n}, opts) do
    minutes = max(n / @fps / 60.0, 1.0e-9)

    taunts = taunt_stats(actions)
    dpad = dpad_stats(controllers)
    astreaks = action_streaks(actions, opts)
    istreaks = input_streaks(controllers, opts)
    cycles = action_cycles(actions, opts)

    %{
      frames: n,
      minutes: minutes,
      taunts: taunts,
      dpad: dpad,
      action_streaks: astreaks,
      input_streaks: istreaks,
      cycles: cycles,
      summary: %{
        taunts_per_min: taunts.entries / minutes,
        dpad_per_min: dpad.presses / minutes,
        action_long_frac: astreaks.long_frac,
        input_long_frac: istreaks.long_frac,
        longest_action_run: astreaks.max,
        longest_input_run: istreaks.max,
        loops_per_min: cycles.count / minutes,
        max_loop_repeats: cycles.max_repeats
      }
    }
  end

  @doc """
  Aggregate a list of per-game reports into mean / median / min / max per
  summary key.

  The RANGE is reported deliberately: the project's standing law is that
  differences under 2x are unresolved, and the 08-28 sweep showed a
  baseline moving 7x across days. A mean without its range is how that
  went unnoticed.
  """
  def aggregate([]), do: %{n: 0, stats: %{}}

  def aggregate(reports) do
    keys = reports |> hd() |> Map.fetch!(:summary) |> Map.keys()

    stats =
      Map.new(keys, fn k ->
        vals = Enum.map(reports, &get_in(&1, [:summary, k]))

        {k,
         %{
           mean: Enum.sum(vals) / length(vals),
           median: percentile(vals, 50),
           min: Enum.min(vals),
           max: Enum.max(vals),
           values: vals
         }}
      end)

    %{n: length(reports), stats: stats}
  end

  # ---- helpers ---------------------------------------------------------------

  # Count entries into a predicate-satisfying region (rising edges),
  # counting a leading hit as one entry.
  defp count_entries([], _pred), do: 0

  defp count_entries([first | _] = values, pred) do
    lead = if pred.(first), do: 1, else: 0

    transitions =
      values
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.count(fn [a, b] -> not pred.(a) and pred.(b) end)

    lead + transitions
  end

  defp percentile([], _p), do: 0

  defp percentile(list, p) do
    sorted = Enum.sort(list)
    idx = min(round(p / 100 * length(sorted)), length(sorted) - 1)
    Enum.at(sorted, max(idx, 0))
  end

  defp safe_div(_a, 0), do: 0.0
  defp safe_div(a, b), do: a / b
end
