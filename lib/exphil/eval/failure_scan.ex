defmodule ExPhil.Eval.FailureScan do
  @moduledoc """
  Outcome detectors that flag training-data gaps from the bot's own games
  (DATA_FLYWHEEL_DESIGN_2026-07-23, stage A1).

  Where `ExPhil.Eval.ScenarioScan` finds *situations* (well-posed "what does
  the policy do next?" moments), `FailureScan` finds *outcomes that reveal a
  gap* — got opened up in neutral, opened the opponent but dropped the
  punish, died off a neutral loss, sat in range without going in. Run it
  over every probe game / exhibition / sparring set; each hit becomes a
  drill handoff and a `GapLedger` entry.

  Same contract as ScenarioScan so the two compose:
  - consumes the slim frame shape from `ScenarioScan.load/1`:
    `[%{frame:, p1:, p2:}]` where each player is `ScenarioScan.player_summary/1`
    (`%{x, y, action, facing, on_ground, stock, percent}`),
  - detectors are pure functions over that list (unit-tested with synthetic
    sequences),
  - `scan/2` curates: min_frame filter + same-type spacing,
  - returns `[%{type:, frame:, note:}]` where `frame` is the DRILL HANDOFF
    (already backed off from the outcome), sorted by frame.

  Port convention: **P1 = the bot** (the subject whose gaps we want). To
  scan the bot when it played P2 in a replay, pass frames through `flip/1`
  first. Action-state ID sets mirror `ExPhil.Interp.ReplayStats` so numbers
  stay comparable across the interp toolkit.
  """

  alias ExPhil.Eval.ScenarioScan

  @types [:neutral_loss, :dropped_punish, :death_sequence, :passivity_window]

  # Mirrors ReplayStats / ScenarioScan.
  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @shield_states MapSet.new([178, 179, 180])
  @attack_states MapSet.new(Enum.to_list(44..69) ++ [212, 214])

  # "Not neutral" = anyone mid-interaction (hitstun / knockdown lifecycle /
  # shielding). Used to require a clean neutral lead-in before an event.
  @non_neutral MapSet.union(@hitstun, MapSet.union(@lifecycle, @shield_states))

  # Tunables (per-call overridable via opts on scan/2's detectors).
  @neutral_lookback 60
  @handoff_backoff 90
  @punish_window 120
  @punish_min_gain 15.0
  @death_neutral_run 30
  @passivity_close_dist 40.0
  @passivity_min_run 300

  @default_min_frame 300
  @default_gap 240

  def types, do: @types

  @doc """
  Load a replay into the slim frame shape (delegates to `ScenarioScan.load/1`
  which already carries percent + stock). Returns `{:ok, frames}` |
  `{:error, reason}`.
  """
  def load(path) do
    case ScenarioScan.load(path) do
      {:ok, %{frames: frames}} -> {:ok, frames}
      other -> other
    end
  end

  @doc """
  Swap p1/p2 in every frame — scan the bot when it played port 2.
  """
  def flip(frames), do: Enum.map(frames, fn f -> %{f | p1: f.p2, p2: f.p1} end)

  @doc """
  Run every detector and curate: keep handoffs >= `:min_frame`, drop
  same-type candidates within `:gap` frames. Returns `[%{type:, frame:, note:}]`.

  ## Options
    - `:min_frame` (default #{@default_min_frame})
    - `:gap` (default #{@default_gap})
    - `:types` — subset of `types/0`
  """
  def scan(frames, opts \\ []) do
    min_frame = Keyword.get(opts, :min_frame, @default_min_frame)
    gap = Keyword.get(opts, :gap, @default_gap)
    wanted = Keyword.get(opts, :types, @types)

    wanted
    |> Enum.flat_map(fn type -> detect(type, frames) end)
    |> Enum.filter(fn c -> c.frame >= min_frame end)
    |> Enum.group_by(& &1.type)
    |> Enum.flat_map(fn {_t, cands} -> space_out(Enum.sort_by(cands, & &1.frame), gap) end)
    |> Enum.sort_by(& &1.frame)
  end

  @doc "Run a single detector (uncurated)."
  def detect(:neutral_loss, frames), do: neutral_loss(frames)
  def detect(:dropped_punish, frames), do: dropped_punish(frames)
  def detect(:death_sequence, frames), do: death_sequence(frames)
  def detect(:passivity_window, frames), do: passivity_window(frames)

  # ==========================================================================
  # Detectors
  # ==========================================================================

  @doc """
  P1 (the bot) enters hitstun straight out of true neutral: for the
  `lookback` (default #{@neutral_lookback}) frames before the hit NEITHER
  player was in a non-neutral state, then P1's action enters hitstun. The
  bot lost the neutral exchange — the gap is whatever it did (or failed to
  do) to get hit. Handoff = hit − #{@handoff_backoff} (the opening began
  before the hit).
  """
  def neutral_loss(frames, lookback \\ @neutral_lookback) do
    arr = index(frames)

    hit_edges(frames, :p1)
    |> Enum.filter(fn {i, _f} -> neutral_before?(arr, i, lookback) end)
    |> Enum.map(fn {i, f} ->
      opener = f.p2.action
      j = max(i - @handoff_backoff, 0)
      dist_then = Float.round(abs(elem(arr[j], 1).p2.x - elem(arr[j], 1).p1.x), 1)

      %{
        type: :neutral_loss,
        frame: elem(arr[j], 1).frame,
        note: "hit@#{f.frame} opener=p2_action#{opener} dist_#{@handoff_backoff}f_prior=#{dist_then}"
      }
    end)
  end

  @doc """
  P1 opens P2 up from true neutral (P2 enters hitstun with a clean neutral
  lead-in — symmetric to `neutral_loss`) but the punish fizzles: within the
  next `window` (default #{@punish_window}) frames P2 returns to an
  actionable state without taking another hit, and P2's percent gain over
  the window is < #{trunc(@punish_min_gain)}. This is gate-10's sibling —
  initiation that goes unrewarded. Handoff = the opening hit (drill the
  conversion from there).
  """
  def dropped_punish(frames, window \\ @punish_window) do
    arr = index(frames)
    n = map_size(arr)

    hit_edges(frames, :p2)
    |> Enum.filter(fn {i, _f} -> neutral_before?(arr, i, @neutral_lookback) end)
    |> Enum.filter(fn {i, f} -> punish_dropped?(arr, i, n, f, window) end)
    |> Enum.map(fn {_i, f} ->
      %{
        type: :dropped_punish,
        frame: f.frame,
        note: "opening@#{f.frame} start_pct=#{Float.round(f.p2.percent, 1)} window=#{window}f"
      }
    end)
  end

  @doc """
  P1 loses a stock. Walk back from the death to the most recent frame where
  both players were neutral for `neutral_run` (default #{@death_neutral_run})
  consecutive frames; the frame the neutral broke is the handoff — drill the
  entry-point of the sequence that killed, not the death itself.
  """
  def death_sequence(frames, neutral_run \\ @death_neutral_run) do
    arr = index(frames)

    stock_losses(frames, :p1)
    |> Enum.map(fn {i, death_f} ->
      case last_neutral_break(arr, i, neutral_run) do
        nil ->
          nil

        {bj, break_f} ->
          %{
            type: :death_sequence,
            frame: break_f.frame,
            note:
              "death@#{death_f.frame} elapsed=#{death_f.frame - break_f.frame}f " <>
                "opener=p2_action#{elem(arr[min(bj + 1, map_size(arr) - 1)], 1).p2.action}"
          }
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  @doc """
  A maximal run of >= `min_run` (default #{@passivity_min_run}) consecutive
  frames where the players are within #{trunc(@passivity_close_dist)}
  x-units AND P1 never enters an attack/grab state. In range, never going
  in — the gate-10 pathology localized to a drillable moment. Handoff = run
  start.

  NOTE: the design doc phrased this as "300 close frames within a 600-frame
  window"; this uses a stricter *consecutive* run (cheaper, cleaner, and a
  contiguous span is exactly what a drill wants). Documented deviation.
  """
  def passivity_window(frames, min_run \\ @passivity_min_run) do
    flags =
      Enum.map(frames, fn f ->
        close = abs(f.p2.x - f.p1.x) <= @passivity_close_dist
        passive = not MapSet.member?(@attack_states, f.p1.action)
        close and passive
      end)

    frames_arr = List.to_tuple(frames)

    flags
    |> Enum.with_index()
    |> Enum.chunk_by(fn {flag, _} -> flag end)
    |> Enum.filter(fn [{flag, _} | _] = chunk -> flag and length(chunk) >= min_run end)
    |> Enum.map(fn [{_, start} | _] = chunk ->
      f = elem(frames_arr, start)
      run = length(chunk)
      mid = elem(frames_arr, start + div(run, 2))

      %{
        type: :passivity_window,
        frame: f.frame,
        note: "passive_run=#{run}f mid_dist=#{Float.round(abs(mid.p2.x - mid.p1.x), 1)}"
      }
    end)
  end

  # ==========================================================================
  # Helpers
  # ==========================================================================

  # Index frames by position 0..n-1 as a map {i => {i, frame}} for O(1)
  # backward/forward lookups (frame NUMBERS are non-contiguous, start at -123).
  defp index(frames) do
    frames
    |> Enum.with_index()
    |> Map.new(fn {f, i} -> {i, {i, f}} end)
  end

  # Rising edges of `port` INTO hitstun: positions i where frame[i-1] not in
  # hitstun and frame[i] in hitstun. Returns [{i, frame_i}].
  defp hit_edges(frames, port) do
    frames
    |> Enum.with_index()
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [{a, _}, {b, _}] ->
      not MapSet.member?(@hitstun, Map.fetch!(a, port).action) and
        MapSet.member?(@hitstun, Map.fetch!(b, port).action)
    end)
    |> Enum.map(fn [_, {b, i}] -> {i, b} end)
  end

  # Rising edges of `port` losing a stock: frame[i-1].stock > frame[i].stock.
  defp stock_losses(frames, port) do
    frames
    |> Enum.with_index()
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [{a, _}, {b, _}] ->
      sa = Map.fetch!(a, port).stock
      sb = Map.fetch!(b, port).stock
      is_integer(sa) and is_integer(sb) and sb < sa
    end)
    |> Enum.map(fn [_, {b, i}] -> {i, b} end)
  end

  # Were BOTH players out of every non-neutral state for the `lookback`
  # frames ending just before position i?
  defp neutral_before?(arr, i, lookback) do
    lo = max(i - lookback, 0)

    Enum.all?(lo..(i - 1)//1, fn k ->
      case arr[k] do
        {_, f} ->
          not MapSet.member?(@non_neutral, f.p1.action) and
            not MapSet.member?(@non_neutral, f.p2.action)

        _ ->
          true
      end
    end)
  end

  # Starting at the opening hit position i, look ahead `window` frames: the
  # punish is DROPPED when P2 leaves the hit interaction, takes no further
  # hit, and gains < @punish_min_gain percent over the window.
  defp punish_dropped?(arr, i, n, hit_f, window) do
    hi = min(i + window, n - 1)
    span = for k <- (i + 1)..hi//1, {_, f} = arr[k], do: f

    if span == [] do
      false
    else
      became_actionable =
        Enum.any?(span, fn f ->
          not MapSet.member?(@hitstun, f.p2.action) and not MapSet.member?(@lifecycle, f.p2.action)
        end)

      # A second opening = P2 re-enters hitstun after having left it.
      second_hit =
        span
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.any?(fn [a, b] ->
          not MapSet.member?(@hitstun, a.p2.action) and MapSet.member?(@hitstun, b.p2.action)
        end)

      max_pct = span |> Enum.map(& &1.p2.percent) |> Enum.max()
      gain = max_pct - hit_f.p2.percent

      became_actionable and not second_hit and gain < @punish_min_gain
    end
  end

  # Walk back from death position i to the end of the most recent run of
  # `neutral_run` consecutive both-neutral frames. Returns {j, frame_j} at
  # the break (last neutral frame), or nil.
  defp last_neutral_break(arr, i, neutral_run) do
    both_neutral = fn j ->
      case arr[j] do
        {_, f} ->
          not MapSet.member?(@non_neutral, f.p1.action) and
            not MapSet.member?(@non_neutral, f.p2.action)

        _ ->
          false
      end
    end

    # Scan backward accumulating a neutral streak; the first position whose
    # streak reaches neutral_run is the end of a qualifying neutral window.
    Enum.reduce_while((i - 1)..0//-1, 0, fn j, streak ->
      if both_neutral.(j) do
        streak = streak + 1
        if streak >= neutral_run, do: {:halt, {:found, j + streak - 1}}, else: {:cont, streak}
      else
        {:cont, 0}
      end
    end)
    |> case do
      {:found, endj} -> {endj, elem(arr[endj], 1)}
      _ -> nil
    end
  end

  # Keep candidates >= `gap` frames apart (list pre-sorted by frame).
  defp space_out(cands, gap) do
    Enum.reduce(cands, [], fn c, acc ->
      case acc do
        [prev | _] when c.frame - prev.frame < gap -> acc
        _ -> [c | acc]
      end
    end)
    |> Enum.reverse()
  end
end
