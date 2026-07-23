defmodule ExPhil.Training.OpenerSampling do
  @moduledoc """
  Per-frame sampling weights that upweight NEUTRAL-OPENER windows — the
  frames where a player commits to going in (approach + the opener attack)
  from true neutral.

  The r16 lesson: conversion-weighting (`ExPhil.Training.ConversionSampling`)
  upweighted the punish and conversions improved, but initiation
  (gate-10 armed-approaches/min) stayed ~0 because nothing upweighted the
  going-in decision. This is that lever's mirror — train harder on the
  approach, not just the reward once in.

  Same contract as `ConversionSampling.frame_weights/2` (aligned with
  `List.flatten(frame_lists)`, drill-normalized so the learner is
  players[1]), so `dagger_drill.exs` composes the two into one
  `sampling_weights` list. Opener detection is `ExPhil.Eval.NeutralScan`.
  """

  alias ExPhil.Eval.NeutralScan

  @default_lookback 30

  @doc """
  Per-frame weights for a pool of per-replay frame lists.

  Each opener event (a neutral→engagement transition on players[1]) marks
  the window `[index - lookback, index]` — the approach plus the commit —
  with `weight`; everything else is 1.0.

  ## Options
    - `:lookback` — frames of approach before the opener to upweight
      (default #{@default_lookback})
    - `:min_neutral` — passed to NeutralScan (default 60)

  Returns `{weights, stats}` with
  `stats = %{frames, upweighted, openers, distribution}`.
  """
  def frame_weights(frame_lists, weight, opts \\ []) when is_number(weight) do
    lookback = Keyword.get(opts, :lookback, @default_lookback)
    min_neutral = Keyword.get(opts, :min_neutral, 60)

    per_replay =
      Enum.map(frame_lists, fn frames ->
        {p1, p2} = action_lists(frames)
        events = NeutralScan.opener_events(p1, p2, min_neutral: min_neutral)
        weights = window_weights(length(frames), events, lookback, weight)
        {weights, events}
      end)

    weights = Enum.flat_map(per_replay, &elem(&1, 0))
    events = Enum.flat_map(per_replay, &elem(&1, 1))

    stats = %{
      frames: length(weights),
      upweighted: Enum.count(weights, &(&1 != 1.0)),
      openers: length(events),
      distribution: NeutralScan.distribution(events)
    }

    {weights, stats}
  end

  # Learner = players[1], opponent = players[2] (drill normalizes to this).
  defp action_lists(frames) do
    p1 = Enum.map(frames, fn f -> trunc((f.game_state.players[1] && f.game_state.players[1].action) || 0) end)
    p2 = Enum.map(frames, fn f -> trunc((f.game_state.players[2] && f.game_state.players[2].action) || 0) end)
    {p1, p2}
  end

  defp window_weights(0, _events, _lb, _w), do: []

  defp window_weights(n, events, lookback, w) do
    marked =
      Enum.reduce(events, MapSet.new(), fn %{index: i}, acc ->
        Enum.reduce(max(i - lookback, 0)..min(i, n - 1), acc, &MapSet.put(&2, &1))
      end)

    Enum.map(0..(n - 1), fn i -> if MapSet.member?(marked, i), do: w * 1.0, else: 1.0 end)
  end

  @doc """
  Combine two aligned weight lists by elementwise MAX (a frame gets the
  largest applicable boost — no multiplicative explosion when a frame is
  both an opener approach and a conversion). Lists must be equal length.
  """
  def combine_max(a, b) do
    Enum.zip_with(a, b, &max/2)
  end
end
