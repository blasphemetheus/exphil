defmodule ExPhil.Training.ConversionSampling do
  @moduledoc """
  Conversion-weighted pool sampling (r15, from the 2026-07-19 human demo:
  the policy has approach mechanics but not the go-in decision — drills
  teach the button sequence, live neutral never selects it).

  Approaches that CONVERT (opponent enters hitstun shortly after
  engagement — `ExPhil.Interp.ReplayStats.conversion_stats/2`) mark a
  frame span from the closure-start decision through the payoff. Frames
  in those spans get sampling weight `w`; everything else stays 1.0. The
  weights feed `Data.batched_sequences(..., sampling_weights: ...)`,
  which replicates each training window by the weight of its supervised
  (last) frame — so converting-approach windows appear ~w times per
  epoch instead of once.

  This biases WHICH windows are drawn (pool sampling), unlike
  `:transition_weight`/`:neutral_weight` which scale per-frame loss
  inside a batch. Weights are computed per source replay so spans never
  cross concat boundaries; replays without position data (or with no
  conversions) contribute all-1.0 weights and sample exactly as before.
  """

  alias ExPhil.Interp.ReplayStats

  @doc """
  Per-frame sampling weights for a pool of per-replay frame lists (the
  drill's post-`shift_actions` lists, so indices align with
  `dataset.frames` exactly).

  Returns `{weights, stats}`: `weights` is a flat list aligned with
  `List.flatten(frame_lists)`; `stats` is
  `%{frames, upweighted, approaches, conversions}`.
  """
  def frame_weights(frame_lists, weight) when is_number(weight) do
    per_replay =
      Enum.map(frame_lists, fn frames ->
        {p1, p2} = stats_shape(frames)
        conv = ReplayStats.conversion_stats(p1, p2)
        weights = span_weights(length(frames), conv.spans, weight)
        {weights, conv}
      end)

    weights = Enum.flat_map(per_replay, &elem(&1, 0))

    stats = %{
      frames: length(weights),
      upweighted: Enum.count(weights, &(&1 != 1.0)),
      approaches: per_replay |> Enum.map(fn {_, c} -> c.approaches end) |> Enum.sum(),
      conversions: per_replay |> Enum.map(fn {_, c} -> c.conversions end) |> Enum.sum()
    }

    {weights, stats}
  end

  # Mirrors ReplayStats.load/1: the drill normalizes every source so the
  # learner is players[1] and the opponent players[2].
  defp stats_shape(frames) do
    port = fn p ->
      players = Enum.map(frames, fn f -> f.game_state.players[p] end)

      %{
        actions: Enum.map(players, fn pl -> trunc((pl && pl.action) || 0) end),
        players: players
      }
    end

    {port.(1), port.(2)}
  end

  defp span_weights(0, _spans, _w), do: []

  defp span_weights(n, spans, w) do
    marked =
      Enum.reduce(spans, MapSet.new(), fn {a, b}, acc ->
        Enum.reduce(max(a, 0)..min(b, n - 1), acc, &MapSet.put(&2, &1))
      end)

    Enum.map(0..(n - 1), fn i -> if MapSet.member?(marked, i), do: w * 1.0, else: 1.0 end)
  end
end
