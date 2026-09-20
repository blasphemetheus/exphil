defmodule ExPhil.Eval.AerialChain do
  @moduledoc """
  GOALS.md Track A gate **A3 — chains: mean connected aerials per opening.**

  Operates on the slim frame shape of `ExPhil.Eval.ScenarioScan.load/1`
  (`%{frame, p1: %{action, percent, stock, ...}, p2: ...}`), attacker P1,
  defender P2 (use `FailureScan.flip/1` for the other port).

  An **opening** is a hit edge on the defender (transition into a hitstun
  state) preceded by `neutral_lookback` frames in which neither player was
  mid-interaction — the same definition `FailureScan.dropped_punish/2`
  uses, so A2/A3 count the same openings.

  From an opening, **connected aerials** are subsequent hit edges on the
  defender where the attacker is in an aerial attack state (65..69) and the
  defender has not been actionable since the previous hit. Actionability is
  inferred from action ids only (the slim shape has no hitlag/IASA data):
  the defender is *actionable* on a frame whose action is neither hitstun,
  nor the knockdown lifecycle, nor a landing state. Every frame like that
  between two hits breaks the chain — a **string**, not a combo. That is
  conservative in the direction the spec asks for ("until actionability is
  verified, label uncertain rather than certify"); the report carries the
  gap length so a later frame-data check can relax it.

  Output per opening: `%{frame, hits, aerials, connected_aerials, chain_end,
  end_reason}`; summary `mean_connected_aerials` is the A3 number.
  """

  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @shield MapSet.new([178, 179, 180, 181, 182])
  @landing MapSet.new([42, 43, 70, 71, 72, 73, 74])
  @aerials MapSet.new(65..69)
  @non_neutral MapSet.union(@hitstun, MapSet.union(@lifecycle, @shield))

  @neutral_lookback 60
  @chain_window 240

  @type opening :: %{
          frame: integer(),
          hits: non_neg_integer(),
          aerials: non_neg_integer(),
          connected_aerials: non_neg_integer(),
          chain_end: integer(),
          end_reason: :actionable_gap | :window | :stock | :end_of_replay
        }

  @doc "Per-opening chain accounting over a replay's slim frames."
  @spec openings([map()], keyword()) :: [opening()]
  def openings(frames, opts \\ []) do
    lookback = Keyword.get(opts, :neutral_lookback, @neutral_lookback)
    window = Keyword.get(opts, :chain_window, @chain_window)
    arr = List.to_tuple(frames)
    n = tuple_size(arr)

    hit_edges(arr)
    |> Enum.filter(fn i -> neutral_before?(arr, i, lookback) end)
    |> Enum.map(fn i -> walk_chain(arr, i, min(i + window, n - 1), window) end)
  end

  @doc "A3: mean connected aerials per opening, plus the distribution."
  @spec summary([map()], keyword()) :: map()
  def summary(frames, opts \\ []) do
    os = openings(frames, opts)
    n = length(os)
    counts = Enum.map(os, & &1.connected_aerials)

    %{
      openings: n,
      mean_connected_aerials: if(n == 0, do: 0.0, else: Enum.sum(counts) / n),
      openings_with_2_plus: Enum.count(counts, &(&1 >= 2)),
      histogram: Enum.frequencies(counts),
      end_reasons: Enum.frequencies_by(os, & &1.end_reason)
    }
  end

  # Walk forward from the opening hit; count hits and aerials until the
  # defender is actionable, the window ends, a stock changes or the replay ends.
  defp walk_chain(arr, start, last, window) do
    first = elem(arr, start)
    stock0 = first.p2.stock

    {hits, aerials, connected, i, reason} =
      Enum.reduce_while((start + 1)..last//1, {1, aerial?(first.p1.action), 0, start, :window}, fn i,
                                                                                          {hits, aerials, connected, _, _} ->
        f = elem(arr, i)
        prev = elem(arr, i - 1)

        cond do
          f.p2.stock != stock0 ->
            {:halt, {hits, aerials, connected, i, :stock}}

          hit_edge?(prev.p2.action, f.p2.action) ->
            a? = aerial?(f.p1.action)
            {:cont, {hits + 1, aerials + b2i(a?), connected + b2i(a?), i, :window}}

          actionable?(f.p2.action) ->
            {:halt, {hits, aerials, connected, i, :actionable_gap}}

          true ->
            {:cont, {hits, aerials, connected, i, :window}}
        end
      end)

    reason = if reason == :window and last == tuple_size(arr) - 1 and last - start < window, do: :end_of_replay, else: reason

    %{frame: first.frame, hits: hits, aerials: aerials, connected_aerials: connected, chain_end: elem(arr, i).frame, end_reason: reason}
  end

  defp hit_edges(arr) do
    for i <- 1..(tuple_size(arr) - 1)//1, hit_edge?(elem(arr, i - 1).p2.action, elem(arr, i).p2.action), do: i
  end

  defp hit_edge?(prev, cur), do: MapSet.member?(@hitstun, cur) and not MapSet.member?(@hitstun, prev)
  defp aerial?(action), do: MapSet.member?(@aerials, action)
  defp b2i(true), do: 1
  defp b2i(false), do: 0

  # Conservative: not in hitstun, not in the knockdown lifecycle, not landing.
  defp actionable?(action),
    do: not (MapSet.member?(@hitstun, action) or MapSet.member?(@lifecycle, action) or MapSet.member?(@landing, action))

  defp neutral_before?(arr, i, lookback) do
    lo = max(i - lookback, 0)

    lo < i and
      Enum.all?(lo..(i - 1)//1, fn j ->
        f = elem(arr, j)
        not MapSet.member?(@non_neutral, f.p1.action) and not MapSet.member?(@non_neutral, f.p2.action)
      end)
  end
end
