defmodule ExPhil.Eval.FairConversion do
  @moduledoc """
  MEWTWO_NEUTRAL_TO_COMBO_HANDOFF.md §4.1: the fair-conversion EVENT scorer.
  Separate from `NeutralExchange`; combo hits are never fresh openings.

  Slim frames (`ScenarioScan.load/1`), attacker P1 = Mewtwo, defender P2.
  A **trial** starts at a *first-fair contact*: a hit edge on the defender
  while the attacker's action is ATTACK_AIR_F (66). The trial is then
  classified by what happens before the defender can act:

    * `:true_two_hit`  — a second hit edge with no actionable defender frame
                          in between (the initial true-combo criterion)
    * `:string_hit`    — a second hit AFTER at least one actionable frame
    * `:escaped`       — actionable frames, no second hit, defender left range
    * `:whiffed`       — attacker attacked again (44..69) after the gap, no hit
    * `:interrupted`   — attacker entered hitstun before a second hit
    * `:no_attempt`    — attacker never attacked again inside the window
    * `:stock`         — a stock changed inside the window
    * `:timeout`       — window elapsed with none of the above
    * `:invalid`       — a stock changed at or before contact, or the trial
                          overlaps a previous trial's window

  Actionability comes from action ids only (no hitlag/IASA in the slim shape):
  the defender is actionable on a frame outside hitstun / knockdown lifecycle
  / landing states. Per the spec, a trial whose classification depended on
  that inference carries `actionability: :inferred`; nothing here certifies a
  true combo until frame data confirms the boundary cases. Percent before /
  after first contact, positions and the gap length are recorded for the
  harness (§4.2).
  """

  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @landing MapSet.new([42, 43, 70, 71, 72, 73, 74])
  @attacks MapSet.new(Enum.to_list(44..69) ++ [212, 214])
  @fair 66
  @window 120
  @escape_dist 40.0

  @type trial :: %{
          contact_frame: integer(),
          outcome: atom(),
          second_contact_frame: integer() | nil,
          actionable_gap: non_neg_integer(),
          percent_before: float(),
          percent_after: float(),
          attacker: map(),
          defender: map(),
          actionability: :inferred
        }

  @doc "All first-fair trials in a replay."
  @spec trials([map()], keyword()) :: [trial()]
  def trials(frames, opts \\ []) do
    window = Keyword.get(opts, :window, @window)
    arr = List.to_tuple(frames)
    n = tuple_size(arr)

    contacts =
      for i <- 1..(n - 1)//1,
          hit_edge?(elem(arr, i - 1).p2.action, elem(arr, i).p2.action),
          elem(arr, i).p1.action == @fair,
          do: i

    {ts, _} =
      # a fair contact inside an active trial's window is that trial's second
      # hit (or a string hit), never a fresh trial
      Enum.reduce(contacts, {[], -1}, fn i, {acc, busy_until} ->
        if i <= busy_until do
          {acc, busy_until}
        else
          t = classify(arr, i, min(i + window, n - 1))
          {[t | acc], if(t.outcome == :invalid, do: busy_until, else: t.window_end)}
        end
      end)

    ts |> Enum.reverse() |> Enum.map(&Map.delete(&1, :window_end))
  end

  @doc "Counts per outcome plus the true-two-hit rate over valid trials."
  @spec summary([map()], keyword()) :: map()
  def summary(frames, opts \\ []) do
    ts = trials(frames, opts)
    valid = Enum.reject(ts, &(&1.outcome == :invalid))
    two = Enum.count(valid, &(&1.outcome == :true_two_hit))

    %{
      trials: length(ts),
      valid: length(valid),
      outcomes: Enum.frequencies_by(ts, & &1.outcome),
      true_two_hit_rate: if(valid == [], do: 0.0, else: two / length(valid)),
      actionability: :inferred
    }
  end

  defp classify(arr, i, last) do
    c = elem(arr, i)
    stock_a = c.p1.stock
    stock_d = c.p2.stock
    prev = elem(arr, i - 1)
    base = %{
      contact_frame: c.frame,
      second_contact_frame: nil,
      actionable_gap: 0,
      percent_before: prev.p2.percent,
      percent_after: c.p2.percent,
      attacker: Map.take(c.p1, [:x, :y, :facing, :on_ground]),
      defender: Map.take(c.p2, [:x, :y, :facing, :on_ground]),
      actionability: :inferred,
      window_end: last
    }

    if prev.p2.stock != stock_d do
      Map.put(base, :outcome, :invalid)
    else
      {outcome, second, gap} =
        Enum.reduce_while((i + 1)..last//1, {:timeout, nil, 0, false}, fn j, {_, _, gap, attacked} ->
          f = elem(arr, j)
          p = elem(arr, j - 1)

          cond do
            f.p1.stock != stock_a or f.p2.stock != stock_d ->
              {:halt, {:stock, nil, gap}}

            MapSet.member?(@hitstun, f.p1.action) and not MapSet.member?(@hitstun, p.p1.action) ->
              {:halt, {:interrupted, nil, gap}}

            hit_edge?(p.p2.action, f.p2.action) ->
              {:halt, {if(gap == 0, do: :true_two_hit, else: :string_hit), f.frame, gap}}

            gap > 0 and abs(f.p1.x - f.p2.x) > @escape_dist and not attacked ->
              {:halt, {:escaped, nil, gap}}

            gap > 0 and attacked and not MapSet.member?(@attacks, f.p1.action) and
                MapSet.member?(@attacks, p.p1.action) ->
              {:halt, {:whiffed, nil, gap}}

            j == last ->
              {:halt, {if(attacked, do: :timeout, else: :no_attempt), nil, gap}}

            true ->
              gap = if actionable?(f.p2.action), do: gap + 1, else: gap
              attacked = attacked or (j > i + 1 and MapSet.member?(@attacks, f.p1.action) and not MapSet.member?(@attacks, p.p1.action))
              {:cont, {:timeout, nil, gap, attacked}}
          end
        end)
        |> case do
          {o, s, g} -> {o, s, g}
          {o, s, g, _} -> {o, s, g}
        end

      base |> Map.put(:outcome, outcome) |> Map.put(:second_contact_frame, second) |> Map.put(:actionable_gap, gap)
    end
  end

  defp hit_edge?(prev, cur), do: MapSet.member?(@hitstun, cur) and not MapSet.member?(@hitstun, prev)

  defp actionable?(action),
    do: not (MapSet.member?(@hitstun, action) or MapSet.member?(@lifecycle, action) or MapSet.member?(@landing, action))
end
