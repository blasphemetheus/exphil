defmodule ExPhil.Eval.NeutralExchangeTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.NeutralExchange, as: Exchange

  defp player(opts \\ []),
    do:
      Map.merge(
        %{action: 14, percent: 0, stock: 4, hitstun: false, offstage: false},
        Map.new(opts)
      )

  defp row(f, s \\ [], o \\ []), do: %{frame: f, subject: player(s), opponent: player(o)}
  defp score(rows, opts \\ []), do: Exchange.score(rows, [lead: 2, trade_frames: 2] ++ opts)

  test "shielding allows a grab opening; attempted grabs alone do not score" do
    result =
      score([
        row(0, [], action: 179),
        row(1, [], action: 179),
        row(2, [action: 214], action: 179),
        row(3, [action: 215], action: 226),
        row(4, [], action: 227),
        row(5, [], action: 227)
      ])

    assert [%{outcome: :subject_opens, first_contact: 3}] = result.events
  end

  test "sustained neutral yields one timeout, not overlapping lookahead wins" do
    result = score(Enum.map(0..9, &row/1), max_frames: 8)
    assert [%{start: 1, finish: 9, outcome: :timeout}] = result.events
  end

  test "combo hits count once until both players regain neutral" do
    rows =
      [row(0), row(1)] ++
        Enum.map(2..20, &row(&1, [], percent: &1, hitstun: true, action: 75))

    assert %{subject_opens: 1} = score(rows).outcomes
  end

  test "retaliation inside the observation window is a trade" do
    rows = [
      row(0),
      row(1),
      row(2, [], percent: 5, hitstun: true),
      row(3, [percent: 4, hitstun: true], percent: 5, hitstun: true)
    ]

    assert [%{outcome: :trade}] = score(rows).events
  end

  test "shield contact and attack startup are not hits" do
    rows = [row(0), row(1), row(2, [action: 57], action: 181), row(3, [action: 57], action: 179)]
    assert [%{outcome: :censored, censor_reason: :end_of_recording}] = score(rows).events
  end

  test "gaps and stock changes censor an exchange instead of creating openings" do
    assert [%{censor_reason: :gap}] = score([row(0), row(1), row(5)]).events

    assert [%{censor_reason: :stock_change}] =
             score([row(0), row(1), row(2, [], stock: 3)]).events
  end

  test "incomplete trade observation is censored" do
    assert [%{outcome: :censored}] = score([row(0), row(1), row(2, [], percent: 2)]).events
  end

  test "offstage transition is not a neutral win" do
    assert [%{outcome: :censored, censor_reason: :offstage}] =
             score([row(0), row(1), row(2, [], offstage: true)]).events
  end

  test "empty recording has no evidence" do
    assert %{exchanges: 0, events: []} = score([])
  end

  test "a long throw animation cannot create a fresh neutral exchange" do
    rows =
      [row(0), row(1), row(2, [action: 216], action: 223), row(3, [action: 221], action: 241)] ++
        Enum.map(4..45, &row(&1, [action: 221], action: 241)) ++
        Enum.map(46..50, &row(&1, [action: 221], action: 90, percent: 12, hitstun: true))

    assert %{subject_opens: 1} = score(rows).outcomes
  end
end
