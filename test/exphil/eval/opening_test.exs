defmodule ExPhil.Eval.OpeningTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.Opening

  defp f(frame, a1, a2, pct2 \\ 0.0, stock2 \\ 4), do: %{frame: frame, p1: %{action: a1, percent: 0.0, stock: 4}, p2: %{action: a2, percent: pct2, stock: stock2}}

  test "a nair hit followed by a second hit before actionability is a converted aerial opening" do
    frames =
      [f(0, 14, 14), f(1, 65, 14), f(2, 65, 76, 10.0), f(3, 20, 76, 10.0), f(4, 68, 76, 10.0), f(5, 68, 77, 22.0), f(6, 14, 77, 22.0), f(7, 14, 14, 22.0)]

    [o] = Opening.openings(frames)
    assert o.opener == :nair and o.family == :aerial and o.hits == 2 and o.converted?
    assert o.damage == 22.0 and o.end_reason == :actionable_gap
  end

  test "an up smash hit with no follow-up is an opening but not converted" do
    frames = [f(0, 14, 14), f(1, 63, 14), f(2, 63, 80, 15.0), f(3, 14, 80, 15.0), f(4, 14, 14, 15.0)]
    [o] = Opening.openings(frames)
    assert o.opener == :usmash and o.family == :smash and o.hits == 1 and not o.converted?
  end

  test "a grab that leads to a throw converts; a grab released does not" do
    thrown = [f(0, 14, 14), f(1, 212, 14), f(2, 214, 223), f(3, 219, 223), f(4, 220, 239, 8.0), f(5, 14, 239, 8.0), f(6, 14, 14, 8.0)]
    [o] = Opening.openings(thrown)
    assert o.family == :grab and o.converted? and o.damage == 8.0

    released = [f(0, 14, 14), f(1, 212, 14), f(2, 214, 223), f(3, 219, 223), f(4, 14, 14)]
    [o2] = Opening.openings(released)
    assert o2.family == :grab and not o2.converted?
  end

  test "fox shine classifies as special and a stock change ends the chain" do
    frames = [f(0, 14, 14), f(1, 360, 14), f(2, 360, 76, 5.0), f(3, 14, 76, 5.0), f(4, 14, 0, 5.0, 3)]
    [o] = Opening.openings(frames)
    assert o.family == :special and o.end_reason == :stock
  end

  test "neutral lookback suppresses openings inside an interaction; summary aggregates" do
    frames = [f(0, 14, 76), f(1, 65, 76), f(2, 65, 14), f(3, 65, 76, 9.0), f(4, 14, 76, 9.0), f(5, 14, 14, 9.0)]
    assert Opening.openings(frames, neutral_lookback: 3) == []
    assert length(Opening.openings(frames)) == 1
    s = Opening.summary(frames)
    assert s.openings == 1 and s.converted == 0 and s.by_family == %{aerial: 1}
  end
end
