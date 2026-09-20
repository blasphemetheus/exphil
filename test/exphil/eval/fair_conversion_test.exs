defmodule ExPhil.Eval.FairConversionTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.FairConversion

  defp f(i, a1, a2, opts \\ []) do
    %{frame: i,
      p1: %{action: a1, percent: 0.0, stock: Keyword.get(opts, :s1, 4), x: Keyword.get(opts, :x1, 0.0), y: 0.0, facing: 1, on_ground: false},
      p2: %{action: a2, percent: Keyword.get(opts, :pct, 0.0), stock: Keyword.get(opts, :s2, 4), x: Keyword.get(opts, :x2, 10.0), y: 0.0, facing: -1, on_ground: true}}
  end

  test "first fair then a second hit before any actionable frame is a true two-hit" do
    frames = [f(0, 29, 14), f(1, 66, 75, pct: 12.0), f(2, 66, 75, pct: 12.0), f(3, 29, 42), f(4, 66, 42), f(5, 66, 84, pct: 24.0), f(6, 29, 14)]
    [t] = FairConversion.trials(frames)
    assert t.outcome == :true_two_hit and t.actionable_gap == 0 and t.second_contact_frame == 5
    assert t.percent_before == 0.0 and t.percent_after == 12.0 and t.actionability == :inferred
    assert FairConversion.summary(frames).true_two_hit_rate == 1.0
  end

  test "a hit after the defender was actionable is a string hit with the gap recorded" do
    frames = [f(0, 29, 14), f(1, 66, 75), f(2, 29, 14), f(3, 29, 14), f(4, 66, 84), f(5, 29, 14)]
    [t] = FairConversion.trials(frames)
    assert t.outcome == :string_hit and t.actionable_gap == 2
  end

  test "a single hit during hitlag is not counted twice; a same-family transition is not a new contact" do
    frames = [f(0, 29, 14), f(1, 66, 75), f(2, 66, 76), f(3, 66, 76), f(4, 29, 14)] ++ for(i <- 5..130, do: f(i, 14, 14))
    assert length(FairConversion.trials(frames)) == 1
  end

  test "attacker hit before a follow-up = interrupted; defender leaving range after acting = escaped; a second attack that misses = whiffed" do
    interrupted = [f(0, 29, 14), f(1, 66, 75), f(2, 29, 14), f(3, 75, 14), f(4, 75, 14)]
    assert hd(FairConversion.trials(interrupted)).outcome == :interrupted

    escaped = [f(0, 29, 14), f(1, 66, 75), f(2, 29, 14), f(3, 29, 14, x2: 80.0)]
    assert hd(FairConversion.trials(escaped)).outcome == :escaped

    whiffed = [f(0, 29, 14), f(1, 66, 75), f(2, 29, 14), f(3, 66, 14), f(4, 66, 14), f(5, 29, 14)]
    assert hd(FairConversion.trials(whiffed)).outcome == :whiffed
  end

  test "stock change inside the window is reported; a fair contact inside an active trial is consumed, not a new trial" do
    stock = [f(0, 29, 14), f(1, 66, 75), f(2, 66, 75), f(3, 66, 75, s2: 3)]
    assert hd(FairConversion.trials(stock)).outcome == :stock

    overlapping = [f(0, 29, 14), f(1, 66, 75), f(2, 29, 14), f(3, 66, 75), f(4, 29, 14)] ++ for(i <- 5..130, do: f(i, 14, 14))
    [t] = FairConversion.trials(overlapping)
    assert t.outcome == :string_hit and t.second_contact_frame == 3

    at_contact = [f(0, 29, 14, s2: 4), f(1, 66, 75, s2: 3)]
    assert hd(FairConversion.trials(at_contact)).outcome == :invalid
  end
end
