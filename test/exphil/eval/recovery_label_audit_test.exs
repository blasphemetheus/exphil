defmodule ExPhil.Eval.RecoveryLabelAuditTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MultishineExpert
  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Eval.RecoveryLabelAudit

  test "compares with actual future commands rather than the model's own projection" do
    neutral = ControllerState.neutral()
    shine = %{neutral | button_b: true}
    expert = %MultishineExpert{table: %{{14, 0, true} => shine}}
    player = %{action: 14, action_frame: 0, on_ground: true}
    first = RecoveryLabelAudit.sample(expert, player, neutral, shine, 100, [0, 1])
    second = RecoveryLabelAudit.sample(expert, player, shine, neutral, 101, [0, 1])
    report = RecoveryLabelAudit.report([first, second])
    future = Enum.find(report.comparisons, &(&1.frame == 100 and &1.shift == 1))
    refute future.matches
    refute future.on_loop
    assert Enum.find(report.comparisons, &(&1.frame == 100 and &1.shift == 0)).matches
    assert length(report.comparisons) == 3
  end

  test "does not score missing intermediate frames or unobserved tails" do
    expert = %MultishineExpert{table: %{}}
    player = %{action: 14, action_frame: 0, on_ground: true}
    sample = RecoveryLabelAudit.sample(expert, player, nil, ControllerState.neutral(), 100, [2])
    assert RecoveryLabelAudit.report([sample, %{sample | frame: 102}]).comparisons == []
  end

  test "duplicate frames fail instead of silently replacing evidence" do
    assert_raise ArgumentError, ~r/duplicate/, fn ->
      RecoveryLabelAudit.report([%{frame: 1}, %{frame: 1}])
    end
  end

  test "on-loop controls match and analog tolerance is applied" do
    neutral = ControllerState.neutral()
    player = %{action: 361, action_frame: 1, on_ground: true}
    key = {361, 1, true}
    expert = %MultishineExpert{table: %{key => neutral}, cycle: [key], phase: %{key => 0}}
    sample = RecoveryLabelAudit.sample(expert, player, nil, neutral, 1, [0])
    assert hd(RecoveryLabelAudit.report([sample]).comparisons).matches
    assert sample.on_loop
    close = put_in(sample, [:issued, :analog], [0.51, 0.5, 0.5, 0.5, 0.0, 0.0])
    assert hd(RecoveryLabelAudit.report([close]).comparisons).matches
    altered = put_in(sample, [:issued, :analog], [0.8, 0.5, 0.5, 0.5, 0.0, 0.0])
    refute hd(RecoveryLabelAudit.report([altered]).comparisons).matches
  end

  test "skipped projections do not become neutral-label matches" do
    expert = %MultishineExpert{table: %{}}
    player = %{action: 0, action_frame: 0, on_ground: false}
    sample = RecoveryLabelAudit.sample(expert, player, nil, ControllerState.neutral(), 1)
    assert RecoveryLabelAudit.report([sample]).comparisons == []
  end
end
