defmodule ExPhil.Training.ActionDelayTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Data

  defp frame(n, b) do
    %{
      game_state: %{frame: n, players: %{}},
      controller: %{button_b: b}
    }
  end

  test "shift_actions pairs state(t) with controller(t+delay)" do
    frames = for n <- 0..4, do: frame(n, n)
    shifted = Data.shift_actions(frames, 1)

    assert length(shifted) == 4
    assert Enum.map(shifted, & &1.game_state.frame) == [0, 1, 2, 3]
    assert Enum.map(shifted, & &1.controller.button_b) == [1, 2, 3, 4]
  end

  test "shift_actions drops frames whose target crosses a replay boundary" do
    # Two replays concatenated: frames 0..2, then a new game restarting at 0
    frames = [frame(0, :a0), frame(1, :a1), frame(2, :a2), frame(0, :b0), frame(1, :b1)]
    shifted = Data.shift_actions(frames, 1)

    # a2 (no contiguous successor) and b1 (list end) are dropped
    assert Enum.map(shifted, & &1.controller.button_b) == [:a1, :a2, :b1]
  end

  test "delay 0 is identity" do
    frames = [frame(0, :x), frame(5, :y)]
    assert Data.shift_actions(frames, 0) == frames
  end
end
