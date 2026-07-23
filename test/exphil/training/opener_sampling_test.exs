defmodule ExPhil.Training.OpenerSamplingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.OpenerSampling

  @wait 14
  @jab 44

  # One replay's worth of drill-shaped frames: p1 action per frame.
  defp frames(p1_actions) do
    Enum.map(p1_actions, fn a ->
      %{game_state: %{players: %{1 => %{action: a}, 2 => %{action: @wait}}}}
    end)
  end

  describe "frame_weights/3" do
    test "upweights the approach window before an opener" do
      # 100 neutral frames, then a jab at index 100 (opener after >=60f neutral).
      actions = List.duplicate(@wait, 100) ++ List.duplicate(@jab, 5)
      {weights, stats} = OpenerSampling.frame_weights([frames(actions)], 3.0, lookback: 30)

      assert stats.openers == 1
      # Window [70, 100] upweighted (31 frames), rest 1.0.
      assert Enum.count(weights, &(&1 == 3.0)) == 31
      assert Enum.at(weights, 100) == 3.0
      assert Enum.at(weights, 69) == 1.0
      assert Enum.at(weights, 0) == 1.0
    end

    test "no openers -> all weights 1.0" do
      actions = List.duplicate(@wait, 50) ++ List.duplicate(@jab, 5)
      {weights, stats} = OpenerSampling.frame_weights([frames(actions)], 3.0)
      assert stats.openers == 0
      assert Enum.all?(weights, &(&1 == 1.0))
    end

    test "weights are flat-aligned across multiple replays" do
      a = List.duplicate(@wait, 80) ++ List.duplicate(@jab, 3)
      b = List.duplicate(@wait, 40)
      {weights, stats} = OpenerSampling.frame_weights([frames(a), frames(b)], 2.0, lookback: 10)

      assert length(weights) == length(a) + length(b)
      assert stats.openers == 1
      # Only replay a's window is boosted; replay b (all neutral) stays 1.0.
      assert Enum.all?(Enum.drop(weights, length(a)), &(&1 == 1.0))
    end
  end

  describe "combine_max/2" do
    test "takes the elementwise max (no multiplicative blowup)" do
      assert OpenerSampling.combine_max([1.0, 3.0, 1.0], [2.0, 1.0, 1.0]) == [2.0, 3.0, 1.0]
    end
  end
end
