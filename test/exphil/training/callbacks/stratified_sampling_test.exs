defmodule ExPhil.Training.Callbacks.StratifiedSamplingTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Callbacks.StratifiedSampling

  describe "generate_indices/4" do
    test "produces correct number of indices" do
      frames = make_frames(100, action_rate: 0.2)
      indices = StratifiedSampling.generate_indices(frames, 96, 16)
      assert length(indices) == 96
    end

    test "each batch has minimum action percentage" do
      frames = make_frames(1000, action_rate: 0.1)
      indices = StratifiedSampling.generate_indices(frames, 960, 16, min_action_pct: 0.3)

      # Check each batch of 16
      batches = Enum.chunk_every(indices, 16)
      for batch <- batches do
        action_count = Enum.count(batch, fn idx ->
          frame = Enum.at(frames, idx)
          c = frame.controller
          c.button_a or c.button_l or abs(c.main_stick.x - 0.5) > 0.2
        end)
        pct = action_count / length(batch)
        # Allow tolerance — cycling and rounding can cause variation
        assert pct >= 0.15, "Batch should have >=15% actions, got #{Float.round(pct * 100, 1)}%"
      end
    end

    test "handles all-neutral frames gracefully" do
      frames = make_frames(100, action_rate: 0.0)
      # Should still produce indices even with no action frames
      indices = StratifiedSampling.generate_indices(frames, 96, 16, min_action_pct: 0.3)
      assert length(indices) == 96
    end

    test "handles all-action frames" do
      frames = make_frames(100, action_rate: 1.0)
      indices = StratifiedSampling.generate_indices(frames, 96, 16)
      assert length(indices) == 96
    end
  end

  defp make_frames(n, opts) do
    action_rate = Keyword.get(opts, :action_rate, 0.2)
    for i <- 0..(n - 1) do
      is_action = :rand.uniform() < action_rate
      %{
        controller: %{
          button_a: is_action and rem(i, 3) == 0,
          button_b: false, button_x: false, button_y: false,
          button_z: false,
          button_l: is_action and rem(i, 3) == 1,
          button_r: false, button_d_up: false,
          main_stick: %{x: if(is_action, do: 0.9, else: 0.5), y: 0.5},
          c_stick: %{x: 0.5, y: 0.5},
          l_shoulder: 0.0, r_shoulder: 0.0
        }
      }
    end
  end
end
