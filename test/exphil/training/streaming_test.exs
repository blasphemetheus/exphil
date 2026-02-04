defmodule ExPhil.Training.StreamingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.{Data, Streaming}
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  # Test helper for creating mock player
  defp mock_player(opts \\ []) do
    %Player{
      character: Keyword.get(opts, :character, 18),
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: Keyword.get(opts, :facing, 1),
      action: Keyword.get(opts, :action, 14),
      action_frame: 0,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }
  end

  defp mock_game_state(opts \\ []) do
    player1 = Keyword.get(opts, :player1, mock_player(x: -30.0))
    player2 = Keyword.get(opts, :player2, mock_player(x: 30.0, facing: 0))

    %GameState{
      frame: Keyword.get(opts, :frame, 0),
      stage: Keyword.get(opts, :stage, 32),
      menu_state: 2,
      players: %{1 => player1, 2 => player2},
      projectiles: [],
      items: [],
      distance: abs(player1.x - player2.x)
    }
  end

  defp mock_controller_state(opts \\ []) do
    %ControllerState{
      main_stick: %{x: Keyword.get(opts, :main_x, 0.5), y: Keyword.get(opts, :main_y, 0.5)},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: 0.0,
      r_shoulder: 0.0,
      button_a: Keyword.get(opts, :button_a, false),
      button_b: Keyword.get(opts, :button_b, false),
      button_x: false,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false,
      button_d_up: false
    }
  end

  describe "chunk_files/2" do
    test "splits files into chunks of specified size" do
      files = ["a.slp", "b.slp", "c.slp", "d.slp", "e.slp"]

      assert Streaming.chunk_files(files, 2) == [
               ["a.slp", "b.slp"],
               ["c.slp", "d.slp"],
               ["e.slp"]
             ]
    end

    test "handles exact divisor" do
      files = ["a.slp", "b.slp", "c.slp", "d.slp"]

      assert Streaming.chunk_files(files, 2) == [
               ["a.slp", "b.slp"],
               ["c.slp", "d.slp"]
             ]
    end

    test "handles single file" do
      assert Streaming.chunk_files(["a.slp"], 5) == [["a.slp"]]
    end

    test "handles empty list" do
      assert Streaming.chunk_files([], 5) == []
    end

    test "handles {path, port} tuples" do
      files = [{"a.slp", 1}, {"b.slp", 2}, {"c.slp", 1}]

      assert Streaming.chunk_files(files, 2) == [
               [{"a.slp", 1}, {"b.slp", 2}],
               [{"c.slp", 1}]
             ]
    end

    test "chunk size of 1 returns individual files" do
      files = ["a.slp", "b.slp", "c.slp"]

      assert Streaming.chunk_files(files, 1) == [
               ["a.slp"],
               ["b.slp"],
               ["c.slp"]
             ]
    end
  end

  describe "format_config/2" do
    test "formats streaming configuration" do
      result = Streaming.format_config(30, 100)
      assert result == "4 chunks of 30 files (100 total)"
    end

    test "handles exact divisor" do
      result = Streaming.format_config(25, 100)
      assert result == "4 chunks of 25 files (100 total)"
    end

    test "handles small datasets" do
      result = Streaming.format_config(50, 20)
      assert result == "1 chunks of 50 files (20 total)"
    end
  end

  describe "streaming mode cache invalidation" do
    # Regression test for bug where process dictionary cache persisted across
    # streaming chunks, causing chunk 2 to use chunk 1's frames array.
    # See commit: fix(streaming): Enable per-chunk precompute
    test "batching multiple datasets uses correct frames for each" do
      # Create two datasets with different frame counts and distinguishable data
      # Dataset 1: 100 frames with button A pressed
      frames1 = for i <- 1..100 do
        %{
          game_state: mock_game_state(frame: i, stage: 31),
          controller: mock_controller_state(button_a: true, button_b: false)
        }
      end

      # Dataset 2: 50 frames with button B pressed (different size to catch cache bugs)
      frames2 = for i <- 1..50 do
        %{
          game_state: mock_game_state(frame: i + 1000, stage: 32),
          controller: mock_controller_state(button_a: false, button_b: true)
        }
      end

      dataset1 = Data.from_frames(frames1)
      dataset2 = Data.from_frames(frames2)

      # Precompute embeddings (simulating streaming mode)
      dataset1 = Data.precompute_frame_embeddings(dataset1, show_progress: false)
      dataset2 = Data.precompute_frame_embeddings(dataset2, show_progress: false)

      # Get batches from dataset1
      batches1 = Data.batched(dataset1, batch_size: 10, shuffle: false) |> Enum.take(1)
      batch1 = hd(batches1)

      # Get batches from dataset2 (this would fail before the fix - would use dataset1's frames)
      batches2 = Data.batched(dataset2, batch_size: 10, shuffle: false) |> Enum.take(1)
      batch2 = hd(batches2)

      # Verify we got the correct data from each dataset
      # Dataset1 has button A pressed, Dataset2 has button B pressed
      assert batch1.actions.buttons != nil
      assert batch2.actions.buttons != nil

      # The actions should be different between the two datasets
      # Button A should be 1.0 in batch1, 0.0 in batch2
      batch1_a_values = Nx.to_flat_list(batch1.actions.buttons[[.., 0]])
      batch2_a_values = Nx.to_flat_list(batch2.actions.buttons[[.., 0]])

      # All frames in dataset1 have A pressed (1.0)
      assert Enum.all?(batch1_a_values, &(&1 == 1.0))
      # All frames in dataset2 have A not pressed (0.0)
      assert Enum.all?(batch2_a_values, &(&1 == 0.0))
    end
  end
end
