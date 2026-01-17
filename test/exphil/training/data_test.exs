defmodule ExPhil.Training.DataTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Data
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  # Helper to create a mock frame
  defp mock_frame(opts) do
    player = %Player{
      character: Keyword.get(opts, :character, 10),  # Mewtwo
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: 1,
      action: 14,  # WAIT
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

    controller = %ControllerState{
      main_stick: %{x: Keyword.get(opts, :stick_x, 0.5), y: Keyword.get(opts, :stick_y, 0.5)},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: Keyword.get(opts, :l_shoulder, 0.0),
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

    game_state = %GameState{
      frame: Keyword.get(opts, :frame, 0),
      stage: 32,  # FD
      menu_state: 2,  # IN_GAME
      players: %{1 => player, 2 => player},
      projectiles: [],
      distance: 50.0
    }

    %{
      game_state: game_state,
      controller: controller
    }
  end

  defp mock_frames(count) do
    Enum.map(0..(count - 1), fn i ->
      mock_frame(frame: i, x: i * 1.0)
    end)
  end

  describe "from_frames/2" do
    test "creates dataset from list of frames" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      assert %Data{} = dataset
      assert dataset.size == 100
      assert length(dataset.frames) == 100
    end

    test "stores metadata when provided" do
      frames = mock_frames(10)
      metadata = %{source: "test", character: :mewtwo}

      dataset = Data.from_frames(frames, metadata: metadata)

      assert dataset.metadata == metadata
    end

    test "uses default embed_config" do
      frames = mock_frames(10)
      dataset = Data.from_frames(frames)

      assert is_map(dataset.embed_config)
    end
  end

  describe "split/2" do
    test "splits dataset into train/val with default ratio" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      {train, val} = Data.split(dataset)

      assert train.size == 90
      assert val.size == 10
      assert train.size + val.size == dataset.size
    end

    test "respects custom ratio" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      {train, val} = Data.split(dataset, ratio: 0.8)

      assert train.size == 80
      assert val.size == 20
    end

    test "can disable shuffling" do
      frames = mock_frames(10)
      dataset = Data.from_frames(frames)

      {train1, _val1} = Data.split(dataset, shuffle: false)
      {train2, _val2} = Data.split(dataset, shuffle: false)

      # Without shuffling, splits should be identical
      assert train1.frames == train2.frames
    end
  end

  describe "sample/2" do
    test "samples n frames from dataset" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      sampled = Data.sample(dataset, 10)

      assert sampled.size == 10
      assert length(sampled.frames) == 10
    end

    test "handles n larger than dataset size" do
      frames = mock_frames(5)
      dataset = Data.from_frames(frames)

      sampled = Data.sample(dataset, 100)

      assert sampled.size == 5
    end
  end

  describe "concat/1" do
    test "concatenates multiple datasets" do
      dataset1 = Data.from_frames(mock_frames(50))
      dataset2 = Data.from_frames(mock_frames(30))
      dataset3 = Data.from_frames(mock_frames(20))

      combined = Data.concat([dataset1, dataset2, dataset3])

      assert combined.size == 100
    end

    test "uses first dataset's embed_config" do
      dataset1 = Data.from_frames(mock_frames(10))
      dataset2 = Data.from_frames(mock_frames(10))

      combined = Data.concat([dataset1, dataset2])

      assert combined.embed_config == dataset1.embed_config
    end
  end

  describe "controller_to_action/2" do
    test "discretizes stick positions" do
      controller = %ControllerState{
        main_stick: %{x: 0.0, y: 1.0},  # Left, Up
        c_stick: %{x: 0.5, y: 0.5},     # Neutral
        l_shoulder: 0.0,
        r_shoulder: 0.0,
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      action = Data.controller_to_action(controller, axis_buckets: 16)

      # x=0.0 -> bucket 0, y=1.0 -> bucket 15 (clamped from 16)
      assert action.main_x == 0
      assert action.main_y == 15
      assert action.c_x == 8  # 0.5 * 16 = 8
      assert action.c_y == 8
    end

    test "extracts button states" do
      controller = %ControllerState{
        main_stick: %{x: 0.5, y: 0.5},
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.0,
        r_shoulder: 0.0,
        button_a: true,
        button_b: false,
        button_x: true,
        button_y: false,
        button_z: true,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      action = Data.controller_to_action(controller)

      assert action.buttons.a == true
      assert action.buttons.b == false
      assert action.buttons.x == true
      assert action.buttons.z == true
    end

    test "discretizes shoulder/trigger" do
      controller = %ControllerState{
        main_stick: %{x: 0.5, y: 0.5},
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.75,
        r_shoulder: 0.25,
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      action = Data.controller_to_action(controller, shoulder_buckets: 4)

      # max(0.75, 0.25) = 0.75, bucket = floor(0.75 * 4) = 3
      assert action.shoulder == 3
    end
  end

  describe "actions_to_tensors/1" do
    test "converts actions list to tensor format" do
      actions = [
        %{
          buttons: %{a: true, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
          main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0
        },
        %{
          buttons: %{a: false, b: true, x: false, y: false, z: false, l: false, r: false, d_up: false},
          main_x: 0, main_y: 15, c_x: 8, c_y: 8, shoulder: 3
        }
      ]

      tensors = Data.actions_to_tensors(actions)

      assert is_struct(tensors.buttons, Nx.Tensor)
      assert Nx.shape(tensors.buttons) == {2, 8}

      assert is_struct(tensors.main_x, Nx.Tensor)
      assert Nx.shape(tensors.main_x) == {2}

      # Check values
      buttons_list = Nx.to_list(tensors.buttons)
      assert Enum.at(buttons_list, 0) == [1, 0, 0, 0, 0, 0, 0, 0]  # a pressed
      assert Enum.at(buttons_list, 1) == [0, 1, 0, 0, 0, 0, 0, 0]  # b pressed

      assert Nx.to_list(tensors.main_x) == [8, 0]
      assert Nx.to_list(tensors.main_y) == [8, 15]
      assert Nx.to_list(tensors.shoulder) == [0, 3]
    end
  end

  describe "stats/1" do
    test "computes button press rates" do
      # Create frames with specific button patterns
      frames = [
        mock_frame(button_a: true, button_b: false),
        mock_frame(button_a: true, button_b: true),
        mock_frame(button_a: false, button_b: true),
        mock_frame(button_a: false, button_b: false)
      ]

      dataset = Data.from_frames(frames)
      stats = Data.stats(dataset)

      assert stats.size == 4
      assert_in_delta stats.button_rates.a, 0.5, 0.01
      assert_in_delta stats.button_rates.b, 0.5, 0.01
    end

    test "computes stick distributions" do
      # Create frames with varied stick positions
      frames = [
        mock_frame(stick_x: 0.0),   # bucket 0
        mock_frame(stick_x: 0.25),  # bucket 4
        mock_frame(stick_x: 0.5),   # bucket 8
        mock_frame(stick_x: 0.75),  # bucket 12
      ]

      dataset = Data.from_frames(frames)
      stats = Data.stats(dataset)

      assert is_list(stats.main_x_distribution)
      assert length(stats.main_x_distribution) == 16
    end
  end

  describe "load_dataset/2" do
    test "returns error for non-existent directory" do
      result = Data.load_dataset("/nonexistent/path/to/replays")

      # Should succeed but with empty frames (logs warning)
      assert {:ok, dataset} = result
      assert dataset.size == 0
    end

    test "creates dataset from directory with parsed files" do
      dir = Path.join(System.tmp_dir!(), "load_dataset_test_#{:rand.uniform(10_000)}")
      File.mkdir_p!(dir)

      try do
        # Create a mock parsed file
        mock_data = %{
          frames: [
            mock_frame(frame: 0),
            mock_frame(frame: 1)
          ]
        }
        File.write!(Path.join(dir, "game1.term"), :erlang.term_to_binary(mock_data))

        {:ok, dataset} = Data.load_dataset(dir)

        assert dataset.size == 2
      after
        File.rm_rf(dir)
      end
    end

    test "respects max_files option" do
      dir = Path.join(System.tmp_dir!(), "max_files_test_#{:rand.uniform(10_000)}")
      File.mkdir_p!(dir)

      try do
        # Create multiple mock files
        for i <- 1..5 do
          mock_data = %{frames: [mock_frame(frame: i)]}
          File.write!(Path.join(dir, "game#{i}.term"), :erlang.term_to_binary(mock_data))
        end

        {:ok, dataset} = Data.load_dataset(dir, max_files: 2)

        # Should only load 2 files = 2 frames
        assert dataset.size == 2
      after
        File.rm_rf(dir)
      end
    end
  end

  describe "batched/2" do
    test "creates batched stream" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      batches = Data.batched(dataset, batch_size: 10, shuffle: false)
      batch_list = Enum.to_list(batches)

      assert length(batch_list) == 10  # 100 / 10
    end

    test "each batch has correct structure" do
      frames = mock_frames(20)
      dataset = Data.from_frames(frames)

      [batch | _] = Data.batched(dataset, batch_size: 5, shuffle: false) |> Enum.take(1)

      assert Map.has_key?(batch, :states)
      assert Map.has_key?(batch, :actions)

      # States should be a tensor
      assert is_struct(batch.states, Nx.Tensor)

      # Actions should have the expected keys
      assert Map.has_key?(batch.actions, :buttons)
      assert Map.has_key?(batch.actions, :main_x)
    end

    test "drop_last removes incomplete final batch" do
      frames = mock_frames(25)
      dataset = Data.from_frames(frames)

      batches_with = Data.batched(dataset, batch_size: 10, drop_last: false, shuffle: false)
      batches_without = Data.batched(dataset, batch_size: 10, drop_last: true, shuffle: false)

      assert length(Enum.to_list(batches_with)) == 3   # 10 + 10 + 5
      assert length(Enum.to_list(batches_without)) == 2 # 10 + 10 (5 dropped)
    end

    test "shuffle produces different orderings" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      batches1 = Data.batched(dataset, batch_size: 10, shuffle: true, seed: 123)
      batches2 = Data.batched(dataset, batch_size: 10, shuffle: true, seed: 456)

      list1 = Enum.to_list(batches1)
      list2 = Enum.to_list(batches2)

      # Different seeds should produce different orderings
      # (Compare first batch states - they should differ)
      states1 = Nx.to_list(hd(list1).states)
      states2 = Nx.to_list(hd(list2).states)

      refute states1 == states2
    end

    test "handles frame_delay option" do
      # With frame_delay, actions come from future frames
      frames = mock_frames(20)
      dataset = Data.from_frames(frames)

      # With delay of 2, we can only use frames 0-17 for states (18-19 are action sources)
      batches = Data.batched(dataset, batch_size: 5, frame_delay: 2, shuffle: false)
      batch_list = Enum.to_list(batches)

      # 18 usable indices / 5 = 3 full batches + 1 partial
      assert length(batch_list) == 4
    end

    # Note: Empty datasets should be filtered upstream before batching
    # The current implementation doesn't explicitly handle size=0
    test "batched returns stream for single-frame dataset" do
      frames = mock_frames(1)
      dataset = Data.from_frames(frames)

      batches = Data.batched(dataset, batch_size: 10, shuffle: false)
      batch_list = Enum.to_list(batches)

      assert length(batch_list) == 1
    end

    test "handles dataset smaller than batch size" do
      frames = mock_frames(5)
      dataset = Data.from_frames(frames)

      batches = Data.batched(dataset, batch_size: 10, shuffle: false)
      batch_list = Enum.to_list(batches)

      assert length(batch_list) == 1
      [batch] = batch_list
      assert Nx.shape(batch.states) |> elem(0) == 5
    end
  end

  describe "edge cases" do
    test "controller_to_action handles edge stick values" do
      # Test boundary values
      controller_min = %ControllerState{
        main_stick: %{x: 0.0, y: 0.0},
        c_stick: %{x: 0.0, y: 0.0},
        l_shoulder: 0.0,
        r_shoulder: 0.0,
        button_a: false, button_b: false, button_x: false, button_y: false,
        button_z: false, button_l: false, button_r: false, button_d_up: false
      }

      controller_max = %ControllerState{
        main_stick: %{x: 1.0, y: 1.0},
        c_stick: %{x: 1.0, y: 1.0},
        l_shoulder: 1.0,
        r_shoulder: 0.0,
        button_a: false, button_b: false, button_x: false, button_y: false,
        button_z: false, button_l: false, button_r: false, button_d_up: false
      }

      action_min = Data.controller_to_action(controller_min, axis_buckets: 16)
      action_max = Data.controller_to_action(controller_max, axis_buckets: 16)

      # Min values should be bucket 0
      assert action_min.main_x == 0
      assert action_min.main_y == 0

      # Max values should be bucket 15 (clamped from 16)
      assert action_max.main_x == 15
      assert action_max.main_y == 15
    end

    test "concat handles single dataset" do
      dataset = Data.from_frames(mock_frames(10))

      combined = Data.concat([dataset])

      assert combined.size == 10
    end

    test "sample returns dataset when n > size" do
      frames = mock_frames(5)
      dataset = Data.from_frames(frames)

      sampled = Data.sample(dataset, 100)

      assert sampled.size == 5
    end

    test "split with ratio 1.0 puts all in train" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      {train, val} = Data.split(dataset, ratio: 1.0, shuffle: false)

      assert train.size == 100
      assert val.size == 0
    end

    test "split with ratio 0.0 puts all in validation" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      {train, val} = Data.split(dataset, ratio: 0.0, shuffle: false)

      assert train.size == 0
      assert val.size == 100
    end
  end
end
