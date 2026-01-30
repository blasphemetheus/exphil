defmodule ExPhil.Training.DataTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Data
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  # Helper to create a mock frame
  defp mock_frame(opts) do
    player = %Player{
      # Mewtwo
      character: Keyword.get(opts, :character, 10),
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: 1,
      # WAIT
      action: 14,
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
      # FD
      stage: 32,
      # IN_GAME
      menu_state: 2,
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

    test "maintains correspondence between frames and embedded_sequences after split" do
      # Regression test: split was shuffling frames but not embedded_sequences,
      # breaking the index correspondence needed for batching
      frames = mock_frames(20)
      dataset = Data.from_frames(frames)

      # Create sequences and embed them
      seq_dataset = Data.to_sequences(dataset, window_size: 5, stride: 2)
      embedded_dataset = Data.precompute_embeddings(seq_dataset)

      # Verify embeddings exist
      assert embedded_dataset.embedded_sequences != nil
      assert embedded_dataset.size > 0

      # Split with shuffle enabled (the problematic case)
      {train, val} = Data.split(embedded_dataset, ratio: 0.8, shuffle: true)

      # Both should have embedded_sequences that correspond to their frames
      assert train.embedded_sequences != nil
      assert val.embedded_sequences != nil

      # Sizes should match
      train_embed_size = :array.size(train.embedded_sequences)
      val_embed_size = :array.size(val.embedded_sequences)

      assert train_embed_size == train.size,
        "Train embedded_sequences size (#{train_embed_size}) should match frames size (#{train.size})"

      assert val_embed_size == val.size,
        "Val embedded_sequences size (#{val_embed_size}) should match frames size (#{val.size})"

      # Verify the embeddings actually correspond to the frames by checking
      # that we can create valid batches (would fail with mismatched indices)
      train_batches = Data.batched_sequences(train, batch_size: 2, shuffle: false)
      val_batches = Data.batched_sequences(val, batch_size: 2, shuffle: false)

      # Should be able to enumerate without error
      assert Enum.count(train_batches) > 0
      assert Enum.count(val_batches) > 0
    end

    test "maintains correspondence between frames and embedded_frames after split" do
      # Regression test for single-frame (MLP) training with precomputed embeddings
      frames = mock_frames(20)
      dataset = Data.from_frames(frames)

      # Precompute frame embeddings
      embedded_dataset = Data.precompute_frame_embeddings(dataset)

      # Verify embeddings exist
      assert embedded_dataset.embedded_frames != nil
      {num_frames, _embed_dim} = Nx.shape(embedded_dataset.embedded_frames)
      assert num_frames == embedded_dataset.size

      # Split with shuffle enabled
      {train, val} = Data.split(embedded_dataset, ratio: 0.8, shuffle: true)

      # Both should have embedded_frames tensors of correct size
      assert train.embedded_frames != nil
      assert val.embedded_frames != nil

      {train_embed_size, _} = Nx.shape(train.embedded_frames)
      {val_embed_size, _} = Nx.shape(val.embedded_frames)

      assert train_embed_size == train.size,
        "Train embedded_frames rows (#{train_embed_size}) should match frames size (#{train.size})"

      assert val_embed_size == val.size,
        "Val embedded_frames rows (#{val_embed_size}) should match frames size (#{val.size})"
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
        # Left, Up
        main_stick: %{x: 0.0, y: 1.0},
        # Neutral
        c_stick: %{x: 0.5, y: 0.5},
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
      # 0.5 * 16 = 8
      assert action.c_x == 8
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
          buttons: %{
            a: true,
            b: false,
            x: false,
            y: false,
            z: false,
            l: false,
            r: false,
            d_up: false
          },
          main_x: 8,
          main_y: 8,
          c_x: 8,
          c_y: 8,
          shoulder: 0
        },
        %{
          buttons: %{
            a: false,
            b: true,
            x: false,
            y: false,
            z: false,
            l: false,
            r: false,
            d_up: false
          },
          main_x: 0,
          main_y: 15,
          c_x: 8,
          c_y: 8,
          shoulder: 3
        }
      ]

      tensors = Data.actions_to_tensors(actions)

      assert is_struct(tensors.buttons, Nx.Tensor)
      assert Nx.shape(tensors.buttons) == {2, 8}

      assert is_struct(tensors.main_x, Nx.Tensor)
      assert Nx.shape(tensors.main_x) == {2}

      # Check values
      buttons_list = Nx.to_list(tensors.buttons)
      # a pressed
      assert Enum.at(buttons_list, 0) == [1, 0, 0, 0, 0, 0, 0, 0]
      # b pressed
      assert Enum.at(buttons_list, 1) == [0, 1, 0, 0, 0, 0, 0, 0]

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
        # bucket 0
        mock_frame(stick_x: 0.0),
        # bucket 4
        mock_frame(stick_x: 0.25),
        # bucket 8
        mock_frame(stick_x: 0.5),
        # bucket 12
        mock_frame(stick_x: 0.75)
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

      # 100 / 10
      assert length(batch_list) == 10
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

      # 10 + 10 + 5
      assert length(Enum.to_list(batches_with)) == 3
      # 10 + 10 (5 dropped)
      assert length(Enum.to_list(batches_without)) == 2
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
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      controller_max = %ControllerState{
        main_stick: %{x: 1.0, y: 1.0},
        c_stick: %{x: 1.0, y: 1.0},
        l_shoulder: 1.0,
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

  # ============================================================================
  # Temporal/Sequence Data Tests
  # ============================================================================

  describe "to_sequences/2" do
    test "creates sequences from frames with default options" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      # Default: window_size=60, stride=1
      seq_dataset = Data.to_sequences(dataset)

      # With 100 frames, window 60, stride 1: 100 - 60 + 1 = 41 sequences
      assert seq_dataset.size == 41
    end

    test "each sequence has correct structure" do
      frames = mock_frames(70)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 5)

      # Check first sequence
      [first_seq | _] = seq_dataset.frames

      assert Map.has_key?(first_seq, :sequence)
      assert Map.has_key?(first_seq, :game_state)
      assert Map.has_key?(first_seq, :controller)
      assert Map.has_key?(first_seq, :action)

      # Sequence should contain the window of frames
      assert length(first_seq.sequence) == 10

      # Action should be from the last frame in the sequence
      assert is_map(first_seq.action)
      assert Map.has_key?(first_seq.action, :buttons)
    end

    test "respects custom window_size" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset, window_size: 20)

      # 50 - 20 + 1 = 31 sequences (stride 1 default)
      assert seq_dataset.size == 31

      # Each sequence should have 20 frames
      [first_seq | _] = seq_dataset.frames
      assert length(first_seq.sequence) == 20
    end

    test "respects stride parameter" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      # window=10, stride=10 -> non-overlapping windows
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 10)

      # (100 - 10) / 10 + 1 = 10 sequences
      # More precisely: indices 0, 10, 20, ..., 90 = 10 sequences
      assert seq_dataset.size == 10
    end

    test "updates metadata with temporal info" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames, metadata: %{source: "test"})

      seq_dataset = Data.to_sequences(dataset, window_size: 30, stride: 5)

      assert seq_dataset.metadata.temporal == true
      assert seq_dataset.metadata.window_size == 30
      assert seq_dataset.metadata.stride == 5
      assert seq_dataset.metadata.original_size == 100
      # Original metadata should be preserved
      assert seq_dataset.metadata.source == "test"
    end

    test "returns empty dataset when frames < window_size" do
      frames = mock_frames(10)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset, window_size: 60)

      assert seq_dataset.size == 0
      assert seq_dataset.frames == []
    end

    test "handles exact window_size match" do
      frames = mock_frames(60)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset, window_size: 60)

      # Exactly one sequence possible
      assert seq_dataset.size == 1
    end

    test "sequences contain consecutive frames" do
      frames = mock_frames(20)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset, window_size: 5, stride: 1)

      # First sequence should have frames 0-4
      first_seq = hd(seq_dataset.frames)
      first_seq_frames = Enum.map(first_seq.sequence, & &1.game_state.frame)
      assert first_seq_frames == [0, 1, 2, 3, 4]

      # Second sequence should have frames 1-5
      second_seq = Enum.at(seq_dataset.frames, 1)
      second_seq_frames = Enum.map(second_seq.sequence, & &1.game_state.frame)
      assert second_seq_frames == [1, 2, 3, 4, 5]
    end

    test "preserves embed_config from original dataset" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset)

      assert seq_dataset.embed_config == dataset.embed_config
    end
  end

  describe "batched_sequences/2" do
    test "creates batched stream from sequence dataset" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 5)

      batches = Data.batched_sequences(seq_dataset, batch_size: 4, shuffle: false)
      batch_list = Enum.to_list(batches)

      # Should have multiple batches
      assert length(batch_list) > 0
    end

    test "each batch has correct structure" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 5)

      [batch | _] =
        Data.batched_sequences(seq_dataset, batch_size: 4, shuffle: false)
        |> Enum.take(1)

      assert Map.has_key?(batch, :states)
      assert Map.has_key?(batch, :actions)

      # States should be a 3D tensor [batch, seq_len, embed_size]
      assert is_struct(batch.states, Nx.Tensor)
      {batch_dim, seq_dim, _embed_dim} = Nx.shape(batch.states)
      assert batch_dim == 4
      assert seq_dim == 10

      # Actions should have expected keys
      assert Map.has_key?(batch.actions, :buttons)
      assert Map.has_key?(batch.actions, :main_x)
    end

    test "drop_last removes incomplete final batch" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 5)

      # seq_dataset.size = (50 - 10) / 5 + 1 = 9 sequences
      # With batch_size=4: 9 / 4 = 2 full + 1 partial

      with_partial =
        Data.batched_sequences(seq_dataset, batch_size: 4, drop_last: false, shuffle: false)

      without_partial =
        Data.batched_sequences(seq_dataset, batch_size: 4, drop_last: true, shuffle: false)

      # 4 + 4 + 1
      assert length(Enum.to_list(with_partial)) == 3
      # 4 + 4 (1 dropped)
      assert length(Enum.to_list(without_partial)) == 2
    end

    test "shuffle produces different orderings" do
      frames = mock_frames(200)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 2)

      batches1 = Data.batched_sequences(seq_dataset, batch_size: 8, shuffle: true, seed: 111)
      batches2 = Data.batched_sequences(seq_dataset, batch_size: 8, shuffle: true, seed: 999)

      list1 = Enum.to_list(batches1)
      list2 = Enum.to_list(batches2)

      # Different seeds should produce different orderings
      states1 = Nx.to_list(hd(list1).states)
      states2 = Nx.to_list(hd(list2).states)

      refute states1 == states2
    end

    test "no shuffle produces deterministic ordering" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 3)

      batches1 = Data.batched_sequences(seq_dataset, batch_size: 5, shuffle: false)
      batches2 = Data.batched_sequences(seq_dataset, batch_size: 5, shuffle: false)

      [batch1 | _] = Enum.to_list(batches1)
      [batch2 | _] = Enum.to_list(batches2)

      # Same ordering without shuffle
      assert Nx.to_list(batch1.states) == Nx.to_list(batch2.states)
    end

    test "handles small sequence dataset" do
      frames = mock_frames(15)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 2)

      # Only 3 sequences: indices 0, 2, 4
      assert seq_dataset.size == 3

      batches = Data.batched_sequences(seq_dataset, batch_size: 10, shuffle: false)
      batch_list = Enum.to_list(batches)

      assert length(batch_list) == 1
      [batch] = batch_list
      {batch_dim, _seq_dim, _embed_dim} = Nx.shape(batch.states)
      assert batch_dim == 3
    end

    test "action tensors have correct shapes" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10)

      [batch | _] =
        Data.batched_sequences(seq_dataset, batch_size: 8, shuffle: false)
        |> Enum.take(1)

      # Button tensor should be [batch, 8]
      assert Nx.shape(batch.actions.buttons) == {8, 8}

      # Axis tensors should be [batch]
      assert Nx.shape(batch.actions.main_x) == {8}
      assert Nx.shape(batch.actions.main_y) == {8}
      assert Nx.shape(batch.actions.c_x) == {8}
      assert Nx.shape(batch.actions.c_y) == {8}
      assert Nx.shape(batch.actions.shoulder) == {8}
    end
  end

  describe "prepare_for_batching/2" do
    test "converts frames list to array" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      # Before: no frames_array
      assert dataset.frames_array == nil

      # After: frames_array is populated
      prepared = Data.prepare_for_batching(dataset)
      assert prepared.frames_array != nil
      assert is_tuple(prepared.frames_array)
      assert elem(prepared.frames_array, 0) == :array
    end

    test "caches character weights in metadata" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      weights = %{fox: 1.0, falco: 2.0}

      prepared = Data.prepare_for_batching(dataset, character_weights: weights)

      assert is_list(prepared.metadata.cached_character_weights)
      assert length(prepared.metadata.cached_character_weights) == dataset.size
    end

    test "prepared_for_batching? returns correct status" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)

      refute Data.prepared_for_batching?(dataset)

      prepared = Data.prepare_for_batching(dataset)
      assert Data.prepared_for_batching?(prepared)
    end

    test "batched_sequences uses cached arrays when available" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: 10, stride: 5)

      # Prepare for batching
      prepared = Data.prepare_for_batching(seq_dataset)

      # Should work the same but use cached arrays internally
      batches = Data.batched_sequences(prepared, batch_size: 4, shuffle: false)
      batch_list = Enum.to_list(batches)

      assert length(batch_list) > 0
    end
  end

  describe "batching performance" do
    @tag :benchmark
    test "batch creation uses O(1) array access, not O(n) list traversal" do
      # Create a large dataset to test performance
      # With O(n) list traversal (Enum.at), this would take ~28s for 100K frames
      # With O(1) array access (:array.get), this should take <1s
      num_frames = 50_000
      batch_size = 512

      frames = for i <- 0..(num_frames - 1), do: mock_frame(frame: i)
      dataset = Data.from_frames(frames)

      # Precompute embeddings (required for fast batch path)
      dataset = Data.precompute_frame_embeddings(dataset)

      # Time multiple batch creations
      batches = Data.batched(dataset, batch_size: batch_size, shuffle: true)

      {time_us, batches_taken} =
        :timer.tc(fn ->
          batches |> Enum.take(10) |> length()
        end)

      time_ms = time_us / 1000

      # With O(1) array access, 10 batches of 512 from 50K frames should take <500ms
      # With O(n) list access, it would take ~14 seconds (28s * 50K/100K * 10/233)
      assert batches_taken == 10
      assert time_ms < 2000, "Batch creation too slow: #{time_ms}ms for 10 batches. " <>
        "Expected <2000ms with O(1) array access. " <>
        "If >10000ms, likely using O(n) Enum.at instead of :array.get"

      # Log actual time for debugging
      IO.puts("\n  [Benchmark] 10 batches of #{batch_size} from #{num_frames} frames: #{Float.round(time_ms, 1)}ms")
    end

    @tag :benchmark
    test "frames_array is cached in process dictionary" do
      frames = for i <- 0..999, do: mock_frame(frame: i)
      dataset = Data.from_frames(frames)
      dataset = Data.precompute_frame_embeddings(dataset)

      # Clear any cached array
      Process.delete(:frames_array_cache)

      # First batch creation should cache the array
      _ = Data.batched(dataset, batch_size: 64, shuffle: false) |> Enum.take(1)

      # Array should now be cached
      cached = Process.get(:frames_array_cache)
      assert cached != nil, "frames_array should be cached in process dictionary"
      assert :array.size(cached) == 1000
    end

    @tag :benchmark
    @tag timeout: 30_000
    test "Data.split completes in reasonable time for 100K frames (regression for O(n²) bug)" do
      # This test guards against regression of the O(n²) bug in Data.split
      # where Enum.at on list caused 1.8M frames to take 10+ minutes.
      # With the fix (Erlang :array), 100K frames should complete in < 5s.
      #
      # See GOTCHAS.md #48: Data.split O(n²) causes multi-minute hangs

      num_frames = 100_000
      frames = for i <- 0..(num_frames - 1), do: mock_frame(frame: i)
      dataset = Data.from_frames(frames)

      # Measure split time
      {time_us, {train_ds, val_ds}} = :timer.tc(fn ->
        Data.split(dataset, ratio: 0.9, shuffle: true)
      end)

      time_ms = time_us / 1000
      time_s = time_ms / 1000

      # Should complete in under 5 seconds (with fix it's ~100ms)
      # The old O(n²) implementation would take ~10s for 100K frames
      assert time_s < 5.0,
        "Data.split took #{Float.round(time_s, 2)}s for #{num_frames} frames. " <>
        "This suggests O(n²) regression. Expected < 5s with O(n log n) implementation."

      # Verify correctness
      assert train_ds.size == 90_000
      assert val_ds.size == 10_000
      assert train_ds.size + val_ds.size == num_frames

      IO.puts("\n  [Regression] Data.split for #{num_frames} frames: #{Float.round(time_ms, 1)}ms")
    end

    @tag :benchmark
    @tag timeout: 30_000
    test "Data.to_sequences completes in reasonable time for 50K frames (regression test)" do
      # Tests sequence building performance
      # With :array optimization it should be O(n), not O(n²)
      num_frames = 50_000
      window_size = 60
      frames = for i <- 0..(num_frames - 1), do: mock_frame(frame: i)
      dataset = Data.from_frames(frames)

      {time_us, seq_ds} = :timer.tc(fn ->
        Data.to_sequences(dataset, window_size: window_size, stride: 1)
      end)

      time_ms = time_us / 1000
      time_s = time_ms / 1000

      # Should complete in under 10 seconds
      assert time_s < 10.0,
        "Data.to_sequences took #{Float.round(time_s, 2)}s for #{num_frames} frames. " <>
        "Expected < 10s with O(n) implementation."

      expected_sequences = num_frames - window_size + 1
      assert seq_ds.size == expected_sequences

      IO.puts("\n  [Regression] Data.to_sequences for #{num_frames} frames: #{Float.round(time_ms, 1)}ms")
    end

    @tag :benchmark
    @tag timeout: 30_000
    test "Data.stats completes in reasonable time for 100K frames (regression test)" do
      # Tests stats calculation performance - should be O(n)
      num_frames = 100_000
      frames = for i <- 0..(num_frames - 1), do: mock_frame(frame: i, button_a: rem(i, 10) == 0)
      dataset = Data.from_frames(frames)

      {time_us, stats} = :timer.tc(fn ->
        Data.stats(dataset)
      end)

      time_ms = time_us / 1000
      time_s = time_ms / 1000

      # Should complete in under 5 seconds
      assert time_s < 5.0,
        "Data.stats took #{Float.round(time_s, 2)}s for #{num_frames} frames. " <>
        "Expected < 5s with O(n) implementation."

      # Verify correctness - 10% of frames have button_a pressed
      assert_in_delta stats.button_rates[:a], 0.1, 0.01

      IO.puts("\n  [Regression] Data.stats for #{num_frames} frames: #{Float.round(time_ms, 1)}ms")
    end

    @tag :benchmark
    @tag timeout: 60_000
    test "batched with precomputed embeddings is fast for 100K frames (regression test)" do
      # Tests that batch creation uses O(1) Nx.take, not O(n) list traversal
      # This guards against regression of the batch creation optimization
      num_frames = 100_000
      batch_size = 512
      frames = for i <- 0..(num_frames - 1), do: mock_frame(frame: i)
      dataset = Data.from_frames(frames)

      # Precompute embeddings (this is O(n) and expected to take time)
      dataset = Data.precompute_frame_embeddings(dataset)

      # Measure time to create and consume 10 batches
      {time_us, batches} = :timer.tc(fn ->
        dataset
        |> Data.batched(batch_size: batch_size, shuffle: true)
        |> Enum.take(10)
      end)

      time_ms = time_us / 1000
      avg_ms_per_batch = time_ms / 10

      # Each batch should take < 500ms (with O(1) Nx.take it's ~10-50ms)
      # O(n) list traversal would be ~1-5s per batch for 100K frames
      assert avg_ms_per_batch < 500,
        "Batch creation averaged #{Float.round(avg_ms_per_batch, 1)}ms/batch. " <>
        "This suggests O(n) regression. Expected < 500ms with Nx.take optimization."

      assert length(batches) == 10

      IO.puts("\n  [Regression] Batch creation for #{num_frames} frames: #{Float.round(avg_ms_per_batch, 1)}ms/batch avg")
    end
  end
end
