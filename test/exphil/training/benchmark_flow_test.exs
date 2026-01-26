defmodule ExPhil.Training.BenchmarkFlowTest do
  @moduledoc """
  Tests for the benchmark script flow to catch issues before GPU deployment.

  These tests verify the critical path through:
  1. Replay loading and frame extraction
  2. Dataset creation from frames
  3. Embedding precomputation
  4. Architecture switching (multiple trainers with memory cleanup)
  5. Cache save/load operations

  Run before deploying to GPU:
    mix test test/exphil/training/benchmark_flow_test.exs
  """
  use ExUnit.Case, async: false

  alias ExPhil.Training.{Data, Imitation}
  alias ExPhil.Embeddings

  @moduletag :unit

  # ============================================================================
  # Replay Loading Tests
  # ============================================================================

  describe "Data.from_frames/2" do
    test "creates valid dataset from factory frames" do
      frames = for i <- 0..9, do: ExPhil.Test.Factory.build_frame(i)

      dataset = Data.from_frames(frames)

      assert dataset.size == 10
      assert length(dataset.frames) == 10
      assert is_struct(dataset.embed_config)
    end

    test "frames have required structure for embedding" do
      frames = for i <- 0..4, do: ExPhil.Test.Factory.build_frame(i)
      dataset = Data.from_frames(frames)

      # Each frame must have game_state with players
      for frame <- dataset.frames do
        assert Map.has_key?(frame, :game_state)
        assert Map.has_key?(frame.game_state, :players)
        assert Map.has_key?(frame.game_state.players, 1)
        assert Map.has_key?(frame.game_state.players, 2)

        # Players must have required fields
        p1 = frame.game_state.players[1]
        assert Map.has_key?(p1, :character)
        assert Map.has_key?(p1, :x)
        assert Map.has_key?(p1, :y)
        assert Map.has_key?(p1, :action)
      end
    end

    test "frames have controller for action extraction" do
      frames = for i <- 0..4, do: ExPhil.Test.Factory.build_frame(i)
      dataset = Data.from_frames(frames)

      for frame <- dataset.frames do
        assert Map.has_key?(frame, :controller)
        controller = frame.controller

        # Controller must have sticks and buttons
        assert Map.has_key?(controller, :main_stick)
        assert Map.has_key?(controller, :c_stick)
        assert Map.has_key?(controller, :button_a)
      end
    end
  end

  # ============================================================================
  # Embedding Pipeline Tests
  # ============================================================================

  describe "embedding pipeline" do
    test "frame embeddings have consistent size" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 5)
      result = Data.precompute_frame_embeddings(dataset, show_progress: false)

      # embedded_frames is now a stacked tensor {num_frames, embed_size}
      assert is_struct(result.embedded_frames, Nx.Tensor)
      {num_frames, embed_size} = Nx.shape(result.embedded_frames)
      assert num_frames == 5
      assert embed_size > 0
    end

    test "sequence embeddings have consistent shape" do
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 3, seq_len: 4)
      result = Data.precompute_embeddings(dataset, show_progress: false)

      # All sequences should have shape {seq_len, embed_size}
      first = :array.get(0, result.embedded_sequences)
      {seq_len, embed_size} = Nx.shape(first)

      assert seq_len == 4

      for i <- 0..2 do
        tensor = :array.get(i, result.embedded_sequences)
        assert Nx.shape(tensor) == {seq_len, embed_size}
      end
    end

    test "embedded dataset can be batched without errors" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 20)
      dataset = Data.precompute_frame_embeddings(dataset, show_progress: false)

      # Should be able to create batches
      batches = Data.batched(dataset, batch_size: 5, shuffle: false)
      batch_list = Enum.take(batches, 3)

      assert length(batch_list) == 3

      for batch <- batch_list do
        assert Map.has_key?(batch, :states)
        assert Map.has_key?(batch, :actions)
        assert is_struct(batch.states, Nx.Tensor)
      end
    end
  end

  # ============================================================================
  # Architecture Switching Tests
  # ============================================================================

  describe "architecture switching" do
    test "can create MLP trainer, train batch, and clean up" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 20)
      dataset = Data.precompute_frame_embeddings(dataset, show_progress: false)

      embed_config = Embeddings.default_config()
      embed_size = Embeddings.embedding_size(embed_config)

      # Create MLP trainer
      trainer =
        Imitation.new(
          embed_size: embed_size,
          embed_config: embed_config,
          temporal: false,
          backbone: :mlp,
          hidden_sizes: [32, 32]
        )

      # Train one batch
      batch = dataset |> Data.batched(batch_size: 5, shuffle: false) |> Enum.take(1) |> hd()
      {_trainer, metrics} = Imitation.train_step(trainer, batch, nil)

      assert is_struct(metrics.loss, Nx.Tensor) or is_float(metrics.loss)

      # Cleanup
      :erlang.garbage_collect()
    end

    test "can switch between MLP and LSTM trainers" do
      # Test that we can create multiple trainers in sequence
      # This is what happens in the benchmark script

      frame_dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 20)
      frame_dataset = Data.precompute_frame_embeddings(frame_dataset, show_progress: false)

      seq_dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 5, seq_len: 4)
      seq_dataset = Data.precompute_embeddings(seq_dataset, show_progress: false)

      embed_config = Embeddings.default_config()
      embed_size = Embeddings.embedding_size(embed_config)

      # First: MLP (non-temporal)
      mlp_trainer =
        Imitation.new(
          embed_size: embed_size,
          embed_config: embed_config,
          temporal: false,
          backbone: :mlp,
          hidden_sizes: [32, 32]
        )

      batch = frame_dataset |> Data.batched(batch_size: 5, shuffle: false) |> Enum.take(1) |> hd()
      {_trainer, mlp_metrics} = Imitation.train_step(mlp_trainer, batch, nil)
      assert is_struct(mlp_metrics.loss, Nx.Tensor) or is_float(mlp_metrics.loss)

      # Force cleanup like benchmark script does
      :erlang.garbage_collect()
      Process.sleep(100)

      # Second: LSTM (temporal)
      lstm_trainer =
        Imitation.new(
          embed_size: embed_size,
          embed_config: embed_config,
          temporal: true,
          backbone: :lstm,
          window_size: 4,
          hidden_size: 32,
          num_layers: 1
        )

      seq_batch =
        seq_dataset
        |> Data.batched_sequences(batch_size: 3, shuffle: false)
        |> Enum.take(1)
        |> hd()

      {_trainer, lstm_metrics} = Imitation.train_step(lstm_trainer, seq_batch, nil)
      assert is_struct(lstm_metrics.loss, Nx.Tensor) or is_float(lstm_metrics.loss)

      # Cleanup
      :erlang.garbage_collect()
    end

    test "can train multiple batches in sequence" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 50)
      dataset = Data.precompute_frame_embeddings(dataset, show_progress: false)

      embed_config = Embeddings.default_config()
      embed_size = Embeddings.embedding_size(embed_config)

      trainer =
        Imitation.new(
          embed_size: embed_size,
          embed_config: embed_config,
          temporal: false,
          backbone: :mlp,
          hidden_sizes: [32, 32]
        )

      # Train multiple batches
      batches = dataset |> Data.batched(batch_size: 10, shuffle: false) |> Enum.take(3)

      final_trainer =
        Enum.reduce(batches, trainer, fn batch, acc_trainer ->
          {new_trainer, _metrics} = Imitation.train_step(acc_trainer, batch, nil)
          new_trainer
        end)

      # Verify step counter increased
      assert final_trainer.step == 3
    end
  end

  # ============================================================================
  # Data.to_sequences Tests
  # ============================================================================

  describe "Data.to_sequences/2" do
    test "creates sequences from frames" do
      frames = for i <- 0..19, do: ExPhil.Test.Factory.build_frame(i)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset, window_size: 5, stride: 1)

      # With 20 frames and window_size=5, we get 16 sequences (0..15 start indices)
      assert seq_dataset.size == 16

      # Each "frame" should have a :sequence key with 5 frames
      first = hd(seq_dataset.frames)
      assert Map.has_key?(first, :sequence)
      assert length(first.sequence) == 5
    end

    test "sequence frames preserve game state structure" do
      frames = for i <- 0..9, do: ExPhil.Test.Factory.build_frame(i)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset, window_size: 3, stride: 1)

      first_seq = hd(seq_dataset.frames)

      for frame <- first_seq.sequence do
        assert Map.has_key?(frame, :game_state)
        assert Map.has_key?(frame.game_state, :players)
      end
    end
  end

  # ============================================================================
  # Fast Batching Performance Tests
  # ============================================================================

  describe "fast batching with stacked tensors" do
    test "batching uses Nx.take for stacked tensor format" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 50)
      dataset = Data.precompute_frame_embeddings(dataset, show_progress: false)

      # Verify stacked tensor format
      assert is_struct(dataset.embedded_frames, Nx.Tensor)

      # Create batches - this should use fast Nx.take path
      batches = Data.batched(dataset, batch_size: 10, shuffle: false)
      batch_list = Enum.take(batches, 3)

      assert length(batch_list) == 3

      # Verify batch shapes
      for batch <- batch_list do
        {batch_size, _embed_size} = Nx.shape(batch.states)
        assert batch_size == 10
      end
    end

    test "multiple batches have correct sequential content" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 20)
      dataset = Data.precompute_frame_embeddings(dataset, show_progress: false)

      # Get all batches without shuffle to verify sequential indexing
      batches = Data.batched(dataset, batch_size: 5, shuffle: false, drop_last: true)
      batch_list = Enum.to_list(batches)

      # Should have 4 batches of 5 (20 frames / 5 = 4)
      assert length(batch_list) == 4

      # Each batch should have states tensor
      for batch <- batch_list do
        assert is_struct(batch.states, Nx.Tensor)
        assert is_map(batch.actions)
      end
    end
  end

  # ============================================================================
  # Integration Test
  # ============================================================================

  describe "full benchmark flow" do
    test "complete flow: frames -> dataset -> embed -> batch -> train" do
      # Simulate the benchmark script flow

      # Step 1: Generate frames (normally from Peppi.to_training_frames)
      frames = for i <- 0..29, do: ExPhil.Test.Factory.build_frame(i)

      # Step 2: Create dataset
      dataset = Data.from_frames(frames)
      assert dataset.size == 30

      # Step 3: Split train/val
      {train_frames, val_frames} = Enum.split(dataset.frames, 24)
      train_dataset = Data.from_frames(train_frames)
      val_dataset = Data.from_frames(val_frames)

      assert train_dataset.size == 24
      assert val_dataset.size == 6

      # Step 4: Precompute embeddings
      train_embedded = Data.precompute_frame_embeddings(train_dataset, show_progress: false)
      val_embedded = Data.precompute_frame_embeddings(val_dataset, show_progress: false)

      # Verify stacked tensors (new fast format)
      assert is_struct(train_embedded.embedded_frames, Nx.Tensor)
      assert is_struct(val_embedded.embedded_frames, Nx.Tensor)
      {train_size, _embed_size} = Nx.shape(train_embedded.embedded_frames)
      assert train_size == 24

      # Step 5: Create trainer
      embed_config = Embeddings.default_config()
      embed_size = Embeddings.embedding_size(embed_config)

      trainer =
        Imitation.new(
          embed_size: embed_size,
          embed_config: embed_config,
          temporal: false,
          backbone: :mlp,
          hidden_sizes: [32, 32]
        )

      # Step 6: Train
      batches = train_embedded |> Data.batched(batch_size: 8, shuffle: false) |> Enum.take(2)

      final_trainer =
        Enum.reduce(batches, trainer, fn batch, acc ->
          {new_trainer, metrics} = Imitation.train_step(acc, batch, nil)
          assert is_struct(metrics.loss, Nx.Tensor) or is_float(metrics.loss)
          new_trainer
        end)

      assert final_trainer.step == 2
    end
  end
end
