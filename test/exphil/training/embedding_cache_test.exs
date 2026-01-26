defmodule ExPhil.Training.EmbeddingCacheTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.{Data, EmbeddingCache}
  alias ExPhil.Embeddings.{Game, Player}

  @moduletag :unit

  # ============================================================================
  # Array Type Detection Tests
  # ============================================================================

  describe "array type detection" do
    test "Erlang arrays are tuples starting with :array" do
      arr = :array.from_list([1, 2, 3])

      assert is_tuple(arr)
      assert elem(arr, 0) == :array
    end

    test "lists are not tuples" do
      list = [1, 2, 3]

      refute is_tuple(list)
    end
  end

  # ============================================================================
  # precompute_embeddings Array Return Type Tests
  # ============================================================================

  describe "Data.precompute_embeddings/2 return type" do
    test "returns embedded_sequences as Erlang array, not list" do
      # Create minimal sequence dataset using test factory
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 5, seq_len: 4)

      result = Data.precompute_embeddings(dataset, show_progress: false)

      # CRITICAL: Must be an Erlang array for cache compatibility
      assert is_tuple(result.embedded_sequences),
             "embedded_sequences should be an Erlang array (tuple), got: #{inspect(type_of(result.embedded_sequences))}"

      assert elem(result.embedded_sequences, 0) == :array,
             "embedded_sequences should start with :array atom"

      # Verify we can access elements with :array.get
      first = :array.get(0, result.embedded_sequences)
      assert is_struct(first, Nx.Tensor)
    end

    test "embedded_sequences array has correct size" do
      num_sequences = 8
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: num_sequences, seq_len: 4)

      result = Data.precompute_embeddings(dataset, show_progress: false)

      assert :array.size(result.embedded_sequences) == num_sequences
    end
  end

  # ============================================================================
  # precompute_frame_embeddings Array Return Type Tests
  # ============================================================================

  describe "Data.precompute_frame_embeddings/2 return type" do
    test "returns embedded_frames as stacked tensor" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 10)

      result = Data.precompute_frame_embeddings(dataset, show_progress: false)

      # NEW: Must be a stacked tensor {num_frames, embed_size} for fast batching
      assert is_struct(result.embedded_frames, Nx.Tensor)
      {num_frames, embed_size} = Nx.shape(result.embedded_frames)
      assert num_frames == 10
      assert embed_size > 0
    end
  end

  # ============================================================================
  # batched_sequences Array/List Handling Tests
  # ============================================================================

  describe "Data.batched_sequences/2 input handling" do
    test "handles embedded_sequences as array (from precompute)" do
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 10, seq_len: 4)
      dataset = Data.precompute_embeddings(dataset, show_progress: false)

      # Verify it's an array before batching
      assert is_tuple(dataset.embedded_sequences)

      # Should not crash
      batches = Data.batched_sequences(dataset, batch_size: 3, shuffle: false)
      batch_list = Enum.take(batches, 2)

      assert length(batch_list) == 2
      assert Map.has_key?(hd(batch_list), :states)
    end

    test "handles embedded_sequences as list (legacy format)" do
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 10, seq_len: 4)
      precomputed = Data.precompute_embeddings(dataset, show_progress: false)

      # Simulate legacy format by converting array back to list
      legacy_seqs =
        for i <- 0..(:array.size(precomputed.embedded_sequences) - 1) do
          :array.get(i, precomputed.embedded_sequences)
        end

      legacy_dataset = %{precomputed | embedded_sequences: legacy_seqs}

      # Should also work with list (backwards compatibility)
      batches = Data.batched_sequences(legacy_dataset, batch_size: 3, shuffle: false)
      batch_list = Enum.take(batches, 2)

      assert length(batch_list) == 2
    end
  end

  # ============================================================================
  # EmbeddingCache Save/Load Round-Trip Tests
  # ============================================================================

  describe "EmbeddingCache save/load round-trip" do
    @tag :tmp_dir
    test "round-trips frame embeddings correctly", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 10)
      dataset = Data.precompute_frame_embeddings(dataset, show_progress: false)

      cache_key = "test_frames_#{System.unique_integer([:positive])}"

      # Save the embedded_frames tensor (stacked format)
      assert :ok = EmbeddingCache.save(cache_key, dataset.embedded_frames, cache_dir: tmp_dir)

      # Load should succeed
      {:ok, loaded} = EmbeddingCache.load(cache_key, cache_dir: tmp_dir)

      # Loaded should be a stacked tensor with same shape
      assert is_struct(loaded, Nx.Tensor)
      {num_frames, _embed_size} = Nx.shape(loaded)
      assert num_frames == 10
    end

    @tag :tmp_dir
    test "round-trips sequence embeddings correctly", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 8, seq_len: 4)
      dataset = Data.precompute_embeddings(dataset, show_progress: false)

      cache_key = "test_seqs_#{System.unique_integer([:positive])}"

      # Save should succeed (was crashing before fix!)
      assert :ok = EmbeddingCache.save(cache_key, dataset, cache_dir: tmp_dir)

      # Load should succeed
      {:ok, loaded} = EmbeddingCache.load(cache_key, cache_dir: tmp_dir)

      # Loaded should be a full dataset struct
      assert is_struct(loaded, Data)
      assert is_tuple(loaded.embedded_sequences)
      assert elem(loaded.embedded_sequences, 0) == :array
      assert :array.size(loaded.embedded_sequences) == 8
    end
  end

  # ============================================================================
  # Cache Key Determinism Tests
  # ============================================================================

  describe "EmbeddingCache.cache_key/3" do
    test "generates deterministic keys for same inputs" do
      files = ["a.slp", "b.slp", "c.slp"]

      embed_config = %Game{
        player: %Player{action_mode: :learned},
        stage_mode: :one_hot_compact
      }

      # Note: argument order is (embed_config, replay_files, opts)
      key1 = EmbeddingCache.cache_key(embed_config, files, window_size: 30)
      key2 = EmbeddingCache.cache_key(embed_config, files, window_size: 30)

      assert key1 == key2
    end

    test "generates different keys for different configs" do
      files = ["a.slp", "b.slp"]

      config1 = %Game{player: %Player{action_mode: :learned}, stage_mode: :one_hot_compact}
      config2 = %Game{player: %Player{action_mode: :one_hot}, stage_mode: :one_hot_compact}

      key1 = EmbeddingCache.cache_key(config1, files, window_size: 30)
      key2 = EmbeddingCache.cache_key(config2, files, window_size: 30)

      assert key1 != key2
    end

    test "generates different keys for different window_size" do
      files = ["a.slp"]
      config = %Game{player: %Player{action_mode: :learned}, stage_mode: :one_hot_compact}

      key1 = EmbeddingCache.cache_key(config, files, window_size: 30)
      key2 = EmbeddingCache.cache_key(config, files, window_size: 60)

      assert key1 != key2
    end
  end

  # Helper to identify type for error messages
  defp type_of(x) when is_list(x), do: :list
  defp type_of(x) when is_tuple(x), do: :tuple
  defp type_of(x) when is_map(x), do: :map
  defp type_of(_), do: :other
end
