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

  # ============================================================================
  # precompute_frame_embeddings_cached Integration Tests
  # ============================================================================

  describe "Data.precompute_frame_embeddings_cached/2" do
    @tag :tmp_dir
    test "saves cache on first call, loads on second", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 10)
      replay_files = ["test1.slp", "test2.slp"]

      opts = [
        cache: true,
        cache_dir: tmp_dir,
        force_recompute: false,
        replay_files: replay_files,
        show_progress: false
      ]

      # First call should compute and save
      result1 = Data.precompute_frame_embeddings_cached(dataset, opts)
      assert is_struct(result1.embedded_frames, Nx.Tensor)

      # Verify cache file was created
      cache_files = Path.wildcard("#{tmp_dir}/*.emb")
      assert length(cache_files) == 1

      # Second call should load from cache (same result)
      result2 = Data.precompute_frame_embeddings_cached(dataset, opts)
      assert Nx.shape(result2.embedded_frames) == Nx.shape(result1.embedded_frames)
    end

    @tag :tmp_dir
    test "force_recompute ignores existing cache", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 10)
      replay_files = ["test1.slp"]

      # First call - cache
      opts1 = [
        cache: true,
        cache_dir: tmp_dir,
        force_recompute: false,
        replay_files: replay_files,
        show_progress: false
      ]

      _result1 = Data.precompute_frame_embeddings_cached(dataset, opts1)

      # Second call with force_recompute - should recompute
      opts2 = Keyword.put(opts1, :force_recompute, true)
      result2 = Data.precompute_frame_embeddings_cached(dataset, opts2)

      # Should still work
      assert is_struct(result2.embedded_frames, Nx.Tensor)
    end

    test "cache disabled falls back to non-cached version" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 5)

      opts = [cache: false, show_progress: false]

      result = Data.precompute_frame_embeddings_cached(dataset, opts)
      assert is_struct(result.embedded_frames, Nx.Tensor)
    end
  end

  # ============================================================================
  # precompute_embeddings_cached Integration Tests (Sequences)
  # ============================================================================

  describe "Data.precompute_embeddings_cached/2" do
    @tag :tmp_dir
    test "saves cache on first call, loads on second", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 8, seq_len: 4)
      replay_files = ["test1.slp", "test2.slp"]

      opts = [
        cache: true,
        cache_dir: tmp_dir,
        force_recompute: false,
        replay_files: replay_files,
        window_size: 30,
        stride: 1,
        show_progress: false
      ]

      # First call should compute and save
      result1 = Data.precompute_embeddings_cached(dataset, opts)
      assert is_tuple(result1.embedded_sequences)

      # Verify cache file was created
      cache_files = Path.wildcard("#{tmp_dir}/*.emb")
      assert length(cache_files) == 1

      # Second call should load from cache
      result2 = Data.precompute_embeddings_cached(dataset, opts)
      assert :array.size(result2.embedded_sequences) == :array.size(result1.embedded_sequences)
    end

    @tag :tmp_dir
    test "different window_size creates different cache", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 8, seq_len: 4)
      replay_files = ["test1.slp"]

      base_opts = [
        cache: true,
        cache_dir: tmp_dir,
        force_recompute: false,
        replay_files: replay_files,
        stride: 1,
        show_progress: false
      ]

      # Cache with window_size 30
      _result1 = Data.precompute_embeddings_cached(dataset, Keyword.put(base_opts, :window_size, 30))

      # Cache with window_size 60 (should create new cache file)
      _result2 = Data.precompute_embeddings_cached(dataset, Keyword.put(base_opts, :window_size, 60))

      # Should have 2 cache files now
      cache_files = Path.wildcard("#{tmp_dir}/*.emb")
      assert length(cache_files) == 2
    end

    test "cache disabled falls back to non-cached version" do
      dataset = ExPhil.Test.Factory.sequence_dataset(num_sequences: 5, seq_len: 4)

      opts = [cache: false, show_progress: false]

      result = Data.precompute_embeddings_cached(dataset, opts)
      assert is_tuple(result.embedded_sequences)
    end
  end

  # ============================================================================
  # Augmented Embedding Cache Tests
  # ============================================================================

  describe "Data.precompute_augmented_frame_embeddings/2" do
    test "returns 3D tensor with variants" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 10)

      result =
        Data.precompute_augmented_frame_embeddings(dataset,
          num_noisy_variants: 2,
          show_progress: false
        )

      # Should be 3D tensor: {num_frames, num_variants, embed_size}
      assert is_struct(result.embedded_frames, Nx.Tensor)
      shape = Nx.shape(result.embedded_frames)
      assert tuple_size(shape) == 3

      {num_frames, num_variants, embed_size} = shape
      assert num_frames == 10
      # 1 original + 1 mirrored + 2 noisy = 4 variants
      assert num_variants == 4
      assert embed_size > 0
    end

    test "variant count matches configuration" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 5)

      # With 3 noisy variants
      result =
        Data.precompute_augmented_frame_embeddings(dataset,
          num_noisy_variants: 3,
          show_progress: false
        )

      {_frames, num_variants, _embed} = Nx.shape(result.embedded_frames)
      # 1 original + 1 mirrored + 3 noisy = 5 variants
      assert num_variants == 5
    end
  end

  describe "Data.has_augmented_embeddings?/1" do
    test "returns true for 3D embeddings" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 5)

      augmented =
        Data.precompute_augmented_frame_embeddings(dataset,
          num_noisy_variants: 2,
          show_progress: false
        )

      assert Data.has_augmented_embeddings?(augmented)
    end

    test "returns false for 2D embeddings" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 5)
      regular = Data.precompute_frame_embeddings(dataset, show_progress: false)

      refute Data.has_augmented_embeddings?(regular)
    end

    test "returns false for nil embeddings" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 5)
      refute Data.has_augmented_embeddings?(dataset)
    end
  end

  describe "Data.select_variant_index/1" do
    test "returns indices in valid range" do
      for _ <- 1..100 do
        idx =
          Data.select_variant_index(
            mirror_prob: 0.5,
            noise_prob: 0.3,
            num_noisy_variants: 2
          )

        # With 2 noisy variants, valid indices are 0, 1, 2, 3
        assert idx >= 0 and idx <= 3
      end
    end

    test "zero probabilities return original (0)" do
      for _ <- 1..20 do
        idx =
          Data.select_variant_index(
            mirror_prob: 0.0,
            noise_prob: 0.0,
            num_noisy_variants: 2
          )

        assert idx == 0
      end
    end

    test "mirror_prob 1.0 with noise_prob 0.0 returns mirrored (1)" do
      # When noise_prob is 0, and mirror_prob is 1, should always mirror
      for _ <- 1..20 do
        idx =
          Data.select_variant_index(
            mirror_prob: 1.0,
            noise_prob: 0.0,
            num_noisy_variants: 2
          )

        assert idx == 1
      end
    end
  end

  describe "EmbeddingCache.cache_key/3 augmentation params" do
    test "includes augmented flag in key" do
      files = ["a.slp"]
      config = %Game{player: %Player{action_mode: :learned}, stage_mode: :one_hot_compact}

      key1 = EmbeddingCache.cache_key(config, files, augmented: false)
      key2 = EmbeddingCache.cache_key(config, files, augmented: true)

      assert key1 != key2
    end

    test "includes num_noisy_variants in key" do
      files = ["a.slp"]
      config = %Game{player: %Player{action_mode: :learned}, stage_mode: :one_hot_compact}

      key1 = EmbeddingCache.cache_key(config, files, augmented: true, num_noisy_variants: 2)
      key2 = EmbeddingCache.cache_key(config, files, augmented: true, num_noisy_variants: 4)

      assert key1 != key2
    end

    test "includes noise_scale in key" do
      files = ["a.slp"]
      config = %Game{player: %Player{action_mode: :learned}, stage_mode: :one_hot_compact}

      key1 = EmbeddingCache.cache_key(config, files, augmented: true, noise_scale: 0.01)
      key2 = EmbeddingCache.cache_key(config, files, augmented: true, noise_scale: 0.05)

      assert key1 != key2
    end
  end

  describe "Data.precompute_augmented_frame_embeddings_cached/2" do
    @tag :tmp_dir
    test "saves and loads augmented cache correctly", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 10)
      replay_files = ["test1.slp"]

      opts = [
        cache: true,
        cache_dir: tmp_dir,
        force_recompute: false,
        replay_files: replay_files,
        num_noisy_variants: 2,
        noise_scale: 0.01,
        show_progress: false
      ]

      # First call should compute and save
      result1 = Data.precompute_augmented_frame_embeddings_cached(dataset, opts)
      assert Data.has_augmented_embeddings?(result1)
      {frames1, variants1, _embed1} = Nx.shape(result1.embedded_frames)
      assert frames1 == 10
      assert variants1 == 4

      # Second call should load from cache
      result2 = Data.precompute_augmented_frame_embeddings_cached(dataset, opts)
      assert Nx.shape(result2.embedded_frames) == Nx.shape(result1.embedded_frames)
    end

    @tag :tmp_dir
    test "different num_noisy_variants creates different cache", %{tmp_dir: tmp_dir} do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 5)
      replay_files = ["test1.slp"]

      base_opts = [
        cache: true,
        cache_dir: tmp_dir,
        force_recompute: false,
        replay_files: replay_files,
        noise_scale: 0.01,
        show_progress: false
      ]

      # Cache with 2 noisy variants
      _result1 =
        Data.precompute_augmented_frame_embeddings_cached(
          dataset,
          Keyword.put(base_opts, :num_noisy_variants, 2)
        )

      # Cache with 4 noisy variants
      _result2 =
        Data.precompute_augmented_frame_embeddings_cached(
          dataset,
          Keyword.put(base_opts, :num_noisy_variants, 4)
        )

      # Should have 2 cache files
      cache_files = Path.wildcard("#{tmp_dir}/*.emb")
      assert length(cache_files) == 2
    end
  end

  describe "batched with augmented embeddings" do
    test "uses variant selection when augmented embeddings exist" do
      dataset = ExPhil.Test.Factory.frame_dataset(num_frames: 20)

      augmented =
        Data.precompute_augmented_frame_embeddings(dataset,
          num_noisy_variants: 2,
          show_progress: false
        )

      # Create batches with augmentation options
      batches =
        Data.batched(augmented,
          batch_size: 5,
          shuffle: false,
          # No augment_fn - should use variant selection
          mirror_prob: 0.5,
          noise_prob: 0.3,
          num_noisy_variants: 2
        )

      # Should not crash and produce valid batches
      batch_list = Enum.take(batches, 2)
      assert length(batch_list) == 2

      batch = hd(batch_list)
      assert Map.has_key?(batch, :states)
      assert Map.has_key?(batch, :actions)

      # States should be 2D (batch_size, embed_size), not 3D
      states_shape = Nx.shape(batch.states)
      assert tuple_size(states_shape) == 2
    end
  end

  # Helper to identify type for error messages
  defp type_of(x) when is_list(x), do: :list
  defp type_of(x) when is_tuple(x), do: :tuple
  defp type_of(x) when is_map(x), do: :map
  defp type_of(_), do: :other
end
