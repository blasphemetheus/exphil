defmodule ExPhil.Training.CachePoisoningRegressionTest do
  @moduledoc """
  Regression tests for GOTCHAS.md #51: a stale embedding-cache entry covering
  only a subset of the dataset (e.g. a pre-reorder train-split tensor) was
  loaded as the full dataset. Out-of-bounds `Nx.take` indices clamp silently
  under XLA, so every val sample became the same row — four months of phantom
  "200-file scaling collapse".

  These pin the guards that make that failure loud.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.{Data, EmbeddingCache}
  alias ExPhil.Embeddings.{Game, Player}

  describe "Data.split/2 bounds check" do
    test "raises when embedded_frames has fewer rows than the dataset" do
      frames = for i <- 1..10, do: %{frame: i}

      dataset = %Data{
        frames: frames,
        size: 10,
        # Simulates the poisoned cache: tensor covers only 7 of 10 frames
        embedded_frames: Nx.broadcast(0.0, {7, 4}),
        embed_config: %{}
      }

      assert_raise ArgumentError, ~r/out of bounds.*7 entries/s, fn ->
        Data.split(dataset, shuffle: false)
      end
    end

    test "splits normally when embedded_frames matches the dataset" do
      frames = for i <- 1..10, do: %{frame: i}

      dataset = %Data{
        frames: frames,
        size: 10,
        embedded_frames: Nx.iota({10, 4}, type: :f32),
        embed_config: %{}
      }

      {train, val} = Data.split(dataset, ratio: 0.8, shuffle: false)

      assert train.size == 8
      assert val.size == 2
      assert Nx.axis_size(train.embedded_frames, 0) == 8
      assert Nx.axis_size(val.embedded_frames, 0) == 2
    end
  end

  describe "NxSafe.take/3" do
    test "raises on out-of-bounds indices instead of clamping" do
      tensor = Nx.iota({5, 3})

      assert_raise ArgumentError, ~r/out of bounds.*5 entries/s, fn ->
        ExPhil.NxSafe.take(tensor, [0, 2, 7], label: "test")
      end

      assert_raise ArgumentError, ~r/out of bounds/, fn ->
        ExPhil.NxSafe.take(tensor, Nx.tensor([-1, 0]), label: "test")
      end
    end

    test "matches Nx.take for valid indices (list and tensor)" do
      tensor = Nx.iota({5, 3})

      assert Nx.to_flat_list(ExPhil.NxSafe.take(tensor, [4, 0, 2])) ==
               Nx.to_flat_list(Nx.take(tensor, Nx.tensor([4, 0, 2], type: :s64)))

      idx = Nx.tensor([1, 3])

      assert Nx.to_flat_list(ExPhil.NxSafe.take(tensor, idx)) ==
               Nx.to_flat_list(Nx.take(tensor, idx))
    end

    test "rejects empty index lists with a clear error" do
      tensor = Nx.iota({5, 3})

      assert_raise ArgumentError, ~r/empty index list/, fn ->
        ExPhil.NxSafe.take(tensor, [])
      end
    end
  end

  describe "EmbeddingCache.load/2 :expected_frames" do
    @tag :tmp_dir
    test "rejects an entry whose row count doesn't match", %{tmp_dir: tmp_dir} do
      # Simulate the poisoned entry: 7-row tensor cached, 10 rows expected
      tensor = Nx.broadcast(0.0, {7, 4})
      :ok = EmbeddingCache.save("poisoned", tensor, cache_dir: tmp_dir)

      assert {:error, %ExPhil.Error.CacheError{reason: :stale}} =
               EmbeddingCache.load("poisoned", cache_dir: tmp_dir, expected_frames: 10)

      # Matching expectation loads fine
      assert {:ok, loaded} =
               EmbeddingCache.load("poisoned", cache_dir: tmp_dir, expected_frames: 7)

      assert Nx.axis_size(loaded, 0) == 7
    end
  end

  describe "EmbeddingCache.cache_key/3 subset-collision resistance" do
    test "same files + config but different frame_count produce different keys" do
      config = %Game{player: %Player{action_mode: :learned}, stage_mode: :one_hot_compact}
      files = ["a.slp", "b.slp"]

      full = EmbeddingCache.cache_key(config, files, temporal: false, frame_count: 1_401_020)
      subset = EmbeddingCache.cache_key(config, files, temporal: false, frame_count: 1_260_918)
      unknown = EmbeddingCache.cache_key(config, files, temporal: false)

      assert full != subset
      assert full != unknown
      assert subset != unknown
    end
  end
end
