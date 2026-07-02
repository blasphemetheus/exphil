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

      assert_raise ArgumentError, ~r/only 7 rows/, fn ->
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
