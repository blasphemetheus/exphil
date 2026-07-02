defmodule ExPhil.Training.BatchSpeedTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :benchmark

  alias ExPhil.Training.Data

  @doc """
  Verify batch creation speed meets performance targets.

  With chunked embeddings, batch creation should be <5ms per batch.
  Without chunking, it was ~25ms. Regression = something broke the chunking.
  """

  describe "chunked lazy batch creation speed" do
    @tag timeout: 120_000
    test "batch creation under 5ms per batch" do
      # Create realistic embedding tensor
      num_frames = 100_000
      embed_dim = 288
      embeddings = Nx.iota({num_frames, embed_dim}, type: :f32) |> Nx.backend_copy(Nx.BinaryBackend)

      # Create minimal dataset with embedded_frames
      frames = for i <- 0..(num_frames - 1) do
        %{
          controller: %{
            button_a: rem(i, 10) == 0, button_b: false, button_x: false, button_y: false,
            button_z: false, button_l: false, button_r: false, button_d_up: false,
            main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5},
            l_shoulder: 0.0, r_shoulder: 0.0
          }
        }
      end

      dataset = %Data{
        frames: frames,
        size: num_frames,
        embedded_frames: embeddings,
        metadata: %{},
        embed_config: %{}
      }

      # Create batch stream
      stream = Data.batched_sequences(dataset,
        batch_size: 16, shuffle: false, lazy: true, gpu: false,
        window_size: 60, stride: 5
      )

      # Time 100 batches
      {us, _} = :timer.tc(fn ->
        stream |> Stream.take(100) |> Enum.each(fn _ -> :ok end)
      end)

      ms_per_batch = us / 100_000
      IO.puts("\n  Batch creation: #{Float.round(ms_per_batch, 2)}ms/batch")

      assert ms_per_batch < 5.0,
        "Batch creation too slow: #{Float.round(ms_per_batch, 2)}ms/batch (target: <5ms). " <>
        "Chunked embedding may not be active."
    end
  end

  describe "embedding chunking" do
    test "chunked slice is faster than unchunked" do
      # Create a large tensor
      tensor = Nx.iota({500_000, 288}, type: :f32) |> Nx.backend_copy(Nx.BinaryBackend)

      # Unchunked: slice from large tensor
      {unchunked_us, _} = :timer.tc(fn ->
        for _ <- 1..100 do
          start = :rand.uniform(400_000)
          Nx.slice(tensor, [start, 0], [60, 288])
        end
      end)

      # Chunked: split into chunks, slice from small tensors
      chunk_size = 16_384
      chunks = for i <- 0..(div(500_000, chunk_size) - 1) do
        start = i * chunk_size
        len = min(chunk_size, 500_000 - start)
        Nx.slice(tensor, [start, 0], [len, 288])
      end
      chunks_array = :array.from_list(chunks)

      {chunked_us, _} = :timer.tc(fn ->
        for _ <- 1..100 do
          start = :rand.uniform(400_000)
          chunk_idx = div(start, chunk_size)
          offset = rem(start, chunk_size)
          chunk = :array.get(chunk_idx, chunks_array)
          Nx.slice(chunk, [offset, 0], [60, 288])
        end
      end)

      speedup = unchunked_us / max(chunked_us, 1)
      IO.puts("\n  Unchunked: #{div(unchunked_us, 1000)}ms, Chunked: #{div(chunked_us, 1000)}ms, Speedup: #{Float.round(speedup, 1)}x")

      assert speedup > 2.0,
        "Chunked slicing should be at least 2x faster, got #{Float.round(speedup, 1)}x"
    end
  end
end
