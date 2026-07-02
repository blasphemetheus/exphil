#!/usr/bin/env elixir
# Benchmark lazy batch creation to find the bottleneck
# Run: mix run scripts/bench_lazy_batch.exs

alias ExPhil.Training.Output

# Create a realistic embedding tensor (same size as 200-file dataset)
IO.puts("Creating 1.26M x 288 embedding tensor on CPU...")
{embed_time, embeddings} = :timer.tc(fn ->
  Nx.iota({1_260_000, 288}, type: :f32) |> Nx.backend_copy(Nx.BinaryBackend)
end)
IO.puts("  Created in #{div(embed_time, 1000)}ms")

# Create frames array for action lookup
IO.puts("Creating frames array (1.26M entries)...")
frames_array = :array.from_list(for i <- 0..1_259_999, do: %{
  controller: %{
    button_a: rem(i, 10) == 0, button_b: false, button_x: false, button_y: false,
    button_z: false, button_l: false, button_r: false, button_d_up: false,
    main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5},
    l_shoulder: 0.0, r_shoulder: 0.0
  }
})

window_size = 60
stride = 5
embed_dim = 288
batch_size = 16
num_sequences = div(1_260_000 - window_size, stride) + 1

IO.puts("\nBenchmarking individual operations (#{batch_size} sequences per batch):")
IO.puts("  num_sequences: #{num_sequences}")
IO.puts("")

# Pick random sequence indices for a batch
indices = Enum.map(1..batch_size, fn _ -> :rand.uniform(num_sequences) - 1 end)

# Benchmark: Nx.slice
{slice_us, slices} = :timer.tc(fn ->
  Enum.map(indices, fn seq_idx ->
    frame_start = seq_idx * stride
    Nx.slice(embeddings, [frame_start, 0], [window_size, embed_dim])
  end)
end)
IO.puts("  Nx.slice x#{batch_size}:       #{div(slice_us, 1000)}ms (#{div(slice_us, batch_size)}us each)")

# Benchmark: Nx.stack
{stack_us, stacked} = :timer.tc(fn ->
  Nx.stack(slices)
end)
IO.puts("  Nx.stack:                #{div(stack_us, 1000)}ms")

# Benchmark: backend_transfer to GPU
{transfer_us, _gpu_tensor} = :timer.tc(fn ->
  Nx.backend_transfer(stacked, EXLA.Backend)
end)
IO.puts("  backend_transfer to GPU: #{div(transfer_us, 1000)}ms")

# Benchmark: action extraction from frames
{action_us, _actions} = :timer.tc(fn ->
  Enum.map(indices, fn seq_idx ->
    frame_idx = seq_idx * stride + window_size - 1
    frame = :array.get(frame_idx, frames_array)
    ExPhil.Training.Data.controller_to_action(frame.controller)
  end)
end)
IO.puts("  Action extraction:       #{div(action_us, 1000)}ms")

# Benchmark: actions_to_tensors
{a2t_us, _} = :timer.tc(fn ->
  actions = Enum.map(indices, fn seq_idx ->
    frame_idx = seq_idx * stride + window_size - 1
    frame = :array.get(frame_idx, frames_array)
    ExPhil.Training.Data.controller_to_action(frame.controller)
  end)
  ExPhil.Training.Data.actions_to_tensors(actions)
end)
IO.puts("  actions_to_tensors:      #{div(a2t_us, 1000)}ms")

# Benchmark: full batch creation (all steps)
IO.puts("\nFull batch creation (100 batches):")
{full_us, _} = :timer.tc(fn ->
  for _ <- 1..100 do
    batch_indices = Enum.map(1..batch_size, fn _ -> :rand.uniform(num_sequences) - 1 end)

    sequences = Enum.map(batch_indices, fn seq_idx ->
      frame_start = seq_idx * stride
      Nx.slice(embeddings, [frame_start, 0], [window_size, embed_dim])
    end)

    states = Nx.stack(sequences) |> Nx.backend_transfer(EXLA.Backend)

    actions = Enum.map(batch_indices, fn seq_idx ->
      frame_idx = seq_idx * stride + window_size - 1
      frame = :array.get(frame_idx, frames_array)
      ExPhil.Training.Data.controller_to_action(frame.controller)
    end)

    _action_tensors = ExPhil.Training.Data.actions_to_tensors(actions)

    # Simulate what happens — the batch would be consumed by train_step
    # Force the GPU tensor to be freed
    :erlang.garbage_collect()

    {states, actions}
  end
end)
IO.puts("  100 batches in #{div(full_us, 1000)}ms = #{div(full_us, 100_000)}ms/batch")

# Now test with a chunked embedding tensor
IO.puts("\nChunked embedding test (100 chunks of ~12.6K rows):")
chunk_size = 12_600
chunks = for i <- 0..(div(1_260_000, chunk_size) - 1) do
  start = i * chunk_size
  len = min(chunk_size, 1_260_000 - start)
  Nx.slice(embeddings, [start, 0], [len, embed_dim])
end
chunks_array = :array.from_list(chunks)

{chunked_us, _} = :timer.tc(fn ->
  for _ <- 1..100 do
    batch_indices = Enum.map(1..batch_size, fn _ -> :rand.uniform(num_sequences) - 1 end)

    sequences = Enum.map(batch_indices, fn seq_idx ->
      frame_start = seq_idx * stride
      chunk_idx = div(frame_start, chunk_size)
      offset = rem(frame_start, chunk_size)
      chunk = :array.get(chunk_idx, chunks_array)

      # Check if sequence crosses chunk boundary
      if offset + window_size <= elem(Nx.shape(chunk), 0) do
        Nx.slice(chunk, [offset, 0], [window_size, embed_dim])
      else
        # Cross-boundary: take from two chunks
        remaining = elem(Nx.shape(chunk), 0) - offset
        part1 = Nx.slice(chunk, [offset, 0], [remaining, embed_dim])
        next_chunk = :array.get(chunk_idx + 1, chunks_array)
        part2 = Nx.slice(next_chunk, [0, 0], [window_size - remaining, embed_dim])
        Nx.concatenate([part1, part2], axis: 0)
      end
    end)

    Nx.stack(sequences) |> Nx.backend_transfer(EXLA.Backend)
    :erlang.garbage_collect()
  end
end)
IO.puts("  100 batches in #{div(chunked_us, 1000)}ms = #{div(chunked_us, 100_000)}ms/batch")
IO.puts("  Speedup: #{Float.round(full_us / chunked_us, 1)}x")
