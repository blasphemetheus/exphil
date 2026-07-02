#!/usr/bin/env elixir
# Profile each step of the real batch creation pipeline
# Run: mix run scripts/profile_batch_pipeline.exs
#
# Shows where time goes in create_sequence_batch_lazy with real data

alias ExPhil.Training.{Config, Pipeline, Output}

opts = Config.parse_args([
  "--backbone", "mamba", "--replays", "./replays/huggingface",
  "--max-files", "10", "--batch-size", "16"
]) |> Config.validate!() |> Config.ensure_checkpoint_name()

Output.puts("Setting up pipeline (10 files)...")
pipeline = Pipeline.setup!(opts)

Output.puts("Getting batch stream...")
{stream, num_batches} = Pipeline.batch_stream(pipeline, [])

Output.puts("Profiling #{num_batches} batches...\n")

# Time 100 batches, break down each step
times = %{total: [], slice: [], stack: [], transfer: [], actions: []}

batches_to_profile = 200

{_final_times, count} =
  stream
  |> Stream.with_index()
  |> Enum.reduce_while({times, 0}, fn {batch, idx}, {acc, _} ->
    if idx >= batches_to_profile do
      {:halt, {acc, idx}}
    else
      # The batch is already created by the stream — we can't break it apart here
      # Instead, measure total batch-to-train_step time
      {:cont, {acc, idx + 1}}
    end
  end)

# Since we can't instrument inside the lazy batch function easily,
# let's time the full batch iteration vs a train step
Output.puts("Timing batch iteration (#{batches_to_profile} batches):")

{iter_us, _} = :timer.tc(fn ->
  stream
  |> Stream.take(batches_to_profile)
  |> Enum.each(fn _batch -> :ok end)
end)
Output.puts("  Batch iteration only: #{div(iter_us, 1000)}ms total, #{div(iter_us, batches_to_profile)}us/batch")

# Now time with GPU transfer check
{iter2_us, _} = :timer.tc(fn ->
  stream
  |> Stream.take(batches_to_profile)
  |> Enum.each(fn batch ->
    # Force evaluation of the batch tensors
    _ = Nx.shape(batch.states)
    _ = Nx.shape(batch.actions.buttons)
  end)
end)
Output.puts("  Batch + shape check: #{div(iter2_us, 1000)}ms total, #{div(iter2_us, batches_to_profile)}us/batch")

# Time a real train step for comparison
trainer = ExPhil.Training.Trainer.new(pipeline, opts)

Output.puts("\nTiming train step (JIT warmup)...")
first_batch = Enum.take(stream, 1) |> hd()
{jit_us, {trainer, _}} = :timer.tc(fn ->
  ExPhil.Training.Imitation.train_step(trainer, first_batch, nil)
end)
Output.puts("  JIT + first step: #{div(jit_us, 1000)}ms")

Output.puts("\nTiming train step (post-JIT, 50 steps):")
{train_us, _} = :timer.tc(fn ->
  stream
  |> Stream.take(50)
  |> Enum.reduce(trainer, fn batch, t ->
    {new_t, _metrics} = ExPhil.Training.Imitation.train_step(t, batch, nil)
    new_t
  end)
end)
Output.puts("  50 train steps: #{div(train_us, 1000)}ms total, #{div(train_us, 50_000)}ms/step")

Output.puts("\nBreakdown estimate:")
batch_ms = div(iter_us, batches_to_profile * 1000)
step_ms = div(train_us, 50_000)
gpu_ms = step_ms - batch_ms
Output.puts("  Batch creation: ~#{batch_ms}ms")
Output.puts("  GPU train step: ~#{gpu_ms}ms")
Output.puts("  Total per step: ~#{step_ms}ms")
