#!/usr/bin/env elixir
# Detailed step-by-step profiling of training iteration
# Run: mix run scripts/profile_step_breakdown.exs

alias ExPhil.Training.{Config, Pipeline, Trainer, Output, Imitation}

opts = Config.parse_args([
  "--backbone", "mamba", "--replays", "./replays/huggingface",
  "--max-files", "10", "--batch-size", "16", "--seed", "42"
]) |> Config.validate!() |> Config.ensure_checkpoint_name()

Output.puts("Setting up pipeline...")
pipeline = Pipeline.setup!(opts)
trainer = Trainer.new(pipeline, opts)

Output.puts("Getting batch stream...")
{stream, num_batches} = Pipeline.batch_stream(pipeline, [])

# Warmup JIT
Output.puts("JIT warmup...")
batch = Enum.take(stream, 1) |> hd()
{trainer, _} = Imitation.train_step(trainer, batch, nil)
:erlang.garbage_collect()

Output.puts("\n=== Profiling 100 iterations ===\n")

# Profile individual components
batch_create_times = []
train_step_times = []
loss_extract_times = []
gc_times = []

{total_us, {_trainer, batch_times, step_times, extract_times}} =
  :timer.tc(fn ->
    stream
    |> Stream.take(100)
    |> Enum.reduce({trainer, [], [], []}, fn batch, {t, bt, st, et} ->
      # Time: train_step (includes forward + backward + optimizer)
      {step_us, {new_t, metrics}} = :timer.tc(fn ->
        Imitation.train_step(t, batch, nil)
      end)

      # Time: loss extraction
      {extract_us, _loss} = :timer.tc(fn ->
        Nx.to_number(metrics.loss)
      end)

      {new_t, [step_us | bt], [step_us | st], [extract_us | et]}
    end)
  end)

# The batch creation time is baked into the stream — we measure it indirectly
# Total = batch_creation + train_step + loss_extract + overhead
# Since we can't separate batch creation from stream consumption,
# let's measure batch creation separately

Output.puts("Batch creation (100 batches, CPU only):")
{batch_us, _} = :timer.tc(fn ->
  stream
  |> Stream.take(100)
  |> Enum.each(fn batch ->
    _ = Nx.shape(batch.states)
  end)
end)
batch_ms = batch_us / 100_000
Output.puts("  #{Float.round(batch_ms, 2)}ms/batch")

Output.puts("\nTrain step (100 steps, includes GPU forward+backward+optimizer):")
avg_step = Enum.sum(step_times) / length(step_times) / 1000
Output.puts("  #{Float.round(avg_step, 2)}ms/step")

Output.puts("\nLoss extraction (100 calls):")
avg_extract = Enum.sum(extract_times) / length(extract_times) / 1000
Output.puts("  #{Float.round(avg_extract, 3)}ms/call")

Output.puts("\nTotal (100 iterations):")
total_ms = total_us / 1000
per_iter = total_ms / 100
Output.puts("  #{Float.round(total_ms, 0)}ms total, #{Float.round(per_iter, 1)}ms/iter")

Output.puts("\nBreakdown estimate:")
gpu_ms = avg_step - batch_ms  # approximate
Output.puts("  Batch creation: ~#{Float.round(batch_ms, 1)}ms")
Output.puts("  GPU compute:    ~#{Float.round(gpu_ms, 1)}ms")
Output.puts("  Loss extract:   ~#{Float.round(avg_extract, 2)}ms")
overhead = per_iter - avg_step
Output.puts("  Stream overhead: ~#{Float.round(overhead, 1)}ms")
Output.puts("  TOTAL:          ~#{Float.round(per_iter, 1)}ms/iter")

# Also time GC impact
Output.puts("\nGC impact:")
{gc_us, _} = :timer.tc(fn ->
  for _ <- 1..100, do: :erlang.garbage_collect()
end)
Output.puts("  100 GC calls: #{div(gc_us, 1000)}ms (#{div(gc_us, 100)}us each)")
