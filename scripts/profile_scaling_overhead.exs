#!/usr/bin/env elixir
# Profile scaling overhead: what gets slower as training progresses?
# Run: mix run scripts/profile_scaling_overhead.exs

alias ExPhil.Training.{Config, Pipeline, Trainer, Output, Imitation, Callback, TrainingState}
alias ExPhil.Training.Callbacks.{ProgressBar, GracefulShutdown, Checkpoint}

opts = Config.parse_args([
  "--backbone", "mamba", "--replays", "./replays/huggingface",
  "--max-files", "10", "--batch-size", "16", "--seed", "42"
]) |> Config.validate!() |> Config.ensure_checkpoint_name()

pipeline = Pipeline.setup!(opts)
trainer = Trainer.new(pipeline, opts)
{stream, _} = Pipeline.batch_stream(pipeline, [])

# JIT warmup
batch = Enum.take(stream, 1) |> hd()
{trainer, _} = Imitation.train_step(trainer, batch, nil)

Output.puts("\n=== Scaling Overhead Tests ===\n")

# 1. epoch_losses list growth
Output.puts("1. Loss list accumulation overhead:")
batches = stream |> Stream.take(500) |> Enum.to_list()

# Small list (like beginning of epoch)
{small_us, _} = :timer.tc(fn ->
  Enum.reduce(Enum.take(batches, 100), {trainer, []}, fn batch, {t, losses} ->
    {new_t, metrics} = Imitation.train_step(t, batch, nil)
    {new_t, [Nx.to_number(metrics.loss) | losses]}
  end)
end)

# Big list (like end of epoch — pre-fill with 15K dummy entries)
big_list = List.duplicate(0.0, 15_000)
{big_us, _} = :timer.tc(fn ->
  Enum.reduce(Enum.take(batches, 100), {trainer, big_list}, fn batch, {t, losses} ->
    {new_t, metrics} = Imitation.train_step(t, batch, nil)
    {new_t, [Nx.to_number(metrics.loss) | losses]}
  end)
end)

Output.puts("   Small list (100 items): #{div(small_us, 100)}us/iter")
Output.puts("   Big list (15K items):   #{div(big_us, 100)}us/iter")
Output.puts("   Difference: #{div(big_us - small_us, 100)}us/iter")

# 2. GC pause at different heap sizes
Output.puts("\n2. GC pause duration:")
for label <- ["small heap", "after 500 batches"] do
  if label == "after 500 batches" do
    # Grow the heap by allocating stuff
    _big = for _ <- 1..100_000, do: %{a: :rand.uniform(), b: :rand.uniform()}
  end

  {gc_us, _} = :timer.tc(fn ->
    for _ <- 1..10, do: :erlang.garbage_collect()
  end)
  Output.puts("   #{label}: #{div(gc_us, 10)}us/gc")
end

# 3. Callback list traversal scaling
Output.puts("\n3. Callback dispatch with N callbacks:")
state = %TrainingState{trainer: trainer, pipeline: pipeline, opts: opts,
  epoch: 1, epochs: 1, step: 100, batch_idx: 100, batch_metrics: %{loss: 1.0},
  event_counts: %{on_batch_end: 100}}

for n <- [0, 3, 6, 10] do
  # Create N dummy callbacks
  cbs = for _ <- 1..n do
    {ExPhil.Training.Callbacks.EarlyStopping, ExPhil.Training.Callbacks.EarlyStopping.init(patience: 999)}
  end

  {cb_us, _} = :timer.tc(fn ->
    for _ <- 1..10_000 do
      Callback.run(cbs, :on_batch_end, state)
    end
  end)
  Output.puts("   #{n} callbacks: #{div(cb_us, 10_000)}us/dispatch")
end

# 4. Stream.with_index overhead
Output.puts("\n4. Stream overhead:")
raw_batches = Enum.take(batches, 200)

{raw_us, _} = :timer.tc(fn ->
  Enum.reduce(raw_batches, trainer, fn batch, t ->
    {new_t, _} = Imitation.train_step(t, batch, nil)
    new_t
  end)
end)

{stream_us, _} = :timer.tc(fn ->
  raw_batches
  |> Stream.with_index()
  |> Enum.reduce(trainer, fn {batch, _idx}, t ->
    {new_t, _} = Imitation.train_step(t, batch, nil)
    new_t
  end)
end)

Output.puts("   Raw Enum.reduce: #{div(raw_us, 200)}us/iter")
Output.puts("   Stream.with_index: #{div(stream_us, 200)}us/iter")
Output.puts("   Stream overhead: #{div(stream_us - raw_us, 200)}us/iter")

# 5. Reduce vs Reduce_while
Output.puts("\n5. Reduce vs Reduce_while:")
{reduce_us, _} = :timer.tc(fn ->
  Enum.reduce(raw_batches, trainer, fn batch, t ->
    {new_t, _} = Imitation.train_step(t, batch, nil)
    new_t
  end)
end)

{reduce_while_us, _} = :timer.tc(fn ->
  Enum.reduce_while(Enum.with_index(raw_batches), trainer, fn {batch, _idx}, t ->
    {new_t, _} = Imitation.train_step(t, batch, nil)
    {:cont, new_t}
  end)
end)

Output.puts("   Enum.reduce: #{div(reduce_us, 200)}us/iter")
Output.puts("   Enum.reduce_while: #{div(reduce_while_us, 200)}us/iter")

Output.puts("\n=== Done ===")
