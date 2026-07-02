#!/usr/bin/env elixir
# Profile why 200 files is 10x slower per iteration than 10 files
# Run: mix run scripts/profile_200file_slowdown.exs

alias ExPhil.Training.{Config, Pipeline, Trainer, Output, Imitation}

Output.banner("200-File Slowdown Investigation")

# Test at multiple scales to find where the slowdown starts
for max_files <- [5, 10, 50, 100, 200] do
  Output.puts("\n=== #{max_files} files ===")

  opts = Config.parse_args([
    "--backbone", "mamba", "--replays", "./replays/huggingface",
    "--max-files", to_string(max_files), "--batch-size", "16", "--seed", "42"
  ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

  # Time pipeline setup
  {setup_us, pipeline} = :timer.tc(fn -> Pipeline.setup!(opts) end)
  train_frames = pipeline.train_dataset.size
  embed_shape = if pipeline.train_dataset.embedded_frames,
    do: inspect(Nx.shape(pipeline.train_dataset.embedded_frames)),
    else: "nil"

  Output.puts("  Setup: #{div(setup_us, 1000)}ms")
  Output.puts("  Train frames: #{train_frames}")
  Output.puts("  Embedding shape: #{embed_shape}")

  # Time batch stream creation
  {stream_us, {stream, num_batches}} = :timer.tc(fn ->
    Pipeline.batch_stream(pipeline, [])
  end)
  Output.puts("  Stream creation: #{div(stream_us, 1000)}ms")
  Output.puts("  Estimated batches: #{num_batches}")

  # Time 20 batch consumptions (just creating batches, no GPU)
  {batch_us, _} = :timer.tc(fn ->
    stream |> Stream.take(20) |> Enum.each(fn b -> _ = Nx.shape(b.states) end)
  end)
  batch_ms = batch_us / 20_000
  Output.puts("  Batch creation: #{Float.round(batch_ms, 2)}ms/batch")

  # Build trainer and time train step
  trainer = Trainer.new(pipeline, opts)

  # JIT warmup
  first_batch = Enum.take(stream, 1) |> hd()
  {trainer, _} = Imitation.train_step(trainer, first_batch, nil)

  # Time 20 real train steps
  batches = stream |> Stream.take(20) |> Enum.to_list()
  {step_us, _} = :timer.tc(fn ->
    Enum.reduce(batches, trainer, fn batch, t ->
      {new_t, metrics} = Imitation.train_step(t, batch, nil)
      _ = Nx.to_number(metrics.loss)
      new_t
    end)
  end)
  step_ms = step_us / 20_000
  Output.puts("  Train step: #{Float.round(step_ms, 2)}ms/step")

  # Memory info
  gpu_info = ExPhil.Training.GPUUtils.get_memory_info()
  case gpu_info do
    {:ok, info} -> Output.puts("  GPU: #{info.used_mb}MB / #{info.total_mb}MB")
    _ -> :ok
  end

  # Force cleanup between scales
  :erlang.garbage_collect()
  Process.sleep(500)
end

Output.puts("\n=== Done ===")
