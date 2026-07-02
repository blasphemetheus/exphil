#!/usr/bin/env elixir
# Stress test the FULL Mamba model with MixedPrecision
# This is the exact model that crashes in training
# Run: mix run scripts/stress_test_full_mamba.exs

alias ExPhil.Training.{Config, Pipeline, Trainer, Output, Imitation}

opts = Config.parse_args([
  "--backbone", "mamba", "--replays", "./replays/huggingface",
  "--max-files", "5", "--batch-size", "16", "--seed", "42"
  # bf16 is the default for mamba backbone
]) |> Config.validate!() |> Config.ensure_checkpoint_name()

Output.puts("Precision: #{opts[:precision]}")

pipeline = Pipeline.setup!(opts)
trainer = Trainer.new(pipeline, opts)

{stream, _} = Pipeline.batch_stream(pipeline, [])

# JIT warmup
batch = Enum.take(stream, 1) |> hd()
Output.puts("JIT compiling full Mamba model...")
{jit_us, {trainer, metrics}} = :timer.tc(fn ->
  Imitation.train_step(trainer, batch, nil)
end)
Output.puts("  JIT: #{div(jit_us, 1000)}ms, loss: #{Nx.to_number(metrics.loss)}")

# Stress test — run many steps
target_steps = 5000
Output.puts("\nRunning #{target_steps} train steps (crash expected ~3000 with bf16 MP)...")
Output.puts("  Batch size: #{opts[:batch_size]}")
Output.puts("")

batches = stream |> Enum.to_list()
num_batches = length(batches)

{_trainer, count} =
  Stream.cycle(batches)
  |> Stream.take(target_steps)
  |> Enum.reduce({trainer, 0}, fn batch, {t, i} ->
    {new_t, metrics} = Imitation.train_step(t, batch, nil)
    _ = Nx.to_number(metrics.loss)

    if rem(i + 1, 500) == 0 do
      IO.puts("  Step #{i + 1}: OK (cycling over #{num_batches} batches)")
    end

    {new_t, i + 1}
  end)

Output.puts("\n#{count} steps completed without crash!")
Output.puts("bf16 MixedPrecision is stable for this configuration.")
