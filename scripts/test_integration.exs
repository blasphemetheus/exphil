#!/usr/bin/env elixir
# Integration test for the new training system
# Runs a quick 2-epoch training on 5 files to verify everything works end-to-end
#
# Usage: mix run scripts/test_integration.exs

alias ExPhil.Training.{Config, Pipeline, Trainer, Output}
alias ExPhil.Training.Callbacks.{
  ProgressBar, Validation, Diagnostics, Checkpoint,
  EarlyStopping, EpochSummary, GracefulShutdown
}

Output.banner("Integration Test")

# Minimal config
opts = Config.parse_args([
  "--backbone", "mamba",
  "--replays", "./replays/huggingface",
  "--max-files", "10",
  "--epochs", "2",
  "--batch-size", "16",
  "--accumulation-steps", "2"
]) |> Config.validate!() |> Config.ensure_checkpoint_name()

Output.puts("Config parsed OK")
Output.puts("  backbone=#{opts[:backbone]} temporal=#{opts[:temporal]} precision=#{opts[:precision]}")
Output.puts("  batch_size=#{opts[:batch_size]} accum=#{opts[:accumulation_steps]}")
Output.puts("  stick_edge_weight=#{opts[:stick_edge_weight]} lr_schedule=#{opts[:lr_schedule]}")
Output.puts("  checkpoint=#{opts[:checkpoint]}")

# Pipeline
Output.puts("\nSetting up pipeline...")
pipeline = Pipeline.setup!(opts)
Output.puts("  train frames: #{pipeline.train_dataset.size}")
Output.puts("  val batches: #{Pipeline.val_batch_count(pipeline)}")
Output.puts("  estimated batches/epoch: #{pipeline.estimated_batches}")

# Trainer
Output.puts("\nBuilding trainer...")
trainer = Trainer.new(pipeline, opts)
Output.puts("  params: #{Trainer.param_count(trainer)}")

# Callbacks
callbacks = [
  {GracefulShutdown, [checkpoint_path: opts[:checkpoint]]},
  {ProgressBar, [log_interval: 50]},
  {Validation, []},
  {EpochSummary, []},
  {Diagnostics, []}
]

# Train
Output.puts("\nTraining...")
{:ok, state} = Trainer.fit(trainer, pipeline, callbacks: callbacks)

# Verify results
Output.puts("\n=== RESULTS ===")
Output.puts("  epochs completed: #{state.epoch}")
Output.puts("  final train_loss: #{state.train_loss}")
Output.puts("  final val_loss: #{state.val_loss}")
Output.puts("  history length: #{length(state.history)}")
Output.puts("  total steps: #{state.step}")
Output.puts("  seed: #{state.meta[:seed]}")

# Assertions
if state.epoch != 2, do: raise("Expected 2 epochs, got #{state.epoch}")
if state.train_loss == nil, do: raise("train_loss is nil")
if state.val_loss == nil, do: raise("val_loss is nil")
if length(state.history) != 2, do: raise("Expected 2 history entries, got #{length(state.history)}")
if state.step == 0, do: raise("step is 0 — no training happened")
if state.meta[:seed] == nil, do: raise("seed not recorded")

# Verify accumulation worked — steps should be ~half the batch count (accum=2)
expected_steps = div(pipeline.estimated_batches * 2, 2)  # 2 epochs, accum 2
if state.step < div(expected_steps, 2), do: raise("Too few steps (#{state.step}) — accumulation may be broken")

Output.puts("\n  ALL CHECKS PASSED")
