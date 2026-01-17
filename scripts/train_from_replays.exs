#!/usr/bin/env elixir
# Training script: Parse replays and run imitation learning
#
# Usage:
#   mix run scripts/train_from_replays.exs [options]
#
# Performance Tips:
#   XLA_FLAGS="--xla_cpu_multi_thread_eigen=true" mix run scripts/train_from_replays.exs
#   - Enables multi-threaded CPU operations (can be 2-3x faster)
#   - Larger batch sizes (128, 256) reduce per-batch overhead if RAM allows
#
# Options:
#   --replays PATH    - Path to replay directory (default: ../replays)
#   --epochs N        - Number of training epochs (default: 10)
#   --batch-size N    - Batch size (default: 64)
#   --max-files N     - Max replay files to use (default: all)
#   --checkpoint PATH - Save checkpoint to path
#   --player PORT     - Player port to learn from (1-4, default: 1)
#   --wandb           - Enable Wandb logging (requires WANDB_API_KEY env)
#   --wandb-project   - Wandb project name (default: exphil)
#   --wandb-name      - Wandb run name (default: auto-generated)
#
# Temporal Training Options:
#   --temporal        - Enable temporal/sequence training with attention
#   --backbone TYPE   - Backbone: sliding_window, hybrid, lstm, mlp (default: sliding_window)
#   --window-size N   - Frames in attention window (default: 60)
#   --stride N        - Stride for sequence sampling (default: 1)

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Training.{Data, Imitation}
alias ExPhil.Embeddings
alias ExPhil.Integrations.Wandb

# Parse command line arguments
args = System.argv()

# Helper to get argument value
get_arg = fn flag, default ->
  case Enum.find_index(args, &(&1 == flag)) do
    nil -> default
    idx -> Enum.at(args, idx + 1) || default
  end
end

has_flag = fn flag -> Enum.member?(args, flag) end

parse_hidden_sizes = fn str ->
  str
  |> String.split(",")
  |> Enum.map(&String.trim/1)
  |> Enum.map(&String.to_integer/1)
end

opts = [
  replays: get_arg.("--replays", "/home/dori/git/melee/replays"),
  epochs: String.to_integer(get_arg.("--epochs", "10")),
  batch_size: String.to_integer(get_arg.("--batch-size", "64")),
  hidden_sizes: parse_hidden_sizes.(get_arg.("--hidden-sizes", "64,64")),
  max_files: case Enum.find_index(args, &(&1 == "--max-files")) do
    nil -> nil
    idx -> String.to_integer(Enum.at(args, idx + 1))
  end,
  checkpoint: get_arg.("--checkpoint", "checkpoints/imitation_latest.axon"),
  player_port: String.to_integer(get_arg.("--player", "1")),
  wandb: has_flag.("--wandb"),
  wandb_project: get_arg.("--wandb-project", "exphil"),
  wandb_name: get_arg.("--wandb-name", nil),
  # Temporal options
  temporal: has_flag.("--temporal"),
  backbone: String.to_atom(get_arg.("--backbone", "sliding_window")),
  window_size: String.to_integer(get_arg.("--window-size", "60")),
  stride: String.to_integer(get_arg.("--stride", "1"))
]

temporal_info = if opts[:temporal] do
  """
    Temporal:    enabled
    Backbone:    #{opts[:backbone]}
    Window:      #{opts[:window_size]} frames
    Stride:      #{opts[:stride]}
  """
else
  "  Temporal:    disabled (single-frame MLP)\n"
end

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║              ExPhil Imitation Learning Training                ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Replays:     #{opts[:replays]}
  Epochs:      #{opts[:epochs]}
  Batch Size:  #{opts[:batch_size]}
  Hidden:      #{inspect(opts[:hidden_sizes], charlists: :as_lists)}
  Max Files:   #{opts[:max_files] || "all"}
  Player Port: #{opts[:player_port]}
  Checkpoint:  #{opts[:checkpoint]}
  Wandb:       #{if opts[:wandb], do: "enabled", else: "disabled"}
#{temporal_info}
""")

# Initialize Wandb if enabled
if opts[:wandb] do
  wandb_opts = [
    project: opts[:wandb_project],
    config: %{
      epochs: opts[:epochs],
      batch_size: opts[:batch_size],
      max_files: opts[:max_files],
      player_port: opts[:player_port],
      hidden_sizes: [64, 64],
      learning_rate: 1.0e-4
    }
  ]

  wandb_opts = if opts[:wandb_name] do
    Keyword.put(wandb_opts, :name, opts[:wandb_name])
  else
    wandb_opts
  end

  case Wandb.start_run(wandb_opts) do
    {:ok, run_id} ->
      IO.puts("Wandb run started: #{run_id}")
    {:error, reason} ->
      IO.puts("Warning: Wandb failed to start: #{inspect(reason)}")
  end
end

# Step 1: Find and parse replays
IO.puts("Step 1: Parsing replays...")

replay_files = Path.wildcard(Path.join(opts[:replays], "**/*.slp"))
replay_files = if opts[:max_files], do: Enum.take(replay_files, opts[:max_files]), else: replay_files

IO.puts("  Found #{length(replay_files)} replay files")

# Parse replays in parallel and collect training frames
{parse_time, all_frames} = :timer.tc(fn ->
  replay_files
  |> Task.async_stream(
    fn path ->
      case Peppi.parse(path, player_port: opts[:player_port]) do
        {:ok, replay} ->
          frames = Peppi.to_training_frames(replay, player_port: opts[:player_port])
          {path, length(frames), frames}
        {:error, reason} ->
          IO.puts("  ⚠ Failed to parse #{Path.basename(path)}: #{reason}")
          {path, 0, []}
      end
    end,
    max_concurrency: System.schedulers_online(),
    timeout: :infinity
  )
  |> Enum.reduce({0, 0, []}, fn
    {:ok, {_path, frame_count, frames}}, {total_files, total_frames, all_frames} ->
      {total_files + 1, total_frames + frame_count, [frames | all_frames]}
    {:exit, _reason}, acc ->
      acc
  end)
  |> then(fn {files, frames, frame_lists} ->
    IO.puts("  Parsed #{files} files, #{frames} total frames")
    List.flatten(frame_lists)
  end)
end)

IO.puts("  Parse time: #{Float.round(parse_time / 1_000_000, 2)}s")
IO.puts("  Total training frames: #{length(all_frames)}")

if length(all_frames) == 0 do
  IO.puts("\n❌ No training frames found. Check replay files and player port.")
  System.halt(1)
end

# Step 2: Create dataset
IO.puts("\nStep 2: Creating dataset...")

dataset = Data.from_frames(all_frames)

# Convert to sequences for temporal training
{train_dataset, val_dataset} = if opts[:temporal] do
  IO.puts("  Converting to sequences (window=#{opts[:window_size]}, stride=#{opts[:stride]})...")
  seq_dataset = Data.to_sequences(dataset,
    window_size: opts[:window_size],
    stride: opts[:stride]
  )
  Data.split(seq_dataset, ratio: 0.9)
else
  Data.split(dataset, ratio: 0.9)
end

IO.puts("  Training #{if opts[:temporal], do: "sequences", else: "frames"}: #{train_dataset.size}")
IO.puts("  Validation #{if opts[:temporal], do: "sequences", else: "frames"}: #{val_dataset.size}")

# Show some statistics
stats = Data.stats(train_dataset)
IO.puts("\n  Button press rates:")
for {button, rate} <- Enum.sort(stats.button_rates) do
  bar = String.duplicate("█", round(rate * 50))
  IO.puts("    #{button |> to_string() |> String.pad_trailing(6)}: #{bar} #{Float.round(rate * 100, 1)}%")
end

# Step 3: Initialize trainer
IO.puts("\nStep 3: Initializing model...")

embed_size = Embeddings.embedding_size()
IO.puts("  Embedding size: #{embed_size}")

trainer_opts = [
  embed_size: embed_size,
  hidden_sizes: opts[:hidden_sizes],
  learning_rate: 1.0e-4,
  batch_size: opts[:batch_size],
  # Temporal options
  temporal: opts[:temporal],
  backbone: opts[:backbone],
  window_size: opts[:window_size],
  num_heads: 2,      # Smaller for CPU training
  head_dim: 32,      # Smaller for CPU training
  num_layers: 1      # Fewer layers for CPU training
]

trainer = Imitation.new(trainer_opts)

if opts[:temporal] do
  IO.puts("  Temporal model initialized (#{opts[:backbone]} backbone)")
else
  IO.puts("  MLP model initialized")
end
IO.puts("  Batch size: #{opts[:batch_size]}")

# Time estimation
# Based on empirical measurements: ~0.1s per batch after JIT, 3-5min for JIT compilation
batches_per_epoch = div(train_dataset.size, opts[:batch_size])
jit_overhead_sec = 300  # ~5 minutes for first JIT compilation
seconds_per_batch = if opts[:temporal], do: 0.5, else: 0.1  # Temporal is slower
estimated_train_sec = (batches_per_epoch * opts[:epochs] * seconds_per_batch) + jit_overhead_sec
estimated_minutes = div(trunc(estimated_train_sec), 60)
estimated_remaining = rem(trunc(estimated_train_sec), 60)

IO.puts("")
IO.puts("  ⏱  Estimated training time: ~#{estimated_minutes}m #{estimated_remaining}s")
IO.puts("      (#{batches_per_epoch} batches/epoch × #{opts[:epochs]} epochs + JIT compilation)")

# Step 4: Training loop
IO.puts("\nStep 4: Training for #{opts[:epochs]} epochs...")
IO.puts("─" |> String.duplicate(60))

start_time = System.monotonic_time(:second)

final_trainer = Enum.reduce(1..opts[:epochs], trainer, fn epoch, current_trainer ->
  epoch_start = System.monotonic_time(:second)

  # Create batched dataset for this epoch
  # Use appropriate batching function based on temporal mode
  batches = if opts[:temporal] do
    Data.batched_sequences(train_dataset,
      batch_size: opts[:batch_size],
      shuffle: true,
      drop_last: true
    )
  else
    Data.batched(train_dataset,
      batch_size: opts[:batch_size],
      shuffle: true,
      drop_last: true
    )
  end
  |> Enum.to_list()

  num_batches = length(batches)

  # Train epoch with JIT compilation indicator
  jit_indicator_shown = if epoch == 1 do
    IO.puts("  ⏳ JIT compiling model (first batch)... this may take 2-5 minutes")
    IO.puts("     (subsequent batches will be fast)")
    true
  else
    false
  end

  {updated_trainer, epoch_losses, _} = Enum.reduce(Enum.with_index(batches), {current_trainer, [], jit_indicator_shown}, fn {batch, batch_idx}, {t, losses, jit_shown} ->
    batch_start = System.monotonic_time(:millisecond)
    {_predict_fn, loss_fn} = Imitation.build_loss_fn(t.policy_model)
    {new_trainer, metrics} = Imitation.train_step(t, batch, loss_fn)
    batch_time = System.monotonic_time(:millisecond) - batch_start

    # Show JIT completion message after first batch
    new_jit_shown = if jit_shown and batch_idx == 0 do
      IO.puts("\n  ✓ JIT compilation complete (took #{Float.round(batch_time / 1000, 1)}s)")
      IO.puts("    Now training...")
      false
    else
      jit_shown
    end

    # Progress indicator every 10%
    if rem(batch_idx + 1, max(1, div(num_batches, 10))) == 0 do
      pct = round((batch_idx + 1) / num_batches * 100)
      elapsed_sec = Float.round(batch_time / 1000, 1)
      IO.write("\r  Epoch #{epoch}: #{pct}% (loss: #{Float.round(metrics.loss, 4)}, #{elapsed_sec}s/batch)")
    end

    {new_trainer, [metrics.loss | losses], new_jit_shown}
  end)

  epoch_time = System.monotonic_time(:second) - epoch_start
  avg_loss = Enum.sum(epoch_losses) / length(epoch_losses)

  # Validation - use appropriate batching for temporal mode
  val_batches = if opts[:temporal] do
    Data.batched_sequences(val_dataset, batch_size: opts[:batch_size], shuffle: false)
  else
    Data.batched(val_dataset, batch_size: opts[:batch_size], shuffle: false)
  end
  val_metrics = Imitation.evaluate(updated_trainer, val_batches)

  IO.puts("\r  Epoch #{epoch}/#{opts[:epochs]}: train_loss=#{Float.round(avg_loss, 4)} val_loss=#{Float.round(val_metrics.loss, 4)} (#{epoch_time}s)")

  updated_trainer
end)

total_time = System.monotonic_time(:second) - start_time
IO.puts("─" |> String.duplicate(60))
IO.puts("Training complete in #{total_time}s")

# Step 5: Save checkpoint
IO.puts("\nStep 5: Saving checkpoint...")
case Imitation.save_checkpoint(final_trainer, opts[:checkpoint]) do
  :ok -> IO.puts("  ✓ Saved to #{opts[:checkpoint]}")
  {:error, reason} -> IO.puts("  ✗ Failed: #{inspect(reason)}")
end

# Also export policy for inference
policy_path = String.replace(opts[:checkpoint], ".axon", "_policy.bin")
case Imitation.export_policy(final_trainer, policy_path) do
  :ok -> IO.puts("  ✓ Policy exported to #{policy_path}")
  {:error, reason} -> IO.puts("  ✗ Failed: #{inspect(reason)}")
end

# Summary
IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                      Training Complete!                        ║
╚════════════════════════════════════════════════════════════════╝

Summary:
  Replays parsed: #{length(replay_files)}
  Training frames: #{train_dataset.size}
  Epochs completed: #{opts[:epochs]}
  Final training loss: #{Float.round(Enum.sum(Enum.take(final_trainer.metrics.loss, 10)) / 10, 4)}
  Checkpoint: #{opts[:checkpoint]}

Next steps:
  1. Test the model: mix run scripts/test_model.exs --checkpoint #{opts[:checkpoint]}
  2. Continue training: mix run scripts/train_from_replays.exs --checkpoint #{opts[:checkpoint]}
  3. Run against CPU: mix run scripts/eval.exs

""")

# Finish Wandb run if active
if Wandb.active?() do
  Wandb.finish_run()
  IO.puts("Wandb run finished.")
end
