#!/usr/bin/env elixir
# Training script: Parse replays and run imitation learning
#
# Usage:
#   mix run scripts/train_from_replays.exs [options]
#
# Performance Tips:
#   - XLA multi-threading is enabled by default (2-3x faster on multi-core CPUs)
#   - Larger batch sizes (128, 256) reduce per-batch overhead if RAM allows
#
# Options:

# Enable XLA multi-threading for CPU training (2-3x speedup on multi-core)
# This must be set BEFORE any EXLA/XLA operations are performed
xla_flags = System.get_env("XLA_FLAGS", "")
unless String.contains?(xla_flags, "xla_cpu_multi_thread_eigen") do
  new_flags = "#{xla_flags} --xla_cpu_multi_thread_eigen=true" |> String.trim()
  System.put_env("XLA_FLAGS", new_flags)
end
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
#   --backbone TYPE   - Backbone: sliding_window, hybrid, lstm, gru, mamba, mlp (default: sliding_window)
#   --window-size N   - Frames in attention window (default: 60)
#   --stride N        - Stride for sequence sampling (default: 1)
#   --truncate-bptt N - Truncate backprop through time to last N steps (default: full)
#   --num-layers N    - Number of backbone layers (default: 2)
#
# Mamba-specific Options (when --backbone mamba):
#   --state-size N    - State dimension for SSM (default: 16)
#   --expand-factor N - Expansion factor for inner dimension (default: 2)
#   --conv-size N     - Causal conv kernel size (default: 4)
#
# Precision Options:
#   --precision TYPE  - Tensor precision: bf16 (default) or f32
#                       BF16 gives ~2x speedup with minimal accuracy loss

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
  stride: String.to_integer(get_arg.("--stride", "1")),
  num_layers: String.to_integer(get_arg.("--num-layers", "2")),
  # Mamba-specific options
  state_size: String.to_integer(get_arg.("--state-size", "16")),
  expand_factor: String.to_integer(get_arg.("--expand-factor", "2")),
  conv_size: String.to_integer(get_arg.("--conv-size", "4")),
  truncate_bptt: case Enum.find_index(args, &(&1 == "--truncate-bptt")) do
    nil -> nil  # Full BPTT
    idx -> String.to_integer(Enum.at(args, idx + 1))
  end,
  # Precision option (bf16 default for ~2x speedup)
  precision: case get_arg.("--precision", "bf16") do
    "f32" -> :f32
    "bf16" -> :bf16
    other -> raise "Unknown precision: #{other}. Use 'bf16' or 'f32'"
  end
]

temporal_info = if opts[:temporal] do
  bptt_info = if opts[:truncate_bptt] do
    "truncated to last #{opts[:truncate_bptt]} steps"
  else
    "full"
  end

  backbone_extra = if opts[:backbone] == :mamba do
    "  SSM State:   #{opts[:state_size]}\n  Expand:      #{opts[:expand_factor]}x\n  Conv Size:   #{opts[:conv_size]}\n"
  else
    ""
  end

  "  Temporal:    enabled\n  Backbone:    #{opts[:backbone]}\n  Layers:      #{opts[:num_layers]}\n  Window:      #{opts[:window_size]} frames\n  Stride:      #{opts[:stride]}\n  BPTT:        #{bptt_info}\n#{backbone_extra}"
else
  "  Temporal:    disabled (single-frame MLP)\n"
end

precision_str = if opts[:precision] == :bf16, do: "bf16 (faster)", else: "f32 (full precision)"

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
  Precision:   #{precision_str}
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

  # Pre-compute embeddings to avoid slow per-batch embedding
  # This embeds all frames ONCE instead of on every batch
  seq_dataset = Data.precompute_embeddings(seq_dataset)

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

# For Mamba, use first hidden_size value as hidden_size (single int)
# For other backbones, use hidden_sizes list
hidden_size = case opts[:hidden_sizes] do
  [h | _] -> h
  _ -> 256
end

trainer_opts = [
  embed_size: embed_size,
  hidden_sizes: opts[:hidden_sizes],
  hidden_size: hidden_size,
  learning_rate: 1.0e-4,
  batch_size: opts[:batch_size],
  precision: opts[:precision],
  # Temporal options
  temporal: opts[:temporal],
  backbone: opts[:backbone],
  window_size: opts[:window_size],
  num_heads: 2,      # Smaller for CPU training
  head_dim: 32,      # Smaller for CPU training
  num_layers: opts[:num_layers],
  truncate_bptt: opts[:truncate_bptt],
  # Mamba-specific options
  state_size: opts[:state_size],
  expand_factor: opts[:expand_factor],
  conv_size: opts[:conv_size]
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

  # Epoch start message
  if epoch > 1 do
    IO.puts("\n  ─── Epoch #{epoch}/#{opts[:epochs]} ───")
    IO.puts("  Starting #{num_batches} batches...")
  end

  # Train epoch with JIT compilation indicator
  jit_indicator_shown = if epoch == 1 do
    IO.puts("  ─── Epoch 1/#{opts[:epochs]} ───")
    IO.puts("  ⏳ JIT compiling model (first batch)... this may take 2-5 minutes")
    IO.puts("     (subsequent batches will be fast)")
    true
  else
    false
  end

  # Track timing for ETA calculation
  epoch_batch_start = System.monotonic_time(:millisecond)

  {updated_trainer, epoch_losses, _} = Enum.reduce(Enum.with_index(batches), {current_trainer, [], jit_indicator_shown}, fn {batch, batch_idx}, {t, losses, jit_shown} ->
    batch_start = System.monotonic_time(:millisecond)
    # Note: loss_fn is ignored by train_step (it uses cached predict_fn internally)
    {new_trainer, metrics} = Imitation.train_step(t, batch, nil)
    batch_time_ms = System.monotonic_time(:millisecond) - batch_start

    # Show JIT completion message after first batch
    new_jit_shown = if jit_shown and batch_idx == 0 do
      IO.puts("\n  ✓ JIT compilation complete (took #{Float.round(batch_time_ms / 1000, 1)}s)")
      IO.puts("    Training started - progress updates every 5%\n")
      true  # Keep as true, we'll handle first progress differently
    else
      jit_shown
    end

    # Progress indicator every 5% (more frequent for better feedback)
    progress_interval = max(1, div(num_batches, 20))
    show_progress = rem(batch_idx + 1, progress_interval) == 0 or batch_idx == 0

    if show_progress and batch_idx > 0 do
      pct = round((batch_idx + 1) / num_batches * 100)
      elapsed_total_ms = System.monotonic_time(:millisecond) - epoch_batch_start
      avg_batch_ms = elapsed_total_ms / (batch_idx + 1)
      remaining_batches = num_batches - (batch_idx + 1)
      eta_sec = round(remaining_batches * avg_batch_ms / 1000)
      eta_min = div(eta_sec, 60)
      eta_sec_rem = rem(eta_sec, 60)

      # Format: Epoch 1: ████████░░ 40% | batch 642/1606 | loss: 0.1234 | 0.5s/batch | ETA: 8m 12s
      bar_width = 10
      filled = round(pct / 100 * bar_width)
      bar = String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled)

      progress_line = "  Epoch #{epoch}: #{bar} #{pct}% | batch #{batch_idx + 1}/#{num_batches} | loss: #{Float.round(metrics.loss, 4)} | #{Float.round(avg_batch_ms / 1000, 2)}s/batch | ETA: #{eta_min}m #{eta_sec_rem}s"
      IO.puts(progress_line)

      # Force flush
      Process.sleep(1)
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

  IO.puts("")
  IO.puts("  ✓ Epoch #{epoch} complete: train_loss=#{Float.round(avg_loss, 4)} val_loss=#{Float.round(val_metrics.loss, 4)} (#{epoch_time}s)")

  updated_trainer
end)

total_time = System.monotonic_time(:second) - start_time
total_min = div(total_time, 60)
total_sec = rem(total_time, 60)
IO.puts("")
IO.puts("─" |> String.duplicate(60))
IO.puts("✓ Training complete in #{total_min}m #{total_sec}s")
IO.puts("─" |> String.duplicate(60))

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
