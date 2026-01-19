#!/usr/bin/env elixir
# Training script: Parse replays and run imitation learning
#
# Usage:
#   mix run scripts/train_from_replays.exs [options]
#   mix run scripts/train_from_replays.exs --preset quick
#   mix run scripts/train_from_replays.exs --preset mewtwo --epochs 10
#
# Presets (recommended for quick start):
#   --preset quick      - Fast iteration (1 epoch, 5 files, small MLP)
#   --preset standard   - Balanced training (10 epochs, 50 files)
#   --preset full       - Maximum quality (50 epochs, all files, Mamba)
#   --preset full_cpu   - Full training optimized for CPU (no temporal)
#
# Character Presets (includes temporal + character-tuned window sizes):
#   --preset mewtwo     - Mewtwo (window=90 for teleport recovery)
#   --preset ganondorf  - Ganondorf (window=60, spacing-focused)
#   --preset link       - Link (window=75 for projectile tracking)
#   --preset gameandwatch - Mr. Game & Watch (window=45, no L-cancel)
#   --preset zelda      - Zelda (window=60, transform tracking)
#
# Note: CLI arguments override preset values (e.g., --preset quick --epochs 3)
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
#
# Early Stopping Options:
#   --early-stopping  - Enable early stopping when validation loss stops improving
#   --patience N      - Stop after N epochs without improvement (default: 5)
#   --min-delta X     - Minimum improvement to count as progress (default: 0.01)
#
# Checkpointing Options:
#   --save-best       - Save best model when validation loss improves (default: on)
#   --save-every N    - Save checkpoint every N epochs (in addition to final)
#   --resume PATH     - Resume training from a checkpoint
#
# Learning Rate Options:
#   --lr X            - Base learning rate (default: 1e-4)
#   --lr-schedule TYPE - Schedule: constant, cosine, exponential, linear (default: constant)
#   --warmup-steps N  - Linear warmup from 0 to base LR (default: 0)
#   --decay-steps N   - Steps for decay schedules (default: 10000)
#
# Gradient Accumulation:
#   --accumulation-steps N - Accumulate gradients over N batches before update (default: 1)
#                            Effective batch size = batch_size * accumulation_steps
#                            Example: --batch-size 32 --accumulation-steps 4 = 128 effective
#
# Validation Split:
#   --val-split X     - Fraction of data for validation, 0.0-1.0 (default: 0.0 = no split)
#                       Example: --val-split 0.1 = 10% validation, 90% training

# Simple unbuffered output - all output goes to stderr which is always line-buffered
# This means `mix run script.exs` just works without any redirection tricks
defmodule Output do
  def puts(line), do: IO.puts(:stderr, line)
end

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Training.{Config, Data, EarlyStopping, Imitation}
alias ExPhil.Embeddings
alias ExPhil.Integrations.Wandb

# Parse command line arguments using Config module
# Validation runs after parsing to catch errors early (before expensive setup)
opts = System.argv()
       |> Config.parse_args()
       |> Config.ensure_checkpoint_name()
       |> Config.validate!()

# Ensure checkpoints directory exists
File.mkdir_p!("checkpoints")

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

preset_str = case opts[:preset] do
  nil -> "none (custom)"
  preset -> "#{preset}"
end

character_str = case opts[:character] do
  nil -> ""
  char -> "  Character:   #{char}\n"
end

# Extract model name from checkpoint path for display
model_name = opts[:name] || Path.basename(opts[:checkpoint], ".axon")

Output.puts("""

╔════════════════════════════════════════════════════════════════╗
║              ExPhil Imitation Learning Training                ║
╚════════════════════════════════════════════════════════════════╝

  Model Name:  #{model_name}

Configuration:
  Preset:      #{preset_str}
#{character_str}  Replays:     #{opts[:replays]}
  Epochs:      #{opts[:epochs]}
  Batch Size:  #{opts[:batch_size]}
  Hidden:      #{inspect(opts[:hidden_sizes], charlists: :as_lists)}
  Max Files:   #{opts[:max_files] || "all"}
  Player Port: #{opts[:player_port]}
  Checkpoint:  #{opts[:checkpoint]}
  Wandb:       #{if opts[:wandb], do: "enabled", else: "disabled"}
  Precision:   #{precision_str}
  Frame Delay: #{if opts[:frame_delay] > 0, do: "#{opts[:frame_delay]} frames (online simulation)", else: "0 (instant feedback)"}
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
      Output.puts("Wandb run started: #{run_id}")
    {:error, reason} ->
      Output.puts("Warning: Wandb failed to start: #{inspect(reason)}")
  end
end

# Step 1: Find and parse replays
Output.puts("Step 1: Parsing replays...")

replay_files = Path.wildcard(Path.join(opts[:replays], "**/*.slp"))
replay_files = if opts[:max_files], do: Enum.take(replay_files, opts[:max_files]), else: replay_files

Output.puts("  Found #{length(replay_files)} replay files")

# Parse replays in parallel and collect training frames
{parse_time, all_frames} = :timer.tc(fn ->
  replay_files
  |> Task.async_stream(
    fn path ->
      case Peppi.parse(path, player_port: opts[:player_port]) do
        {:ok, replay} ->
          frames = Peppi.to_training_frames(replay,
            player_port: opts[:player_port],
            frame_delay: opts[:frame_delay]
          )
          {path, length(frames), frames}
        {:error, reason} ->
          Output.puts("  ⚠ Failed to parse #{Path.basename(path)}: #{reason}")
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
    Output.puts("  Parsed #{files} files, #{frames} total frames")
    List.flatten(frame_lists)
  end)
end)

Output.puts("  Parse time: #{Float.round(parse_time / 1_000_000, 2)}s")
Output.puts("  Total training frames: #{length(all_frames)}")

if length(all_frames) == 0 do
  Output.puts("\n❌ No training frames found. Check replay files and player port.")
  System.halt(1)
end

# Step 2: Create dataset
Output.puts("\nStep 2: Creating dataset...")

dataset = Data.from_frames(all_frames)

# Convert to sequences for temporal training
base_dataset = if opts[:temporal] do
  Output.puts("  Converting to sequences (window=#{opts[:window_size]}, stride=#{opts[:stride]})...")
  seq_dataset = Data.to_sequences(dataset,
    window_size: opts[:window_size],
    stride: opts[:stride]
  )

  # Pre-compute embeddings to avoid slow per-batch embedding
  # This embeds all frames ONCE instead of on every batch
  Data.precompute_embeddings(seq_dataset)
else
  dataset
end

# Split into train/val based on val_split option
# val_split = 0.0 means no validation set, val_split = 0.1 means 10% validation
{train_dataset, val_dataset} = if opts[:val_split] > 0.0 do
  train_ratio = 1.0 - opts[:val_split]
  Data.split(base_dataset, ratio: train_ratio)
else
  # No validation split - use all data for training, create empty val dataset
  {base_dataset, Data.empty(base_dataset)}
end

data_type = if opts[:temporal], do: "sequences", else: "frames"
Output.puts("  Training #{data_type}: #{train_dataset.size}")
if val_dataset.size > 0 do
  Output.puts("  Validation #{data_type}: #{val_dataset.size} (#{Float.round(opts[:val_split] * 100, 1)}%)")
else
  Output.puts("  Validation: disabled (--val-split 0.0)")
end

# Show some statistics
stats = Data.stats(train_dataset)
Output.puts("\n  Button press rates:")
for {button, rate} <- Enum.sort(stats.button_rates) do
  bar = String.duplicate("█", round(rate * 50))
  Output.puts("    #{button |> to_string() |> String.pad_trailing(6)}: #{bar} #{Float.round(rate * 100, 1)}%")
end

# Step 3: Initialize trainer
Output.puts("\nStep 3: Initializing model...")

embed_size = Embeddings.embedding_size()
Output.puts("  Embedding size: #{embed_size}")

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
  learning_rate: opts[:learning_rate],
  lr_schedule: opts[:lr_schedule],
  warmup_steps: opts[:warmup_steps],
  decay_steps: opts[:decay_steps],
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
  conv_size: opts[:conv_size],
  # Gradient accumulation
  accumulation_steps: opts[:accumulation_steps]
]

# Create trainer (or load from checkpoint for resumption)
{trainer, resumed_step} = if opts[:resume] do
  Output.puts("  Resuming from checkpoint: #{opts[:resume]}")
  base_trainer = Imitation.new(trainer_opts)
  case Imitation.load_checkpoint(base_trainer, opts[:resume]) do
    {:ok, loaded_trainer} ->
      Output.puts("  ✓ Loaded checkpoint at step #{loaded_trainer.step}")
      {loaded_trainer, loaded_trainer.step}
    {:error, reason} ->
      Output.puts("  ✗ Failed to load checkpoint: #{inspect(reason)}")
      System.halt(1)
  end
else
  {Imitation.new(trainer_opts), 0}
end

if opts[:temporal] do
  Output.puts("  Temporal model initialized (#{opts[:backbone]} backbone)")
else
  Output.puts("  MLP model initialized")
end
Output.puts("  Batch size: #{opts[:batch_size]}")
if opts[:accumulation_steps] > 1 do
  effective_batch = opts[:batch_size] * opts[:accumulation_steps]
  Output.puts("  Gradient accumulation: #{opts[:accumulation_steps]}x (effective batch: #{effective_batch})")
end
if resumed_step > 0 do
  Output.puts("  Resumed at step: #{resumed_step}")
end

# Time estimation
# Based on empirical measurements: ~0.1s per batch after JIT, 3-5min for JIT compilation
batches_per_epoch = div(train_dataset.size, opts[:batch_size])
jit_overhead_sec = 300  # ~5 minutes for first JIT compilation
seconds_per_batch = if opts[:temporal], do: 0.5, else: 0.1  # Temporal is slower
estimated_train_sec = (batches_per_epoch * opts[:epochs] * seconds_per_batch) + jit_overhead_sec
estimated_minutes = div(trunc(estimated_train_sec), 60)
estimated_remaining = rem(trunc(estimated_train_sec), 60)

Output.puts("")
Output.puts("  ⏱  Estimated training time: ~#{estimated_minutes}m #{estimated_remaining}s")
Output.puts("      (#{batches_per_epoch} batches/epoch × #{opts[:epochs]} epochs + JIT compilation)")

# Step 4: Training loop
early_stopping_msg = if opts[:early_stopping] do
  " (early stopping: patience=#{opts[:patience]}, min_delta=#{opts[:min_delta]})"
else
  ""
end
Output.puts("\nStep 4: Training for #{opts[:epochs]} epochs#{early_stopping_msg}...")
Output.puts("─" |> String.duplicate(60))

start_time = System.monotonic_time(:second)

# Initialize early stopping state if enabled
early_stopping_state = if opts[:early_stopping] do
  EarlyStopping.init(patience: opts[:patience], min_delta: opts[:min_delta])
else
  nil
end

# Training loop with early stopping and best model tracking
# Returns {trainer, epochs_completed, stopped_early, early_stopping_state, best_val_loss}
initial_state = {trainer, 0, false, early_stopping_state, nil}

{final_trainer, epochs_completed, stopped_early, _es_state, _best_val} =
  Enum.reduce_while(1..opts[:epochs], initial_state, fn epoch, {current_trainer, _, _, es_state, best_val_loss} ->
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
      Output.puts("\n  ─── Epoch #{epoch}/#{opts[:epochs]} ───")
      Output.puts("  Starting #{num_batches} batches...")
    end

    # Train epoch with JIT compilation indicator
    jit_indicator_shown = if epoch == 1 do
      Output.puts("  ─── Epoch 1/#{opts[:epochs]} ───")
      Output.puts("  ⏳ JIT compiling model (first batch)... this may take 2-5 minutes")
      Output.puts("     (subsequent batches will be fast)")
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
        Output.puts("\n  ✓ JIT compilation complete (took #{Float.round(batch_time_ms / 1000, 1)}s)")
        Output.puts("    Training started - progress updates every 5%\n")
        true  # Keep as true, we'll handle first progress differently
      else
        jit_shown
      end

      # Progress indicator: every 2% OR at least every 50 batches (frequent updates)
      progress_interval = max(1, min(50, div(num_batches, 50)))
      show_progress = (rem(batch_idx + 1, progress_interval) == 0) and batch_idx > 0

      if show_progress do
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
        Output.puts(progress_line)
      end

      {new_trainer, [metrics.loss | losses], new_jit_shown}
    end)

    epoch_time = System.monotonic_time(:second) - epoch_start
    avg_loss = Enum.sum(epoch_losses) / length(epoch_losses)

    # Validation - only if we have validation data
    {val_loss, val_metrics} = if val_dataset.size > 0 do
      val_batches = if opts[:temporal] do
        Data.batched_sequences(val_dataset, batch_size: opts[:batch_size], shuffle: false)
      else
        Data.batched(val_dataset, batch_size: opts[:batch_size], shuffle: false)
      end
      metrics = Imitation.evaluate(updated_trainer, val_batches)
      {metrics.loss, metrics}
    else
      # No validation set - use training loss as proxy
      {avg_loss, %{loss: avg_loss}}
    end

    # Check early stopping (uses val_loss or train_loss if no validation)
    {new_es_state, es_decision, es_message} = if es_state do
      {new_state, decision} = EarlyStopping.check(es_state, val_loss)
      {new_state, decision, " | #{EarlyStopping.status_message(new_state)}"}
    else
      {nil, :continue, ""}
    end

    Output.puts("")
    if val_dataset.size > 0 do
      Output.puts("  ✓ Epoch #{epoch} complete: train_loss=#{Float.round(avg_loss, 4)} val_loss=#{Float.round(val_loss, 4)} (#{epoch_time}s)#{es_message}")
    else
      Output.puts("  ✓ Epoch #{epoch} complete: train_loss=#{Float.round(avg_loss, 4)} (#{epoch_time}s)#{es_message}")
    end

    # Save best model if this is the best loss so far (val_loss if available, else train_loss)
    is_new_best = best_val_loss == nil or val_loss < best_val_loss
    new_best_val_loss = if is_new_best, do: val_loss, else: best_val_loss

    if opts[:save_best] and is_new_best do
      best_checkpoint_path = Config.derive_best_checkpoint_path(opts[:checkpoint])
      best_policy_path = Config.derive_best_policy_path(opts[:checkpoint])

      loss_type = if val_dataset.size > 0, do: "val_loss", else: "train_loss"
      case Imitation.save_checkpoint(updated_trainer, best_checkpoint_path) do
        :ok ->
          case Imitation.export_policy(updated_trainer, best_policy_path) do
            :ok -> Output.puts("    ★ New best model saved (#{loss_type}=#{Float.round(val_loss, 4)})")
            {:error, _} -> Output.puts("    ★ Best checkpoint saved, policy export failed")
          end
        {:error, reason} ->
          Output.puts("    ⚠ Failed to save best model: #{inspect(reason)}")
      end
    end

    # Save periodic checkpoint if configured
    if opts[:save_every] && rem(epoch, opts[:save_every]) == 0 do
      epoch_checkpoint = String.replace(opts[:checkpoint], ".axon", "_epoch#{epoch}.axon")
      case Imitation.save_checkpoint(updated_trainer, epoch_checkpoint) do
        :ok -> Output.puts("    📁 Epoch #{epoch} checkpoint saved")
        {:error, _} -> :ok
      end
    end

    # Decide whether to continue or stop
    case es_decision do
      :stop ->
        Output.puts("\n  ⚠ Early stopping triggered - no improvement for #{opts[:patience]} epochs")
        {:halt, {updated_trainer, epoch, true, new_es_state, new_best_val_loss}}
      :continue ->
        {:cont, {updated_trainer, epoch, false, new_es_state, new_best_val_loss}}
    end
  end)

total_time = System.monotonic_time(:second) - start_time
total_min = div(total_time, 60)
total_sec = rem(total_time, 60)
Output.puts("")
Output.puts("─" |> String.duplicate(60))
Output.puts("✓ Training complete in #{total_min}m #{total_sec}s")
Output.puts("─" |> String.duplicate(60))

# Step 5: Save checkpoint
Output.puts("\nStep 5: Saving checkpoint...")
case Imitation.save_checkpoint(final_trainer, opts[:checkpoint]) do
  :ok -> Output.puts("  ✓ Saved to #{opts[:checkpoint]}")
  {:error, reason} -> Output.puts("  ✗ Failed: #{inspect(reason)}")
end

# Also export policy for inference
policy_path = String.replace(opts[:checkpoint], ".axon", "_policy.bin")
case Imitation.export_policy(final_trainer, policy_path) do
  :ok -> Output.puts("  ✓ Policy exported to #{policy_path}")
  {:error, reason} -> Output.puts("  ✗ Failed: #{inspect(reason)}")
end

# Save training config as JSON (for reproducibility)
config_path = Config.derive_config_path(opts[:checkpoint])
training_results = %{
  embed_size: embed_size,
  training_frames: train_dataset.size,
  validation_frames: val_dataset.size,
  total_time_seconds: total_time,
  final_training_loss: Float.round(Enum.sum(Enum.take(final_trainer.metrics.loss, 10)) / 10, 4),
  epochs_completed: epochs_completed,
  stopped_early: stopped_early
}
training_config = Config.build_config_json(opts, training_results)

case File.write(config_path, Jason.encode!(training_config, pretty: true)) do
  :ok -> Output.puts("  ✓ Config saved to #{config_path}")
  {:error, reason} -> Output.puts("  ✗ Config save failed: #{inspect(reason)}")
end

# Summary
early_stop_note = if stopped_early, do: " (stopped early)", else: ""
best_model_note = if opts[:save_best] do
  best_policy = Config.derive_best_policy_path(opts[:checkpoint])
  "  Best model: #{best_policy}\n"
else
  ""
end

Output.puts("""

╔════════════════════════════════════════════════════════════════╗
║                      Training Complete!                        ║
╚════════════════════════════════════════════════════════════════╝

Summary:
  Replays parsed: #{length(replay_files)}
  Training frames: #{train_dataset.size}
  Epochs completed: #{epochs_completed}/#{opts[:epochs]}#{early_stop_note}
  Final training loss: #{Float.round(Enum.sum(Enum.take(final_trainer.metrics.loss, 10)) / 10, 4)}
  Checkpoint: #{opts[:checkpoint]}
#{best_model_note}
Next steps:
  1. Test the model: mix run scripts/test_model.exs --checkpoint #{opts[:checkpoint]}
  2. Continue training: mix run scripts/train_from_replays.exs --checkpoint #{opts[:checkpoint]}
  3. Run against CPU: mix run scripts/eval.exs

""")

# Finish Wandb run if active
if Wandb.active?() do
  Wandb.finish_run()
  Output.puts("Wandb run finished.")
end
