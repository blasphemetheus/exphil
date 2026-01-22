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
#
# Data Augmentation:
#   --augment         - Enable data augmentation (mirror + noise)
#   --mirror-prob X   - Probability of horizontal flip (default: 0.5)
#   --noise-prob X    - Probability of noise injection (default: 0.3)
#   --noise-scale X   - Scale of Gaussian noise (default: 0.01)
#
# Label Smoothing:
#   --label-smoothing X - Smoothing factor for targets (default: 0.0, typical: 0.1)
#                         Reduces overconfidence by softening hard labels
#
# Model Registry:
#   --no-register       - Skip registering model in registry (for test runs)
#
# Checkpoint Pruning:
#   --keep-best N       - Keep only the best N epoch checkpoints (by val/train loss)
#                         Requires --save-every to have any effect
#
# Model EMA (Exponential Moving Average):
#   --ema               - Enable EMA weight tracking (often better generalization)
#   --ema-decay X       - EMA decay rate (default: 0.999, range: 0-1)
#                         Higher = slower updates, smoother weights

# Use the training output module for colored output, progress bars, and formatting
alias ExPhil.Training.Output

# Build auto-tags based on training configuration
build_model_tags = fn opts ->
  tags = []

  # Backbone tag
  tags = if opts[:backbone], do: [to_string(opts[:backbone]) | tags], else: tags

  # Temporal tag
  tags = if opts[:temporal], do: ["temporal" | tags], else: tags

  # Augmentation tag
  tags = if opts[:augment], do: ["augmented" | tags], else: tags

  # Preset tag
  tags = if opts[:preset], do: ["preset:#{opts[:preset]}" | tags], else: tags

  # Label smoothing tag
  tags = if opts[:label_smoothing] && opts[:label_smoothing] > 0, do: ["smoothed" | tags], else: tags

  Enum.reverse(tags)
end

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Training.{Augmentation, CheckpointPruning, Config, Data, EarlyStopping, EMA, GPUUtils, Imitation, Plots, Prefetcher, Recovery, Registry, ReplayValidation}
alias ExPhil.Embeddings
alias ExPhil.Integrations.Wandb

# Helper function for dual-port training - parses both players from a replay
parse_dual_port = fn path, frame_delay ->
  # Get metadata to find which ports have players
  case Peppi.metadata(path) do
    {:ok, meta} ->
      ports = Enum.map(meta.players, & &1.port)

      # Parse frames for each port and combine
      all_frames = Enum.flat_map(ports, fn port ->
        case Peppi.parse(path, player_port: port) do
          {:ok, replay} ->
            Peppi.to_training_frames(replay,
              player_port: port,
              frame_delay: frame_delay
            )
          {:error, _} ->
            []
        end
      end)

      {:ok, path, length(all_frames), all_frames}

    {:error, reason} ->
      {:error, path, reason}
  end
end

# Parse command line arguments using Config module
# First validate that all flags are recognized (catches typos early)
# Then validation runs after parsing to catch value errors (before expensive setup)
args = System.argv()
Config.validate_args!(args)

opts = args
       |> Config.parse_args()
       |> Config.ensure_checkpoint_name()
       |> Config.validate!()

# Ensure checkpoints directory exists
File.mkdir_p!("checkpoints")

# Check for incomplete training run and offer to resume
opts = case Recovery.check_incomplete(opts[:checkpoint]) do
  {:incomplete, state} ->
    Output.puts("")
    Output.puts("⚠️  " <> Recovery.format_incomplete_info(state))
    Output.puts("")

    resume_checkpoint = Recovery.get_resume_checkpoint(opts[:checkpoint])

    if resume_checkpoint do
      Output.puts("  Found checkpoint: #{resume_checkpoint}")
      Output.puts("  Resume training? [Y/n] ")

      case IO.gets("") |> String.trim() |> String.downcase() do
        response when response in ["", "y", "yes"] ->
          Output.puts("  → Resuming from #{resume_checkpoint}")
          # Update opts to resume from the checkpoint
          Keyword.put(opts, :resume, resume_checkpoint)
        _ ->
          Output.puts("  → Starting fresh (marker will be overwritten)")
          opts
      end
    else
      Output.puts("  No checkpoint found to resume from. Starting fresh.")
      opts
    end

  :ok ->
    opts
end

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

character_str = cond do
  opts[:train_character] ->
    "  Train Char:  #{opts[:train_character]} (auto-select port)\n"
  opts[:character] ->
    "  Character:   #{opts[:character]}\n"
  true ->
    ""
end

# Format frame delay display
format_frame_delay = fn opts ->
  cond do
    opts[:frame_delay_augment] ->
      "augmented (#{opts[:frame_delay_min]}-#{opts[:frame_delay_max]} frames)"
    opts[:frame_delay] > 0 ->
      "#{opts[:frame_delay]} frames (online simulation)"
    true ->
      "0 (instant feedback)"
  end
end

# Extract model name from checkpoint path for display
model_name = opts[:name] || Path.basename(opts[:checkpoint], ".axon")

# Get GPU info for display
gpu_info = case GPUUtils.device_name() do
  {:ok, name} ->
    case GPUUtils.get_memory_info() do
      {:ok, %{total_mb: total}} -> "#{name} (#{GPUUtils.format_mb(total)})"
      _ -> name
    end
  {:error, _} -> "N/A (CPU mode)"
end

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
  Player Port: #{if opts[:dual_port], do: "both (dual-port)", else: opts[:player_port]}
  Checkpoint:  #{opts[:checkpoint]}
  Wandb:       #{if opts[:wandb], do: "enabled", else: "disabled"}
  Precision:   #{precision_str}
  Frame Delay: #{format_frame_delay.(opts)}
  Augment:     #{if opts[:augment], do: "enabled (mirror=#{opts[:mirror_prob]}, noise=#{opts[:noise_prob]})", else: "disabled"}
  Prefetch:    #{if opts[:prefetch], do: "enabled (buffer=#{opts[:prefetch_buffer]})", else: "disabled"}
  Grad Ckpt:   #{if opts[:gradient_checkpoint], do: "enabled (every #{opts[:checkpoint_every]} layers)", else: "disabled"}
  GPU:         #{gpu_info}
#{temporal_info}
""")

# Show config diff from defaults (helps catch config mistakes)
case Config.format_diff(opts) do
  nil -> :ok
  diff ->
    Output.puts("Settings changed from defaults:")
    Output.puts_raw(diff)
    Output.puts("")
end

# Warn if frame delay augmentation is used with temporal mode
# (frame delay augmentation only works for single-frame batching)
if opts[:frame_delay_augment] and opts[:temporal] do
  Output.warning("Frame delay augmentation (--frame-delay-augment/--online-robust) " <>
    "only works with non-temporal training. Using fixed frame_delay=#{opts[:frame_delay]} instead.")
end

# Dry run mode - validate config and show what would happen, then exit
if opts[:dry_run] do
  Output.section("Dry Run Mode")
  Output.success("Configuration is valid")
  Output.puts_raw("")
  Output.kv("Replay files", "#{length(Path.wildcard(Path.join(opts[:replays], "**/*.slp")))} found")
  Output.kv("Would train for", "#{opts[:epochs]} epochs")
  Output.kv("Checkpoint", opts[:checkpoint])

  if opts[:wandb] do
    Output.kv("Wandb", "would log to project '#{opts[:wandb_project]}'")
  end

  Output.puts_raw("")
  Output.puts("Run without --dry-run to start training", :cyan)
  System.halt(0)
end

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

# Step 1: Find and validate replays
Output.puts("Step 1: Finding replays...", :cyan)

initial_replay_files = Path.wildcard(Path.join(opts[:replays], "**/*.slp"))
initial_count = length(initial_replay_files)
Output.puts("  Found #{initial_count} replay files")

# Quick validation to filter out obviously bad files early
replay_files = if initial_count > 0 and initial_count <= 5000 do
  Output.puts("  Validating replay files...")
  {:ok, validated_files, validation_stats} = ReplayValidation.validate(initial_replay_files,
    show_progress: false,
    verbose: false
  )

  if validation_stats.invalid > 0 do
    Output.warning("#{validation_stats.invalid} invalid replay files will be skipped")
    if validation_stats.invalid <= 5 do
      # Show details for small number of errors
      Enum.each(Enum.take(validation_stats.errors, 5), fn {path, reason} ->
        Output.puts_raw("    - #{Path.basename(path)}: #{reason}")
      end)
    end
  end

  validated_files
else
  if initial_count > 5000 do
    Output.puts("  Skipping validation for large dataset (#{initial_count} files)")
  end
  initial_replay_files
end

# Filter by character/stage if specified (uses fast metadata parsing)
# Also collect stats for display
character_filter = opts[:characters] || []
stage_filter = opts[:stages] || []

# Stage ID to name mapping for stats display
stage_id_to_name = %{
  2 => "Fountain of Dreams",
  3 => "Pokemon Stadium",
  8 => "Yoshi's Story",
  28 => "Dream Land",
  31 => "Battlefield",
  32 => "Final Destination"
}

# Collect metadata for all replays (fast) - used for filtering and stats
{replay_files, replay_stats} = if character_filter != [] or stage_filter != [] or initial_count <= 1000 do
  # Get canonical character names for matching
  char_names = Enum.map(character_filter, &Config.character_name/1) |> Enum.uniq()
  stage_ids = Enum.map(stage_filter, &Config.stage_id/1) |> Enum.filter(& &1)

  if character_filter != [] or stage_filter != [] do
    filter_desc = []
    filter_desc = if char_names != [], do: ["characters: #{Enum.join(char_names, ", ")}" | filter_desc], else: filter_desc
    filter_desc = if stage_ids != [] do
      stage_names = Enum.map(stage_filter, fn s ->
        case Config.stage_info(s) do
          {name, _} -> name
          nil -> to_string(s)
        end
      end) |> Enum.uniq()
      ["stages: #{Enum.join(stage_names, ", ")}" | filter_desc]
    else
      filter_desc
    end
    Output.puts("  Filtering by #{Enum.join(Enum.reverse(filter_desc), ", ")}...")
  else
    Output.puts("  Collecting replay metadata...")
  end

  # Collect metadata and filter
  # If train_character is set, dynamically find port; otherwise use fixed port
  train_character = opts[:train_character]
  target_char_name = if train_character, do: Config.character_name(train_character), else: nil
  default_port = opts[:player_port]

  {filtered, char_counts, stage_counts, port_map} = replay_files
  |> Task.async_stream(
    fn path ->
      case Peppi.metadata(path) do
        {:ok, meta} -> {:ok, path, meta}
        {:error, _} -> {:skip, path}
      end
    end,
    max_concurrency: System.schedulers_online(),
    timeout: 30_000
  )
  |> Enum.reduce({[], %{}, %{}, %{}}, fn
    {:ok, {:ok, path, meta}}, {paths, chars, stages, ports} ->
      # Determine training port - either dynamic (by character) or fixed
      {training_port, training_player} = if target_char_name do
        # Find port that has the target character
        player = Enum.find(meta.players, fn p -> p.character_name == target_char_name end)
        if player, do: {player.port, player}, else: {nil, nil}
      else
        # Use fixed port
        player = Enum.find(meta.players, fn p -> p.port == default_port end)
        {default_port, player}
      end

      # Skip replay if target character not found
      if training_player == nil do
        {paths, chars, stages, ports}
      else
        # Collect character stats for the training player
        chars = Map.update(chars, training_player.character_name, 1, & &1 + 1)

        # Collect stage stats
        stage_name = Map.get(stage_id_to_name, meta.stage, "Stage #{meta.stage}")
        stages = Map.update(stages, stage_name, 1, & &1 + 1)

        # Check filters - only check training player's character
        training_char = training_player.character_name
        char_match = char_names == [] or training_char in char_names
        stage_match = stage_ids == [] or meta.stage in stage_ids

        if char_match and stage_match do
          # Store the port for this replay (for frame parsing later)
          ports = Map.put(ports, path, training_port)
          {[path | paths], chars, stages, ports}
        else
          {paths, chars, stages, ports}
        end
      end

    _, acc -> acc
  end)

  filtered = Enum.reverse(filtered)

  if character_filter != [] or stage_filter != [] do
    Output.puts("  #{length(filtered)}/#{initial_count} replays match filters")
  end

  stats = %{
    total: initial_count,
    characters: char_counts,
    stages: stage_counts,
    port_map: port_map
  }

  {filtered, stats}
else
  # Skip metadata collection for large datasets without filters
  {replay_files, nil}
end

# Apply max_files limit after filtering
replay_files = if opts[:max_files], do: Enum.take(replay_files, opts[:max_files]), else: replay_files

# Show replay stats if collected
if replay_stats do
  Output.replay_stats(replay_stats)
end

if (character_filter != [] or stage_filter != []) and opts[:max_files] do
  Output.puts("  Using #{length(replay_files)} replays for training (limited by --max-files)")
end

# Parse replays in parallel and collect training frames
# Error handling options
skip_errors = Keyword.get(opts, :skip_errors, true)
show_errors = Keyword.get(opts, :show_errors, true)
error_log = Keyword.get(opts, :error_log)

# Get port map for dynamic port selection (if train_character was used)
port_map = if replay_stats, do: Map.get(replay_stats, :port_map, %{}), else: %{}
default_port = opts[:player_port]
dual_port = opts[:dual_port] || false

if dual_port do
  Output.puts("  Dual-port mode: training on BOTH players per replay (2x data)")
end

# Initialize error log file if specified
if error_log do
  File.write!(error_log, "# ExPhil Replay Parsing Errors\n# #{DateTime.utc_now()}\n\n")
end

{parse_time, {all_frames, errors}} = :timer.tc(fn ->
  replay_files
  |> Task.async_stream(
    fn path ->
      if dual_port do
        # Dual-port: parse both ports and combine frames
        parse_dual_port.(path, opts[:frame_delay])
      else
        # Single port: use dynamic port from port_map if available
        player_port = Map.get(port_map, path, default_port)

        case Peppi.parse(path, player_port: player_port) do
          {:ok, replay} ->
            frames = Peppi.to_training_frames(replay,
              player_port: player_port,
              frame_delay: opts[:frame_delay]
            )
            {:ok, path, length(frames), frames}
          {:error, reason} ->
            {:error, path, reason}
        end
      end
    end,
    max_concurrency: System.schedulers_online(),
    timeout: :infinity
  )
  |> Enum.reduce({0, 0, [], []}, fn
    {:ok, {:ok, _path, frame_count, frames}}, {total_files, total_frames, all_frames, errors} ->
      {total_files + 1, total_frames + frame_count, [frames | all_frames], errors}

    {:ok, {:error, path, reason}}, {total_files, total_frames, all_frames, errors} ->
      error = %{path: path, reason: reason}

      # Show error if enabled
      if show_errors do
        Output.puts("  ⚠ Failed: #{Path.basename(path)}")
        Output.puts("    Reason: #{inspect(reason)}")
      end

      # Log error to file if specified
      if error_log do
        File.write!(error_log, "#{path}\n  #{inspect(reason)}\n\n", [:append])
      end

      # Fail fast if not skipping errors
      unless skip_errors do
        Output.puts("\n❌ Stopping due to error (use --skip-errors to continue)")
        Output.puts("  File: #{path}")
        Output.puts("  Reason: #{inspect(reason)}")
        System.halt(1)
      end

      {total_files, total_frames, all_frames, [error | errors]}

    {:exit, reason}, {total_files, total_frames, all_frames, errors} ->
      # Task crashed - treat as error
      error = %{path: "unknown", reason: {:exit, reason}}
      {total_files, total_frames, all_frames, [error | errors]}
  end)
  |> then(fn {files, frames, frame_lists, errors} ->
    Output.puts("  Parsed #{files} files, #{frames} total frames")
    if length(errors) > 0 do
      Output.puts("  ⚠ #{length(errors)} files failed to parse")
    end
    {List.flatten(frame_lists), Enum.reverse(errors)}
  end)
end)

Output.puts("  Parse time: #{Float.round(parse_time / 1_000_000, 2)}s")
Output.puts("  Total training frames: #{length(all_frames)}")

# Show error summary if there were failures
if length(errors) > 0 do
  Output.puts("\n  Error Summary (#{length(errors)} failed files):")
  if show_errors do
    errors
    |> Enum.take(10)  # Show first 10
    |> Enum.each(fn %{path: path, reason: reason} ->
      Output.puts("    - #{Path.basename(path)}: #{inspect(reason)}")
    end)
    if length(errors) > 10 do
      Output.puts("    ... and #{length(errors) - 10} more")
    end
  end
  if error_log do
    Output.puts("  Full error log: #{error_log}")
  end
end

if length(all_frames) == 0 do
  Output.puts("\n❌ No training frames found. Check replay files and player port.")
  System.halt(1)
end

# Step 2: Create dataset
Output.puts("\nStep 2: Creating dataset...", :cyan)

dataset = Data.from_frames(all_frames)

# Convert to sequences for temporal training, or precompute embeddings for MLP
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
  # For MLP training, optionally precompute frame embeddings
  if opts[:precompute] do
    Output.puts("  Pre-computing embeddings (2-3x speedup)...")
    Data.precompute_frame_embeddings(dataset)
  else
    dataset
  end
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
Output.puts("\nStep 3: Initializing model...", :cyan)

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
  accumulation_steps: opts[:accumulation_steps],
  # Label smoothing
  label_smoothing: opts[:label_smoothing],
  # Layer normalization for MLP backbone
  layer_norm: opts[:layer_norm],
  # Optimizer selection
  optimizer: opts[:optimizer],
  # Gradient checkpointing (memory vs compute trade-off)
  gradient_checkpoint: opts[:gradient_checkpoint],
  checkpoint_every: opts[:checkpoint_every]
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

# Count parameters and show estimates
param_count = GPUUtils.count_params(trainer.policy_params)
Output.puts("  Parameters: #{Float.round(param_count / 1_000_000, 2)}M")

# Check for checkpoint size warning
case GPUUtils.check_checkpoint_size_warning(param_count: param_count, threshold_mb: 500) do
  {:warning, msg} -> Output.warning(msg)
  :ok -> :ok
end

# Estimate GPU memory requirement and check free memory
estimated_mem_mb = GPUUtils.estimate_memory_mb(
  param_count: param_count,
  batch_size: opts[:batch_size],
  precision: opts[:precision],
  temporal: opts[:temporal],
  window_size: opts[:window_size]
)

case GPUUtils.check_free_memory(required_mb: estimated_mem_mb) do
  {:warning, msg} -> Output.warning(msg)
  :ok -> :ok
  {:error, _} -> :ok  # No GPU, skip check
end
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
# Create augmentation function if enabled
augment_fn = if opts[:augment] do
  fn frame ->
    Augmentation.augment(frame,
      mirror_prob: opts[:mirror_prob],
      noise_prob: opts[:noise_prob],
      noise_scale: opts[:noise_scale]
    )
  end
else
  nil
end

early_stopping_msg = if opts[:early_stopping] do
  " (early stopping: patience=#{opts[:patience]}, min_delta=#{opts[:min_delta]})"
else
  ""
end
Output.puts("\nStep 4: Training for #{opts[:epochs]} epochs#{early_stopping_msg}...", :cyan)
Output.divider()

# Create incomplete marker for crash recovery
Recovery.mark_started(opts[:checkpoint], opts)

start_time = System.monotonic_time(:second)

# Initialize early stopping state if enabled
early_stopping_state = if opts[:early_stopping] do
  EarlyStopping.init(patience: opts[:patience], min_delta: opts[:min_delta])
else
  nil
end

# Initialize checkpoint pruner if configured
pruner = if opts[:keep_best] && opts[:save_every] do
  CheckpointPruning.new(keep_best: opts[:keep_best], metric: :loss)
else
  nil
end

# Initialize EMA if configured
ema = if opts[:ema] do
  Output.puts("  EMA enabled (decay=#{opts[:ema_decay]})")
  EMA.new(trainer.policy_params, decay: opts[:ema_decay])
else
  nil
end

# Training loop with early stopping and best model tracking
# Returns {trainer, epochs_completed, stopped_early, early_stopping_state, best_val_loss, pruner, ema, history}
initial_state = {trainer, 0, false, early_stopping_state, nil, pruner, ema, []}

{final_trainer, epochs_completed, stopped_early, _es_state, _best_val, _final_pruner, final_ema, training_history} =
  Enum.reduce_while(1..opts[:epochs], initial_state, fn epoch, {current_trainer, _, _, es_state, best_val_loss, current_pruner, current_ema, history} ->
    epoch_start = System.monotonic_time(:second)

    # Create batched dataset for this epoch
    # Use appropriate batching function based on temporal mode
    # Note: augmentation is only applied to non-temporal (single-frame) batches
    # Temporal batches use pre-computed embeddings, so augmentation happens at sequence creation
    # Create batch stream (kept lazy for true async prefetching)
    batch_stream = if opts[:temporal] do
      Data.batched_sequences(train_dataset,
        batch_size: opts[:batch_size],
        shuffle: true,
        drop_last: true
      )
    else
      Data.batched(train_dataset,
        batch_size: opts[:batch_size],
        shuffle: true,
        drop_last: true,
        augment_fn: augment_fn,
        # Frame delay augmentation for online play robustness
        frame_delay: opts[:frame_delay],
        frame_delay_augment: opts[:frame_delay_augment],
        frame_delay_min: opts[:frame_delay_min],
        frame_delay_max: opts[:frame_delay_max]
      )
    end

    # Calculate number of batches for progress display
    # (we count without materializing to keep the stream lazy)
    num_batches = div(train_dataset.size, opts[:batch_size])

    # Epoch start message with GPU memory status
    gpu_status = GPUUtils.memory_status_string()

    if epoch > 1 do
      Output.puts("\n  ─── Epoch #{epoch}/#{opts[:epochs]} ───")
      Output.puts("  #{gpu_status}")

      # Check for high memory usage and warn
      case GPUUtils.check_memory_warning(threshold: 0.90) do
        {:warning, msg} -> Output.warning(msg)
        _ -> :ok
      end

      Output.puts("  Starting #{num_batches} batches...")
    end

    # Train epoch with JIT compilation indicator
    jit_indicator_shown = if epoch == 1 do
      Output.puts("  ─── Epoch 1/#{opts[:epochs]} ───")
      Output.puts("  #{gpu_status}")
      Output.puts("  ⏳ JIT compiling model (first batch)... this may take 2-5 minutes")
      Output.puts("     (subsequent batches will be fast)")
      true
    else
      false
    end

    # Track timing for ETA calculation
    epoch_batch_start = System.monotonic_time(:millisecond)

    # Define batch processing function (shared by prefetch and non-prefetch paths)
    process_batch = fn batch, batch_idx, {t, losses, jit_shown} ->
      batch_start = System.monotonic_time(:millisecond)
      # Note: loss_fn is ignored by train_step (it uses cached predict_fn internally)
      {new_trainer, metrics} = Imitation.train_step(t, batch, nil)
      batch_time_ms = System.monotonic_time(:millisecond) - batch_start

      # Show JIT completion message after first batch
      new_jit_shown = if jit_shown and batch_idx == 0 do
        Output.puts("\n  ✓ JIT compilation complete (took #{Float.round(batch_time_ms / 1000, 1)}s)")
        true
      else
        jit_shown
      end

      # Live progress bar - updates in place using carriage return
      # Update every batch for smooth progress (terminal handles the refresh)
      pct = round((batch_idx + 1) / num_batches * 100)
      elapsed_total_ms = System.monotonic_time(:millisecond) - epoch_batch_start
      avg_batch_ms = elapsed_total_ms / (batch_idx + 1)
      remaining_batches = num_batches - (batch_idx + 1)
      eta_sec = round(remaining_batches * avg_batch_ms / 1000)
      eta_min = div(eta_sec, 60)
      eta_sec_rem = rem(eta_sec, 60)

      # Format: Epoch 1: ████████░░ 40% | 642/1606 | loss: 0.1234 | 0.5s/it | ETA: 8m 12s
      bar_width = 20
      filled = round(pct / 100 * bar_width)
      bar = String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled)

      # Pad percentage to fixed width for stable display
      pct_str = pct |> Integer.to_string() |> String.pad_leading(3)

      # Convert tensor loss to number for display (metrics.loss is now a tensor)
      loss_val = Nx.to_number(metrics.loss)
      progress_line = "  Epoch #{epoch}: #{bar} #{pct_str}% | #{batch_idx + 1}/#{num_batches} | loss: #{Float.round(loss_val, 4)} | #{Float.round(avg_batch_ms / 1000, 2)}s/it | ETA: #{eta_min}m #{eta_sec_rem}s"

      # Use carriage return to overwrite line (no newline until epoch complete)
      # Write directly to stderr to bypass Output module's timestamp
      IO.write(:stderr, "\r#{progress_line}")

      # Print newline at end of epoch
      if batch_idx + 1 == num_batches do
        IO.write(:stderr, "\n")
      end

      # Accumulate tensor loss (already converted for display above, so use loss_val)
      {new_trainer, [loss_val | losses], new_jit_shown}
    end

    # Use prefetcher if enabled (computes next batch while GPU trains on current)
    {updated_trainer, epoch_losses, _} = if opts[:prefetch] do
      # Use streaming prefetcher for true async overlap
      Prefetcher.reduce_stream_indexed(
        batch_stream,
        {current_trainer, [], jit_indicator_shown},
        process_batch,
        buffer_size: opts[:prefetch_buffer]
      )
    else
      # Standard sequential processing (materializes batches)
      batches = Enum.to_list(batch_stream)
      Enum.reduce(Enum.with_index(batches), {current_trainer, [], jit_indicator_shown}, fn {batch, batch_idx}, acc ->
        process_batch.(batch, batch_idx, acc)
      end)
    end

    epoch_time = System.monotonic_time(:second) - epoch_start
    # epoch_losses is now a list of numbers (converted during progress display)
    avg_loss = Enum.sum(epoch_losses) / length(epoch_losses)

    # Validation - only if we have validation data
    {val_loss, _val_metrics} = if val_dataset.size > 0 do
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

    # Track epoch metrics for loss plot
    epoch_entry = %{
      epoch: epoch,
      train_loss: avg_loss,
      val_loss: if(val_dataset.size > 0, do: val_loss, else: nil),
      time_seconds: epoch_time
    }
    updated_history = [epoch_entry | history]

    Output.puts("")
    if val_dataset.size > 0 do
      Output.puts("  ✓ Epoch #{epoch} complete: train_loss=#{Float.round(avg_loss, 4)} val_loss=#{Float.round(val_loss, 4)} (#{epoch_time}s)#{es_message}")
    else
      Output.puts("  ✓ Epoch #{epoch} complete: train_loss=#{Float.round(avg_loss, 4)} (#{epoch_time}s)#{es_message}")
    end

    # Update incomplete marker for crash recovery
    Recovery.mark_epoch_complete(opts[:checkpoint], epoch, val_loss)

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

    # Save periodic checkpoint if configured and track for pruning
    updated_pruner = if opts[:save_every] && rem(epoch, opts[:save_every]) == 0 do
      epoch_checkpoint = String.replace(opts[:checkpoint], ".axon", "_epoch#{epoch}.axon")
      case Imitation.save_checkpoint(updated_trainer, epoch_checkpoint) do
        :ok ->
          Output.puts("    📁 Epoch #{epoch} checkpoint saved")

          # Track and prune if pruner is configured
          if current_pruner do
            new_pruner = CheckpointPruning.track(current_pruner, epoch_checkpoint, val_loss, epoch: epoch)

            if CheckpointPruning.needs_pruning?(new_pruner) do
              {pruned_state, deleted} = CheckpointPruning.prune(new_pruner)
              if length(deleted) > 0 do
                Output.puts("    🗑️  Pruned #{length(deleted)} checkpoint(s)")
              end
              pruned_state
            else
              new_pruner
            end
          else
            current_pruner
          end

        {:error, _} ->
          current_pruner
      end
    else
      current_pruner
    end

    # Update EMA weights if enabled
    updated_ema = if current_ema do
      EMA.update(current_ema, updated_trainer.policy_params)
    else
      nil
    end

    # Decide whether to continue or stop
    case es_decision do
      :stop ->
        Output.puts("\n  ⚠ Early stopping triggered - no improvement for #{opts[:patience]} epochs")
        {:halt, {updated_trainer, epoch, true, new_es_state, new_best_val_loss, updated_pruner, updated_ema, updated_history}}
      :continue ->
        {:cont, {updated_trainer, epoch, false, new_es_state, new_best_val_loss, updated_pruner, updated_ema, updated_history}}
    end
  end)

total_time = System.monotonic_time(:second) - start_time
total_min = div(total_time, 60)
total_sec = rem(total_time, 60)
Output.puts("")
Output.puts_raw("─" |> String.duplicate(60))
Output.puts("✓ Training complete in #{total_min}m #{total_sec}s")
Output.puts_raw("─" |> String.duplicate(60))

# Step 5: Save checkpoint
Output.puts("\nStep 5: Saving checkpoint...", :cyan)
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

# Save EMA weights if enabled
if final_ema do
  ema_path = String.replace(opts[:checkpoint], ".axon", "_ema.bin")
  ema_binary = EMA.serialize(final_ema)
  case File.write(ema_path, ema_binary) do
    :ok -> Output.puts("  ✓ EMA weights saved to #{ema_path}")
    {:error, reason} -> Output.puts("  ✗ EMA save failed: #{inspect(reason)}")
  end

  # Also export EMA as inference policy (often better than raw weights)
  ema_policy_path = String.replace(opts[:checkpoint], ".axon", "_ema_policy.bin")
  ema_trainer = %{final_trainer | params: EMA.get_params(final_ema)}
  case Imitation.export_policy(ema_trainer, ema_policy_path) do
    :ok -> Output.puts("  ✓ EMA policy exported to #{ema_policy_path}")
    {:error, reason} -> Output.puts("  ✗ EMA policy export failed: #{inspect(reason)}")
  end
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

# Generate training loss plot
if length(training_history) > 0 do
  # History is in reverse order (most recent first)
  plot_history = Enum.reverse(training_history)
  plot_path = String.replace(opts[:checkpoint], ".axon", "_loss.html")

  try do
    Plots.save_report!(plot_history, plot_path,
      title: "Training Report: #{model_name}",
      metadata: [
        preset: opts[:preset] || "custom",
        epochs: epochs_completed,
        batch_size: opts[:batch_size],
        temporal: opts[:temporal],
        backbone: opts[:backbone]
      ]
    )
    Output.puts("  ✓ Loss plot saved to #{plot_path}")
  rescue
    e -> Output.puts("  ⚠ Loss plot generation failed: #{inspect(e)}")
  end
end

# Step 6: Register model in registry
unless opts[:no_register] do
  Output.puts("\nStep 6: Registering model...", :cyan)

  # Determine parent model if resuming from checkpoint
  parent_id = if opts[:resume] do
    case Registry.list() do
      {:ok, models} ->
        Enum.find_value(models, fn m ->
          if m.checkpoint_path == opts[:resume], do: m.id
        end)
      _ -> nil
    end
  else
    nil
  end

  registry_entry = %{
    checkpoint_path: opts[:checkpoint],
    policy_path: policy_path,
    config_path: config_path,
    training_config: opts,
    metrics: %{
      final_loss: Float.round(Enum.sum(Enum.take(final_trainer.metrics.loss, 10)) / 10, 4),
      epochs_completed: epochs_completed,
      training_frames: train_dataset.size,
      validation_frames: val_dataset.size,
      stopped_early: stopped_early,
      total_time_seconds: total_time
    },
    tags: build_model_tags.(opts),
    parent_id: parent_id
  }

  case Registry.register(registry_entry) do
    {:ok, entry} ->
      Output.puts("  ✓ Registered as '#{entry.name}' (#{entry.id})")
    {:error, reason} ->
      Output.puts("  ✗ Registry failed: #{inspect(reason)}")
  end
end

# Show terminal loss graph if training completed multiple epochs
if length(training_history) > 1 do
  Output.puts_raw("")
  Output.terminal_loss_graph(Enum.reverse(training_history), title: "Training Loss", width: 70, height: 14)
end

# Summary with colors
final_loss = Float.round(Enum.sum(Enum.take(final_trainer.metrics.loss, 10)) / 10, 4)

# Find best loss from training history
best_entry = Enum.min_by(training_history, & &1.train_loss, fn -> %{train_loss: final_loss, epoch: epochs_completed} end)
best_loss = Float.round(best_entry.train_loss, 4)
best_epoch = best_entry.epoch

Output.puts_raw("")
Output.puts_raw(Output.colorize("╔════════════════════════════════════════════════════════════════╗", :green))
Output.puts_raw(Output.colorize("║                      Training Complete!                        ║", :green))
Output.puts_raw(Output.colorize("╚════════════════════════════════════════════════════════════════╝", :green))

Output.training_summary(%{
  total_time_ms: total_time * 1000,
  epochs_completed: epochs_completed,
  epochs_total: opts[:epochs],
  total_steps: final_trainer.step,
  final_loss: final_loss,
  best_loss: best_loss,
  best_epoch: best_epoch,
  checkpoint_path: opts[:checkpoint]
})

Output.kv("Training frames", "#{train_dataset.size}")
Output.kv("Replays parsed", "#{length(replay_files)}")

if stopped_early do
  Output.puts_raw("  " <> Output.colorize("Early stopping triggered", :yellow))
end

if opts[:save_best] do
  best_policy = Config.derive_best_policy_path(opts[:checkpoint])
  Output.kv("Best policy", best_policy)
end

Output.puts_raw("")
Output.puts_raw(Output.colorize("Next steps:", :bold))
Output.puts_raw("  1. Evaluate: mix run scripts/eval_model.exs --checkpoint #{opts[:checkpoint]}")
Output.puts_raw("  2. Continue training: mix run scripts/train_from_replays.exs --resume #{opts[:checkpoint]}")
Output.puts_raw("  3. Self-play refinement: mix run scripts/train_self_play.exs --pretrained #{String.replace(opts[:checkpoint], ".axon", "_policy.bin")}")
Output.puts_raw("")

# Finish Wandb run if active
if Wandb.active?() do
  Wandb.finish_run()
  Output.puts("Wandb run finished.")
end

# Remove incomplete marker - training completed successfully
Recovery.mark_complete(opts[:checkpoint])
