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
# Memory Management:
#   --stream-chunk-size N - Load files in chunks of N to bound memory usage
#                           Enables training on large datasets without OOM
#                           Example: --stream-chunk-size 30 for 56GB RAM
#                           Trade-off: ~10-20% slower due to repeated I/O
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
# Character Balancing:
#   --balance-characters - Balance sampling by inverse character frequency
#                          Rare characters (Link, G&W) get higher sampling weights
#                          Common characters (Fox) get lower weights
#                          Useful for multi-character training:
#                          --character mewtwo,ganondorf,link --balance-characters
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
#
# Debugging:
#   EXPHIL_FULL_STACKTRACE=1  - Show full stack traces (default: simplified, hides Nx/EXLA internals)

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
  tags =
    if opts[:label_smoothing] && opts[:label_smoothing] > 0, do: ["smoothed" | tags], else: tags

  Enum.reverse(tags)
end

require Logger

alias ExPhil.Data.Peppi

alias ExPhil.Training.{
  Augmentation,
  CharacterBalance,
  CheckpointPruning,
  Config,
  Data,
  DuplicateDetector,
  EarlyStopping,
  EMA,
  GPUUtils,
  Imitation,
  Plots,
  Prefetcher,
  Recovery,
  Registry,
  ReplayQuality,
  ReplayValidation,
  Stacktrace,
  Streaming
}

alias ExPhil.Embeddings
alias ExPhil.Integrations.Wandb

# Helper function to safely round floats that might be NaN or infinity
# Returns "NaN" or "Inf" strings instead of crashing
safe_round = fn
  x, precision when is_float(x) ->
    cond do
      # NaN check: NaN != NaN
      x != x -> "NaN"
      x == :infinity or x > 1.0e38 -> "Inf"
      x == :neg_infinity or x < -1.0e38 -> "-Inf"
      true -> Float.round(x, precision)
    end

  :nan, _precision ->
    "NaN"

  :infinity, _precision ->
    "Inf"

  :neg_infinity, _precision ->
    "-Inf"

  x, precision when is_integer(x) ->
    Float.round(x * 1.0, precision)

  # Fallback for unexpected types
  x, _precision ->
    inspect(x)
end

# Helper to check if a number is NaN
is_nan? = fn
  x when is_float(x) -> x != x
  :nan -> true
  _ -> false
end

# Helper function for dual-port training - parses both players from a replay
parse_dual_port = fn path, frame_delay, min_quality ->
  # Get metadata to find which ports have players
  case Peppi.metadata(path) do
    {:ok, meta} ->
      ports = Enum.map(meta.players, & &1.port)

      # Parse the full replay once for quality scoring (if enabled)
      case Peppi.parse(path) do
        {:ok, replay} ->
          # Check quality if filtering enabled
          quality_score =
            if min_quality do
              quality_data = ReplayQuality.from_parsed_replay(replay)
              ReplayQuality.score(quality_data)
            else
              nil
            end

          if min_quality && (quality_score == :rejected or quality_score < min_quality) do
            {:quality_rejected, path, quality_score}
          else
            # Parse frames for each port and combine
            all_frames =
              Enum.flat_map(ports, fn port ->
                Peppi.to_training_frames(replay,
                  player_port: port,
                  frame_delay: frame_delay
                )
              end)

            {:ok, path, length(all_frames), all_frames, quality_score}
          end

        {:error, reason} ->
          {:error, path, reason}
      end

    {:error, reason} ->
      {:error, path, reason}
  end
end

# Parse command line arguments using Config module
# First validate that all flags are recognized (catches typos early)
# Then validation runs after parsing to catch value errors (before expensive setup)
args = System.argv()
Config.validate_args!(args)

opts =
  args
  |> Config.parse_args()
  |> Config.ensure_checkpoint_name()
  |> Config.validate!()

# Set verbosity level (affects all Output calls)
Output.set_verbosity(opts[:verbosity])

# Initialize random seed for reproducibility
seed = Config.init_seed(opts[:seed])
Output.debug("Random seed: #{seed}")

# Ensure checkpoints directory exists
File.mkdir_p!("checkpoints")

# Check for checkpoint collision (unless resuming)
unless opts[:resume] do
  case Config.check_checkpoint_path(opts[:checkpoint], overwrite: opts[:overwrite]) do
    {:ok, :new} ->
      :ok

    {:ok, :overwrite, info} ->
      Output.warning("Checkpoint '#{opts[:checkpoint]}' already exists")
      Output.puts("       #{Config.format_file_info(info)}")

      # Create backup if enabled
      if opts[:backup] do
        case Config.backup_checkpoint(opts[:checkpoint], backup_count: opts[:backup_count]) do
          {:ok, backup_path} when is_binary(backup_path) ->
            Output.puts("       Backup created: #{backup_path}")

          {:ok, nil} ->
            :ok

          {:error, reason} ->
            Output.warning("Failed to create backup: #{inspect(reason)}")
        end
      end

    {:error, :exists, info} ->
      Output.error("Checkpoint '#{opts[:checkpoint]}' already exists!")
      Output.puts("       #{Config.format_file_info(info)}")
      Output.puts("       Use --overwrite to replace, or choose a different --name")
      System.halt(1)
  end
end

# Check for incomplete training run and offer to resume
opts =
  case Recovery.check_incomplete(opts[:checkpoint]) do
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

temporal_info =
  if opts[:temporal] do
    bptt_info =
      if opts[:truncate_bptt] do
        "truncated to last #{opts[:truncate_bptt]} steps"
      else
        "full"
      end

    backbone_extra =
      if opts[:backbone] == :mamba do
        "  SSM State:   #{opts[:state_size]}\n  Expand:      #{opts[:expand_factor]}x\n  Conv Size:   #{opts[:conv_size]}\n"
      else
        ""
      end

    "  Temporal:    enabled\n  Backbone:    #{opts[:backbone]}\n  Layers:      #{opts[:num_layers]}\n  Window:      #{opts[:window_size]} frames\n  Stride:      #{opts[:stride]}\n  BPTT:        #{bptt_info}\n#{backbone_extra}"
  else
    "  Temporal:    disabled (single-frame MLP)\n"
  end

precision_str = if opts[:precision] == :bf16, do: "bf16 (faster)", else: "f32 (full precision)"

preset_str =
  case opts[:preset] do
    nil -> "none (custom)"
    preset -> "#{preset}"
  end

character_str =
  cond do
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
gpu_info =
  case GPUUtils.device_name() do
    {:ok, name} ->
      case GPUUtils.get_memory_info() do
        {:ok, %{total_mb: total}} -> "#{name} (#{GPUUtils.format_mb(total)})"
        _ -> name
      end

    {:error, _} ->
      "N/A (CPU mode)"
  end

# Format verbosity for display
verbosity_str =
  case opts[:verbosity] do
    0 -> "quiet"
    1 -> "normal"
    2 -> "verbose"
    _ -> "unknown"
  end

Output.puts("""

╔════════════════════════════════════════════════════════════════╗
║              ExPhil Imitation Learning Training                ║
╚════════════════════════════════════════════════════════════════╝

  Model Name:  #{model_name}
  Seed:        #{seed} (use --seed #{seed} to reproduce)

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
  Streaming:   #{if opts[:stream_chunk_size], do: "enabled (#{opts[:stream_chunk_size]} files/chunk)", else: "disabled"}
  Batch Save:  #{if opts[:save_every_batches], do: "every #{opts[:save_every_batches]} batches", else: "disabled"}
  K-means:     #{if opts[:kmeans_centers], do: "enabled (#{opts[:kmeans_centers]})", else: "disabled (uniform 17 buckets)"}
  Verbosity:   #{verbosity_str}
#{temporal_info}
""")

# Show config diff from defaults (helps catch config mistakes)
case Config.format_diff(opts) do
  nil ->
    :ok

  diff ->
    Output.puts("Settings changed from defaults:")
    Output.puts_raw(diff)
    Output.puts("")
end

# Warn if frame delay augmentation is used with temporal mode
# (frame delay augmentation only works for single-frame batching)
if opts[:frame_delay_augment] and opts[:temporal] do
  Output.warning(
    "Frame delay augmentation (--frame-delay-augment/--online-robust) " <>
      "only works with non-temporal training. Using fixed frame_delay=#{opts[:frame_delay]} instead."
  )
end

# Warn about streaming mode limitations
if opts[:stream_chunk_size] do
  Output.puts("")

  Output.puts(
    "📦 Streaming mode enabled: Files will be loaded in chunks of #{opts[:stream_chunk_size]}",
    :cyan
  )

  Output.puts("   Memory usage bounded regardless of total dataset size")

  if opts[:val_split] > 0.0 do
    Output.warning(
      "Validation in streaming mode uses samples from each chunk, not a global split"
    )
  end
end

# ============================================================================
# Flag combination warnings (shown before dry-run so users see them)
# ============================================================================

# Warn about early stopping without validation split
if opts[:early_stopping] and opts[:val_split] == 0.0 do
  Output.warning("--early-stopping without --val-split uses training loss")
  Output.puts("    This may not detect overfitting. Consider adding --val-split 0.1")
end

# Warn about caching with augmentation (cache would be invalid)
if opts[:cache_embeddings] and opts[:augment] do
  Output.warning("--cache-embeddings with --augment is not recommended")
  Output.puts("    Augmentation randomizes data each run, making cached embeddings invalid")
end

# Warn about caching without precompute (nothing to cache)
if opts[:cache_embeddings] and opts[:no_precompute] do
  Output.warning("--cache-embeddings has no effect with --no-precompute")
  Output.puts("    Embedding cache requires precomputation. Remove --no-precompute to use caching.")
end

# Warn about BPTT truncation without temporal mode
if opts[:truncate_bptt] != nil and not opts[:temporal] do
  Output.warning("--truncate-bptt has no effect without --temporal")
  Output.puts("    BPTT truncation only applies to recurrent models (LSTM, GRU, Mamba)")
end

# Warn about MLP-specific flags with temporal backbone
if opts[:temporal] and (opts[:residual] or opts[:layer_norm]) do
  backbone = opts[:backbone]

  if backbone not in [:mlp, :sliding_window] do
    if opts[:residual] do
      Output.warning("--residual has no effect with --backbone #{backbone}")
      Output.puts("    Residual connections only apply to MLP backbone")
    end

    if opts[:layer_norm] do
      Output.warning("--layer-norm has no effect with --backbone #{backbone}")
      Output.puts("    Layer normalization only applies to MLP backbone")
    end
  end
end

# Dry run mode - validate config and show what would happen, then exit
if opts[:dry_run] do
  Output.section("Dry Run Mode")
  Output.success("Configuration is valid")
  Output.puts_raw("")

  Output.kv(
    "Replay files",
    "#{length(Path.wildcard(Path.join(opts[:replays], "**/*.slp")))} found"
  )

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

  wandb_opts =
    if opts[:wandb_name] do
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
replay_files =
  if initial_count > 0 and initial_count <= 5000 do
    Output.puts("  Validating replay files...")

    {:ok, validated_files, validation_stats} =
      ReplayValidation.validate(initial_replay_files,
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

# Duplicate detection (skip files with identical content)
replay_files =
  if opts[:skip_duplicates] && length(replay_files) > 1 do
    Output.puts("  Checking for duplicate files...")

    {unique_files, dup_stats} =
      DuplicateDetector.filter_duplicates(replay_files,
        show_progress: length(replay_files) > 100,
        parallel: length(replay_files) > 10
      )

    if dup_stats.duplicates > 0 do
      pct = Float.round(dup_stats.duplicates / dup_stats.total * 100, 1)
      Output.puts("  Removed #{dup_stats.duplicates} duplicate files (#{pct}%)")
    end

    unique_files
  else
    replay_files
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
# Note: train_character requires metadata to find which port has the target character
train_character = opts[:train_character]

{replay_files, replay_stats} =
  if character_filter != [] or stage_filter != [] or train_character != nil or
       initial_count <= 1000 do
    # Get canonical character names for matching
    char_names = Enum.map(character_filter, &Config.character_name/1) |> Enum.uniq()
    stage_ids = Enum.map(stage_filter, &Config.stage_id/1) |> Enum.filter(& &1)

    if character_filter != [] or stage_filter != [] do
      filter_desc = []

      filter_desc =
        if char_names != [],
          do: ["characters: #{Enum.join(char_names, ", ")}" | filter_desc],
          else: filter_desc

      filter_desc =
        if stage_ids != [] do
          stage_names =
            Enum.map(stage_filter, fn s ->
              case Config.stage_info(s) do
                {name, _} -> name
                nil -> to_string(s)
              end
            end)
            |> Enum.uniq()

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
    target_char_name = if train_character, do: Config.character_name(train_character), else: nil
    default_port = opts[:player_port]

    {filtered, char_counts, stage_counts, port_map, total_frames} =
      replay_files
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
      |> Enum.reduce({[], %{}, %{}, %{}, 0}, fn
        {:ok, {:ok, path, meta}}, {paths, chars, stages, ports, frames} ->
          # Determine training players - either dynamic (by character) or fixed
          matching_players =
            if target_char_name do
              # Find ALL ports that have the target character (for dittos like Mewtwo vs Mewtwo)
              Enum.filter(meta.players, fn p -> p.character_name == target_char_name end)
            else
              # Use fixed port (or dual-port if enabled)
              if opts[:dual_port] do
                meta.players
              else
                case Enum.find(meta.players, fn p -> p.port == default_port end) do
                  nil -> []
                  player -> [player]
                end
              end
            end

          # Skip replay if no matching players found
          if matching_players == [] do
            {paths, chars, stages, ports, frames}
          else
            # Collect stats and add paths for EACH matching player
            # Add frame count once per replay (not per player in dittos)
            replay_frames = meta.duration_frames || 0
            frames = frames + replay_frames

            {p, c, s, pt} =
              Enum.reduce(matching_players, {paths, chars, stages, ports}, fn player,
                                                                              {p, c, s, pt} ->
                # Collect character stats
                c = Map.update(c, player.character_name, 1, &(&1 + 1))

                # Collect stage stats (only once per replay)
                stage_name = Map.get(stage_id_to_name, meta.stage, "Stage #{meta.stage}")

                s =
                  if player == hd(matching_players) do
                    Map.update(s, stage_name, 1, &(&1 + 1))
                  else
                    s
                  end

                # Check filters
                char_match = char_names == [] or player.character_name in char_names
                stage_match = stage_ids == [] or meta.stage in stage_ids

                if char_match and stage_match do
                  # Use {path, port} tuple to uniquely identify each training example
                  path_key = {path, player.port}
                  pt = Map.put(pt, path_key, player.port)
                  {[path_key | p], c, s, pt}
                else
                  {p, c, s, pt}
                end
              end)

            {p, c, s, pt, frames}
          end

        _, acc ->
          acc
      end)

    filtered = Enum.reverse(filtered)

    if character_filter != [] or stage_filter != [] do
      Output.puts("  #{length(filtered)}/#{initial_count} replays match filters")
    end

    stats = %{
      total: initial_count,
      characters: char_counts,
      stages: stage_counts,
      port_map: port_map,
      total_frames: total_frames
    }

    {filtered, stats}
  else
    # Skip metadata collection for large datasets without filters
    {replay_files, nil}
  end

# Apply max_files limit after filtering
replay_files =
  if opts[:max_files], do: Enum.take(replay_files, opts[:max_files]), else: replay_files

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

# Streaming mode: prepare file chunks but don't load data yet
# Data will be loaded chunk-by-chunk during training
streaming_mode = opts[:stream_chunk_size] != nil

# Warn if prefetch is enabled without streaming mode (it's a no-op)
if opts[:prefetch] and not streaming_mode do
  Output.warning("--prefetch has no effect without --stream-chunk-size")
  Output.puts("    Prefetching requires streaming mode due to EXLA tensor process limitations")
  Output.puts("    Either add --stream-chunk-size N or remove --prefetch to silence this warning")
end

file_chunks =
  if streaming_mode do
    chunk_size = opts[:stream_chunk_size]
    chunks = Streaming.chunk_files(replay_files, chunk_size)
    Output.puts("  Streaming: #{length(chunks)} chunks of up to #{chunk_size} files")

    # Warn if character balancing is requested with streaming mode
    if opts[:balance_characters] do
      Output.warning("--balance-characters is not fully supported in streaming mode")
      Output.puts("    Character weights will not be computed (data not available upfront)")
      Output.puts("    Consider using standard mode for multi-character training")
    end

    chunks
  else
    nil
  end

# Standard mode: parse all files upfront
{parse_time, {all_frames, errors}} =
  if not streaming_mode do
    :timer.tc(fn ->
      replay_files
      |> Task.async_stream(
        fn path_or_tuple ->
          # Handle both {path, port} tuples (from character filter) and plain paths
          {path, player_port} =
            case path_or_tuple do
              {p, port} ->
                {p, port}

              p when is_binary(p) ->
                if dual_port do
                  # dual_port will parse both
                  {p, nil}
                else
                  {p, Map.get(port_map, p, default_port)}
                end
            end

          if dual_port do
            # Dual-port: parse both ports and combine frames
            parse_dual_port.(path, opts[:frame_delay], opts[:min_quality])
          else
            # Single port: use the determined port
            case Peppi.parse(path, player_port: player_port) do
              {:ok, replay} ->
                # Quality filtering if enabled
                if opts[:min_quality] do
                  quality_data = ReplayQuality.from_parsed_replay(replay)
                  score = ReplayQuality.score(quality_data)

                  if score == :rejected or score < opts[:min_quality] do
                    {:quality_rejected, path, score}
                  else
                    frames =
                      Peppi.to_training_frames(replay,
                        player_port: player_port,
                        frame_delay: opts[:frame_delay]
                      )

                    {:ok, path, length(frames), frames, score}
                  end
                else
                  frames =
                    Peppi.to_training_frames(replay,
                      player_port: player_port,
                      frame_delay: opts[:frame_delay]
                    )

                  {:ok, path, length(frames), frames, nil}
                end

              {:error, reason} ->
                {:error, path, reason}
            end
          end
        end,
        max_concurrency: System.schedulers_online(),
        timeout: :infinity
      )
      |> Enum.reduce({0, 0, [], [], 0, []}, fn
        {:ok, {:ok, _path, frame_count, frames, quality_score}},
        {total_files, total_frames, all_frames, errors, rejected, scores} ->
          scores = if quality_score, do: [quality_score | scores], else: scores

          {total_files + 1, total_frames + frame_count, [frames | all_frames], errors, rejected,
           scores}

        {:ok, {:quality_rejected, _path, _score}},
        {total_files, total_frames, all_frames, errors, rejected, scores} ->
          {total_files, total_frames, all_frames, errors, rejected + 1, scores}

        {:ok, {:error, path, reason}},
        {total_files, total_frames, all_frames, errors, rejected, scores} ->
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

          {total_files, total_frames, all_frames, [error | errors], rejected, scores}

        {:exit, reason}, {total_files, total_frames, all_frames, errors, rejected, scores} ->
          # Task crashed - treat as error
          error = %{path: "unknown", reason: {:exit, reason}}
          {total_files, total_frames, all_frames, [error | errors], rejected, scores}
      end)
      |> then(fn {files, frames, frame_lists, errors, quality_rejected, quality_scores} ->
        Output.puts("  Parsed #{files} files, #{frames} total frames")

        if quality_rejected > 0 do
          Output.puts("  Filtered #{quality_rejected} replays below quality threshold")
        end

        if length(errors) > 0 do
          Output.puts("  ⚠ #{length(errors)} files failed to parse")
        end

        # Show quality stats if enabled
        if opts[:show_quality_stats] and length(quality_scores) > 0 do
          avg = Enum.sum(quality_scores) / length(quality_scores)
          min_q = Enum.min(quality_scores)
          max_q = Enum.max(quality_scores)
          Output.puts("  Quality: avg=#{Float.round(avg, 1)}, min=#{min_q}, max=#{max_q}")
        end

        {List.flatten(frame_lists), Enum.reverse(errors)}
      end)
    end)
  else
    # Streaming mode: skip upfront parsing, data loaded per-chunk during training
    {0, {[], []}}
  end

if not streaming_mode do
  Output.puts("  Parse time: #{Float.round(parse_time / 1_000_000, 2)}s")
  Output.puts("  Total training frames: #{length(all_frames)}")

  # Show error summary if there were failures
  if length(errors) > 0 do
    Output.puts("\n  Error Summary (#{length(errors)} failed files):")

    if show_errors do
      errors
      # Show first 10
      |> Enum.take(10)
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
end

# Build embedding config with stage_mode and num_player_names options (needed for dataset creation)
embed_config =
  Embeddings.config(
    stage_mode: opts[:stage_mode],
    num_player_names: opts[:num_player_names]
  )

# Build player registry for style-conditional training (if enabled)
alias ExPhil.Training.PlayerRegistry

player_registry =
  if opts[:learn_player_styles] do
    Output.puts("\n  Building player registry for style-conditional training...", :cyan)

    # Helper to build registry from replays
    build_from_replays = fn files ->
      {:ok, reg} =
        PlayerRegistry.from_replays(files,
          max_players: opts[:num_player_names],
          min_games: opts[:min_player_games],
          unknown_strategy: :hash
        )

      Output.puts("    ✓ Found #{PlayerRegistry.size(reg)} unique players")

      # Show top players
      if PlayerRegistry.size(reg) > 0 do
        top_tags = reg |> PlayerRegistry.list_tags() |> Enum.take(5)
        Output.puts("    Top players: #{Enum.join(top_tags, ", ")}")
      end

      reg
    end

    registry =
      cond do
        # Load existing registry if provided
        opts[:player_registry] && File.exists?(opts[:player_registry]) ->
          Output.puts("    Loading registry from #{opts[:player_registry]}...")

          case PlayerRegistry.from_json(opts[:player_registry]) do
            {:ok, reg} ->
              Output.puts("    ✓ Loaded #{PlayerRegistry.size(reg)} players")
              reg

            {:error, reason} ->
              Output.warning("Failed to load registry: #{inspect(reason)}")
              Output.puts("    Building new registry from replays...")
              build_from_replays.(replay_files)
          end

        # Build from replays
        true ->
          build_from_replays.(replay_files)
      end

    # Save registry if path provided
    if opts[:player_registry] do
      case PlayerRegistry.to_json(registry, opts[:player_registry]) do
        :ok -> Output.puts("    ✓ Saved registry to #{opts[:player_registry]}")
        {:error, reason} -> Output.warning("Failed to save registry: #{inspect(reason)}")
      end
    end

    registry
  else
    nil
  end

# Step 2: Create dataset (skipped in streaming mode - data loaded per-chunk)
{train_dataset, val_dataset} =
  if not streaming_mode do
    Output.puts("\nStep 2: Creating dataset...", :cyan)

    dataset =
      Data.from_frames(all_frames,
        embed_config: embed_config,
        player_registry: player_registry
      )

    # Convert to sequences for temporal training, or precompute embeddings for MLP
    base_dataset =
      if opts[:temporal] do
        Output.puts(
          "  Converting to sequences (window=#{opts[:window_size]}, stride=#{opts[:stride]})..."
        )

        seq_dataset =
          Data.to_sequences(dataset,
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
    {train_ds, val_ds} =
      if opts[:val_split] > 0.0 do
        train_ratio = 1.0 - opts[:val_split]
        Data.split(base_dataset, ratio: train_ratio)
      else
        # No validation split - use all data for training, create empty val dataset
        {base_dataset, Data.empty(base_dataset)}
      end

    data_type = if opts[:temporal], do: "sequences", else: "frames"
    Output.puts("  Training #{data_type}: #{train_ds.size}")

    if val_ds.size > 0 do
      Output.puts(
        "  Validation #{data_type}: #{val_ds.size} (#{Float.round(opts[:val_split] * 100, 1)}%)"
      )
    else
      Output.puts("  Validation: disabled (--val-split 0.0)")
    end

    # Show some statistics
    stats = Data.stats(train_ds)
    Output.puts("\n  Button press rates:")

    for {button, rate} <- Enum.sort(stats.button_rates) do
      bar = String.duplicate("█", round(rate * 50))

      Output.puts(
        "    #{button |> to_string() |> String.pad_trailing(6)}: #{bar} #{Float.round(rate * 100, 1)}%"
      )
    end

    {train_ds, val_ds}
  else
    # Streaming mode: datasets are created per-chunk during training
    Output.puts("\nStep 2: Dataset creation deferred (streaming mode)", :cyan)
    Output.puts("  Data will be loaded in #{length(file_chunks)} chunks during training")
    {nil, nil}
  end

# Character balancing: compute weights if enabled
character_weights =
  if opts[:balance_characters] and train_dataset != nil do
    Output.puts("\n  Character balancing enabled:")

    # Count characters from training dataset frames
    char_counts = CharacterBalance.count_characters(train_dataset.frames)

    if map_size(char_counts) > 1 do
      # Compute inverse frequency weights
      weights = CharacterBalance.compute_weights(char_counts)

      # Display distribution with weights
      for line <- CharacterBalance.format_distribution(char_counts, weights) do
        Output.puts(line)
      end

      Output.puts("  Rare characters will be sampled more frequently")
      weights
    else
      Output.puts("  Only one character found - balancing not needed")
      nil
    end
  else
    nil
  end

# Step 3: Initialize trainer
Output.puts("\nStep 3: Initializing model...", :cyan)

# Use embed_config created earlier (before dataset creation)
embed_size = Embeddings.embedding_size(embed_config)
Output.puts("  Embedding size: #{embed_size}")

if opts[:stage_mode] != :one_hot_full do
  Output.puts("  Stage mode: #{opts[:stage_mode]}")
end

# For Mamba, use first hidden_size value as hidden_size (single int)
# For other backbones, use hidden_sizes list
hidden_size =
  case opts[:hidden_sizes] do
    [h | _] -> h
    _ -> 256
  end

trainer_opts = [
  embed_config: embed_config,
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
  # Smaller for CPU training
  num_heads: 2,
  # Smaller for CPU training
  head_dim: 32,
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
  checkpoint_every: opts[:checkpoint_every],
  # K-means stick discretization
  kmeans_centers: opts[:kmeans_centers]
]

# Create trainer (or load from checkpoint for resumption)
{trainer, resumed_step} =
  if opts[:resume] do
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
estimated_mem_mb =
  GPUUtils.estimate_memory_mb(
    param_count: param_count,
    batch_size: opts[:batch_size],
    precision: opts[:precision],
    temporal: opts[:temporal],
    window_size: opts[:window_size]
  )

case GPUUtils.check_free_memory(required_mb: estimated_mem_mb) do
  {:warning, msg} -> Output.warning(msg)
  :ok -> :ok
  # No GPU, skip check
  {:error, _} -> :ok
end

if opts[:accumulation_steps] > 1 do
  effective_batch = opts[:batch_size] * opts[:accumulation_steps]

  Output.puts(
    "  Gradient accumulation: #{opts[:accumulation_steps]}x (effective batch: #{effective_batch})"
  )
end

if resumed_step > 0 do
  Output.puts("  Resumed at step: #{resumed_step}")
end

# Time estimation
# Based on empirical measurements: ~0.1s per batch after JIT, 3-5min for JIT compilation
if not streaming_mode and train_dataset != nil do
  batches_per_epoch = div(train_dataset.size, opts[:batch_size])
  # ~5 minutes for first JIT compilation
  jit_overhead_sec = 300
  # Temporal is slower
  seconds_per_batch = if opts[:temporal], do: 0.5, else: 0.1
  estimated_train_sec = batches_per_epoch * opts[:epochs] * seconds_per_batch + jit_overhead_sec
  estimated_minutes = div(trunc(estimated_train_sec), 60)
  estimated_remaining = rem(trunc(estimated_train_sec), 60)

  Output.puts("")
  Output.puts("  ⏱  Estimated training time: ~#{estimated_minutes}m #{estimated_remaining}s")

  Output.puts(
    "      (#{batches_per_epoch} batches/epoch × #{opts[:epochs]} epochs + JIT compilation)"
  )
else
  Output.puts("")
  Output.puts("  ⏱  Streaming mode: time estimate not available")
end

# Step 4: Training loop
# Create augmentation function if enabled
augment_fn =
  if opts[:augment] do
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

early_stopping_msg =
  if opts[:early_stopping] do
    " (early stopping: patience=#{opts[:patience]}, min_delta=#{opts[:min_delta]})"
  else
    ""
  end

Output.puts("\nStep 4: Training for #{opts[:epochs]} epochs#{early_stopping_msg}...", :cyan)
Output.divider()

# Create incomplete marker for crash recovery
Recovery.mark_started(opts[:checkpoint], opts)

# Set up graceful shutdown handler for SIGTERM/SIGINT (Ctrl+C)
# Uses an Agent to track trainer state since signal handlers can't access reduce state
interrupt_checkpoint_path = String.replace(opts[:checkpoint], ".axon", "_interrupt.axon")

{:ok, _trainer_state_agent} = Agent.start_link(fn -> nil end, name: :trainer_state)

graceful_shutdown = fn signal ->
  IO.write(:stderr, "\n\n")
  Output.warning("Received #{signal} - saving checkpoint before exit...")

  case Agent.get(:trainer_state, fn state -> state end, 5000) do
    {trainer, epoch, batch_idx} when trainer != nil ->
      Output.puts("  Saving trainer state (epoch #{epoch}, batch #{batch_idx})...")

      case Imitation.save_checkpoint(trainer, interrupt_checkpoint_path) do
        :ok ->
          Output.success("Interrupt checkpoint saved to #{interrupt_checkpoint_path}")
          Output.puts("  Resume with: --resume #{interrupt_checkpoint_path}")

        {:error, reason} ->
          Output.error("Failed to save checkpoint: #{inspect(reason)}")
      end

    _ ->
      Output.warning("No trainer state available to save")
  end

  # Clean up the agent
  Agent.stop(:trainer_state)

  # Exit with appropriate code
  System.halt(if signal == :sigint, do: 130, else: 143)
end

# Trap SIGTERM for graceful shutdown
# Note: :sigint is not supported by System.trap_signal/3 (Ctrl+C handled by BEAM)
for signal <- [:sigterm] do
  try do
    case System.trap_signal(signal, fn -> graceful_shutdown.(signal) end) do
      {:ok, _} ->
        :ok

      {:error, :not_sup} ->
        Output.warning("Signal #{signal} trapping not supported on this platform")
    end
  rescue
    _ -> Output.warning("Signal #{signal} trapping failed")
  end
end

start_time = System.monotonic_time(:second)

# Initialize early stopping state if enabled
early_stopping_state =
  if opts[:early_stopping] do
    EarlyStopping.init(patience: opts[:patience], min_delta: opts[:min_delta])
  else
    nil
  end

# Initialize checkpoint pruner if configured
pruner =
  if opts[:keep_best] && opts[:save_every] do
    CheckpointPruning.new(keep_best: opts[:keep_best], metric: :loss)
  else
    nil
  end

# Initialize EMA if configured
ema =
  if opts[:ema] do
    Output.puts("  EMA enabled (decay=#{opts[:ema_decay]})")
    EMA.new(trainer.policy_params, decay: opts[:ema_decay])
  else
    nil
  end

# Pre-compute streaming config and batch estimate (once, not per epoch)
{streaming_chunk_opts, streaming_dataset_opts, estimated_streaming_batches} =
  if streaming_mode do
    chunk_opts = [
      player_port: default_port,
      port_map: port_map,
      dual_port: dual_port,
      frame_delay: opts[:frame_delay]
    ]

    dataset_opts = [
      temporal: opts[:temporal],
      window_size: opts[:window_size],
      stride: opts[:stride],
      precompute: opts[:precompute],
      embed_config: embed_config,
      player_registry: player_registry
    ]

    # Estimate total batches from metadata (no parsing needed)
    # IMPORTANT: In streaming mode, this estimates batches for ONE EPOCH (all chunks)
    estimated_batches =
      if replay_stats && replay_stats[:total_frames] do
        total_frames = replay_stats[:total_frames]
        window_size = opts[:window_size]
        stride = opts[:stride]
        batch_size = opts[:batch_size]

        # Each replay produces (frames - window_size) / stride sequences
        # For dittos/dual-port, we train on both players (2 training examples per replay)
        num_training_examples = length(replay_files)

        # total_frames counts each replay ONCE, but num_training_examples may be doubled for dual-port
        # So we need to use num_replays for avg_frames calculation
        num_replays =
          if dual_port, do: max(div(num_training_examples, 2), 1), else: num_training_examples

        avg_frames_per_replay = if num_replays > 0, do: div(total_frames, num_replays), else: 0

        # Sequences per training example (each example processes the full replay)
        sequences_per_example = max(div(avg_frames_per_replay - window_size, stride), 0)
        total_sequences = sequences_per_example * num_training_examples
        total_batches = max(div(total_sequences, batch_size), 1)

        Output.puts("  Estimated #{total_batches} batches per epoch")

        Output.puts(
          "    (#{total_frames} frames, #{num_replays} replays, #{num_training_examples} examples, window=#{window_size}, stride=#{stride})"
        )

        total_batches
      else
        # Better fallback: estimate based on typical Melee replay stats
        # Average replay: ~10,000 frames, with window=90, stride=1: ~9,910 sequences
        # With batch_size=32: ~310 batches per replay
        # Adjust for max_files limit
        num_files = min(length(replay_files), opts[:max_files] || length(replay_files))
        window_size = opts[:window_size] || 60
        stride = opts[:stride] || 1
        batch_size = opts[:batch_size] || 32
        # Conservative estimate for Melee replays
        avg_frames_per_replay = 10_000

        sequences_per_replay = max(div(avg_frames_per_replay - window_size, stride), 1)
        total_batches = max(div(sequences_per_replay * num_files, batch_size), 1)

        Output.puts("  Estimated ~#{total_batches} batches per epoch (from #{num_files} files)")
        total_batches
      end

    {chunk_opts, dataset_opts, estimated_batches}
  else
    {nil, nil, nil}
  end

# Training loop with early stopping and best model tracking
# Returns {trainer, epochs_completed, stopped_early, early_stopping_state, best_val_loss, pruner, ema, history, global_batch_idx}
initial_state = {trainer, 0, false, early_stopping_state, nil, pruner, ema, [], 0}

# Batch checkpoint interval (nil = disabled)
save_every_batches = opts[:save_every_batches]

batch_checkpoint_path =
  if save_every_batches do
    String.replace(opts[:checkpoint], ".axon", "_batch.axon")
  end

{final_trainer, epochs_completed, stopped_early, _es_state, _best_val, _final_pruner, final_ema,
 training_history,
 _final_batch_idx} =
  Enum.reduce_while(1..opts[:epochs], initial_state, fn epoch,
                                                        {current_trainer, _, _, es_state,
                                                         best_val_loss, current_pruner,
                                                         current_ema, history,
                                                         global_batch_idx} ->
    epoch_start = System.monotonic_time(:second)

    # Create batched dataset for this epoch
    # Use appropriate batching function based on temporal mode
    # Note: augmentation is only applied to non-temporal (single-frame) batches
    # Temporal batches use pre-computed embeddings, so augmentation happens at sequence creation
    # Create batch stream (kept lazy for true async prefetching)
    {batch_stream, num_batches} =
      if streaming_mode do
        # Streaming mode: process each chunk and chain batches together
        # Each chunk is parsed, embedded, and batched on-demand
        # (chunk_opts, dataset_opts, and batch estimate computed once before epoch loop)
        num_chunks = length(file_chunks)

        stream =
          file_chunks
          |> Enum.with_index(1)
          |> Stream.flat_map(fn {chunk, chunk_idx} ->
            # Show chunk progress (helps user understand streaming progress)
            IO.write(:stderr, "\n")

            Output.puts(
              "  📦 Processing chunk #{chunk_idx}/#{num_chunks} (#{length(chunk)} files)..."
            )

            # Parse and create dataset for this chunk
            {:ok, chunk_frames, errors} = Streaming.parse_chunk(chunk, streaming_chunk_opts)

            # Debug: show chunk stats
            if length(chunk_frames) == 0 do
              Output.warning("Chunk produced 0 frames from #{length(chunk)} files")

              if errors != [] do
                Output.puts("    Errors: #{inspect(Enum.take(errors, 3))}")
              end
            end

            chunk_dataset = Streaming.create_dataset(chunk_frames, streaming_dataset_opts)

            # Debug: show dataset size after sequence conversion
            if chunk_dataset.size == 0 and length(chunk_frames) > 0 do
              Output.warning(
                "Dataset has 0 sequences (frames: #{length(chunk_frames)}, window: #{opts[:window_size]})"
              )
            end

            # Create batches from this chunk
            # Note: character_weights is nil in streaming mode (computed per-chunk would be less effective)
            if opts[:temporal] do
              Data.batched_sequences(chunk_dataset,
                batch_size: opts[:batch_size],
                shuffle: true,
                # Don't drop - small chunks may lose data
                drop_last: false,
                character_weights: character_weights
              )
            else
              Data.batched(chunk_dataset,
                batch_size: opts[:batch_size],
                shuffle: true,
                drop_last: false,
                augment_fn: augment_fn,
                frame_delay: opts[:frame_delay],
                frame_delay_augment: opts[:frame_delay_augment],
                frame_delay_min: opts[:frame_delay_min],
                frame_delay_max: opts[:frame_delay_max],
                character_weights: character_weights
              )
            end
          end)

        {stream, estimated_streaming_batches}
      else
        # Standard mode: use pre-loaded dataset
        stream =
          if opts[:temporal] do
            Data.batched_sequences(train_dataset,
              batch_size: opts[:batch_size],
              shuffle: true,
              drop_last: true,
              # Character-balanced sampling (if enabled)
              character_weights: character_weights
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
              frame_delay_max: opts[:frame_delay_max],
              # Character-balanced sampling (if enabled)
              character_weights: character_weights
            )
          end

        # Calculate number of batches for progress display
        # (we count without materializing to keep the stream lazy)
        batches = div(train_dataset.size, opts[:batch_size])
        {stream, batches}
      end

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
    jit_indicator_shown =
      if epoch == 1 do
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
    # State includes global_batch_idx for cross-epoch batch counting
    process_batch = fn batch, batch_idx, {t, losses, jit_shown, curr_global_idx} ->
      batch_start = System.monotonic_time(:millisecond)
      # Note: loss_fn is ignored by train_step (it uses cached predict_fn internally)
      {new_trainer, metrics} = Imitation.train_step(t, batch, nil)
      batch_time_ms = System.monotonic_time(:millisecond) - batch_start

      # Increment global batch index
      new_global_idx = curr_global_idx + 1

      # Show JIT completion message after first batch
      new_jit_shown =
        if jit_shown and batch_idx == 0 do
          Output.puts(
            "\n  ✓ JIT compilation complete (took #{Float.round(batch_time_ms / 1000, 1)}s)"
          )

          true
        else
          jit_shown
        end

      # Batch-interval checkpointing (for streaming mode resilience)
      if save_every_batches && rem(new_global_idx, save_every_batches) == 0 do
        IO.write(:stderr, "\n")
        Output.puts("  💾 Saving batch checkpoint (batch #{new_global_idx})...")

        case Imitation.save_checkpoint(new_trainer, batch_checkpoint_path) do
          :ok ->
            Output.puts("  ✓ Saved to #{batch_checkpoint_path}")

          {:error, reason} ->
            Output.warning("Failed to save batch checkpoint: #{inspect(reason)}")
        end
      end

      # Update trainer state agent for graceful shutdown (Ctrl+C)
      # Only update every 10 batches to minimize overhead
      if rem(new_global_idx, 10) == 0 do
        Agent.update(:trainer_state, fn _ -> {new_trainer, epoch, batch_idx} end)
      end

      # Live progress bar - updates in place using carriage return
      # Update every batch for smooth progress (terminal handles the refresh)
      elapsed_total_ms = System.monotonic_time(:millisecond) - epoch_batch_start
      avg_batch_ms = elapsed_total_ms / (batch_idx + 1)

      # Handle case where actual batches exceed estimate (estimate was wrong)
      # Use the larger of estimated or actual count for display
      display_total = max(num_batches, batch_idx + 1)
      pct = min(round((batch_idx + 1) / display_total * 100), 100)

      remaining_batches = max(display_total - (batch_idx + 1), 0)
      eta_sec = round(remaining_batches * avg_batch_ms / 1000)
      eta_min = div(eta_sec, 60)
      eta_sec_rem = rem(eta_sec, 60)

      # Format: Epoch 1: ████████░░ 40% | 642/1606 | loss: 0.1234 | 0.5s/it | ETA: 8m 12s
      bar_width = 20
      # Clamp to avoid overflow
      filled = min(round(pct / 100 * bar_width), bar_width)
      bar = String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled)

      # Pad percentage to fixed width for stable display
      pct_str = pct |> Integer.to_string() |> String.pad_leading(3)

      # Show "~" prefix on total if we exceeded the estimate (indicating it's now a live count)
      total_str =
        if batch_idx + 1 > num_batches, do: "~#{display_total}", else: "#{display_total}"

      # IMPORTANT: Only convert loss to number periodically to avoid GPU→CPU sync every batch
      # (See gotcha #18: Nx.to_number blocks GPU utilization)
      # Convert every 50 batches for display, but accumulate tensor for epoch-end averaging
      display_loss =
        if rem(batch_idx, 50) == 0 or batch_idx == 0 do
          Nx.to_number(metrics.loss)
        else
          # Use previous display loss (stored in accumulator) or 0.0 for display only
          case losses do
            [{prev_display, _} | _] -> prev_display
            _ -> 0.0
          end
        end

      # Check for NaN loss - training has diverged and should stop
      if is_nan?.(display_loss) do
        IO.write(:stderr, "\n")
        Output.error("Loss became NaN at batch #{batch_idx + 1} - training diverged!")
        Output.puts("  This usually indicates:")
        Output.puts("    - Learning rate too high (try --learning-rate 1e-5)")
        Output.puts("    - Gradient explosion (try --grad-clip 1.0)")
        Output.puts("    - Numerical instability (try --precision f32 instead of bf16)")
        raise "Training diverged: loss is NaN"
      end

      # Use safe_round to handle edge cases (NaN, Inf) gracefully
      progress_line =
        "  Epoch #{epoch}: #{bar} #{pct_str}% | #{batch_idx + 1}/#{total_str} | loss: #{safe_round.(display_loss, 4)} | #{safe_round.(avg_batch_ms / 1000, 2)}s/it | ETA: #{eta_min}m #{eta_sec_rem}s"

      # Use carriage return to overwrite line (no newline until epoch complete)
      # Write directly to stderr to bypass Output module's timestamp
      IO.write(:stderr, "\r#{progress_line}")

      # Accumulate tensor loss (keep as tensor, convert at epoch end)
      # Store tuple of {display_loss, tensor} so we can show loss but compute mean from tensors
      {new_trainer, [{display_loss, metrics.loss} | losses], new_jit_shown, new_global_idx}
    end

    # Use prefetcher if enabled (computes next batch while GPU trains on current)
    {updated_trainer, epoch_losses, _, updated_global_batch_idx} =
      cond do
        opts[:prefetch] and streaming_mode ->
          # Streaming mode: use stream-based prefetcher for lazy iteration
          # This avoids materializing all chunks at once
          Prefetcher.reduce_stream_indexed(
            batch_stream,
            {current_trainer, [], jit_indicator_shown, global_batch_idx},
            process_batch,
            buffer_size: opts[:prefetch_buffer]
          )

        opts[:prefetch] ->
          # Non-streaming mode: use list-based prefetcher
          # reduce_indexed materializes the stream first (safe for EXLA tensors)
          # then does simple Task-based prefetching
          Prefetcher.reduce_indexed(
            batch_stream,
            {current_trainer, [], jit_indicator_shown, global_batch_idx},
            process_batch
          )

        true ->
          # No prefetching - standard sequential processing
          batch_stream
          |> Stream.with_index()
          |> Enum.reduce({current_trainer, [], jit_indicator_shown, global_batch_idx}, fn {batch,
                                                                                           batch_idx},
                                                                                          acc ->
            process_batch.(batch, batch_idx, acc)
          end)
      end

    epoch_time = System.monotonic_time(:second) - epoch_start
    # epoch_losses is a list of {display_loss, tensor} tuples
    # Extract tensors and compute mean (single GPU→CPU transfer at epoch end)
    avg_loss =
      if epoch_losses == [] do
        Output.warning("No batches processed this epoch - check replay data")
        0.0
      else
        # Extract tensor losses, stack, and compute mean
        tensor_losses = Enum.map(epoch_losses, fn {_display, tensor} -> tensor end)

        tensor_losses
        |> Nx.stack()
        |> Nx.mean()
        |> Nx.to_number()
      end

    # Validation - only if we have validation data (not in streaming mode)
    {val_loss, _val_metrics} =
      cond do
        streaming_mode ->
          # In streaming mode, use training loss as proxy
          # (validation would require holding extra data in memory)
          {avg_loss, %{loss: avg_loss}}

        val_dataset != nil and val_dataset.size > 0 ->
          val_batches =
            if opts[:temporal] do
              Data.batched_sequences(val_dataset, batch_size: opts[:batch_size], shuffle: false)
            else
              Data.batched(val_dataset, batch_size: opts[:batch_size], shuffle: false)
            end

          metrics = Imitation.evaluate(updated_trainer, val_batches)
          {metrics.loss, metrics}

        true ->
          # No validation set - use training loss as proxy
          {avg_loss, %{loss: avg_loss}}
      end

    # Check early stopping (uses val_loss or train_loss if no validation)
    {new_es_state, es_decision, es_message} =
      if es_state do
        {new_state, decision} = EarlyStopping.check(es_state, val_loss)
        {new_state, decision, " | #{EarlyStopping.status_message(new_state)}"}
      else
        {nil, :continue, ""}
      end

    # Track epoch metrics for loss plot
    # Determine if we have separate validation (not in streaming mode and val_dataset exists)
    has_validation = not streaming_mode and val_dataset != nil and val_dataset.size > 0

    epoch_entry = %{
      epoch: epoch,
      train_loss: avg_loss,
      val_loss: if(has_validation, do: val_loss, else: nil),
      time_seconds: epoch_time
    }

    updated_history = [epoch_entry | history]

    Output.puts("")

    if has_validation do
      Output.puts(
        "  ✓ Epoch #{epoch} complete: train_loss=#{safe_round.(avg_loss, 4)} val_loss=#{safe_round.(val_loss, 4)} (#{epoch_time}s)#{es_message}"
      )
    else
      Output.puts(
        "  ✓ Epoch #{epoch} complete: train_loss=#{safe_round.(avg_loss, 4)} (#{epoch_time}s)#{es_message}"
      )
    end

    # Update incomplete marker for crash recovery
    Recovery.mark_epoch_complete(opts[:checkpoint], epoch, val_loss)

    # Save best model if this is the best loss so far (val_loss if available, else train_loss)
    # Don't save if loss is NaN
    is_valid_loss = not is_nan?.(val_loss)
    is_new_best = is_valid_loss and (best_val_loss == nil or val_loss < best_val_loss)
    new_best_val_loss = if is_new_best, do: val_loss, else: best_val_loss

    if opts[:save_best] and is_new_best do
      best_checkpoint_path = Config.derive_best_checkpoint_path(opts[:checkpoint])
      best_policy_path = Config.derive_best_policy_path(opts[:checkpoint])

      loss_type = if has_validation, do: "val_loss", else: "train_loss"

      case Imitation.save_checkpoint(updated_trainer, best_checkpoint_path) do
        :ok ->
          case Imitation.export_policy(updated_trainer, best_policy_path) do
            :ok ->
              Output.puts("    ★ New best model saved (#{loss_type}=#{safe_round.(val_loss, 4)})")

            {:error, _} ->
              Output.puts("    ★ Best checkpoint saved, policy export failed")
          end

        {:error, reason} ->
          Output.puts("    ⚠ Failed to save best model: #{inspect(reason)}")
      end
    end

    # Save periodic checkpoint if configured and track for pruning
    updated_pruner =
      if opts[:save_every] && rem(epoch, opts[:save_every]) == 0 do
        epoch_checkpoint = String.replace(opts[:checkpoint], ".axon", "_epoch#{epoch}.axon")

        case Imitation.save_checkpoint(updated_trainer, epoch_checkpoint) do
          :ok ->
            Output.puts("    📁 Epoch #{epoch} checkpoint saved")

            # Track and prune if pruner is configured
            if current_pruner do
              new_pruner =
                CheckpointPruning.track(current_pruner, epoch_checkpoint, val_loss, epoch: epoch)

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
    updated_ema =
      if current_ema do
        EMA.update(current_ema, updated_trainer.policy_params)
      else
        nil
      end

    # Decide whether to continue or stop
    case es_decision do
      :stop ->
        Output.puts(
          "\n  ⚠ Early stopping triggered - no improvement for #{opts[:patience]} epochs"
        )

        {:halt,
         {updated_trainer, epoch, true, new_es_state, new_best_val_loss, updated_pruner,
          updated_ema, updated_history, updated_global_batch_idx}}

      :continue ->
        {:cont,
         {updated_trainer, epoch, false, new_es_state, new_best_val_loss, updated_pruner,
          updated_ema, updated_history, updated_global_batch_idx}}
    end
  end)

# Training complete - stop the trainer state agent and untrap signals
Agent.stop(:trainer_state)

for signal <- [:sigterm] do
  try do
    System.untrap_signal(signal)
  rescue
    _ -> :ok
  end
end

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

# Build replay manifest for provenance tracking
# Store file paths (relative to replays_dir for portability) if <= 500 files, else just hash
replay_manifest =
  if length(replay_files) <= 500 do
    # Store relative paths for portability
    base_dir = opts[:replays]

    Enum.map(replay_files, fn path_or_tuple ->
      path = if is_tuple(path_or_tuple), do: elem(path_or_tuple, 0), else: path_or_tuple
      Path.relative_to(path, base_dir)
    end)
  else
    # Too many files, just use hash
    nil
  end

# Get absolute paths for hashing
replay_paths_for_hash =
  Enum.map(replay_files, fn path_or_tuple ->
    if is_tuple(path_or_tuple), do: elem(path_or_tuple, 0), else: path_or_tuple
  end)

# Extract character distribution from replay_stats if available
character_distribution =
  if replay_stats && replay_stats[:characters] do
    replay_stats[:characters]
    |> Enum.map(fn {char, count} -> {to_string(char), count} end)
    |> Map.new()
  else
    nil
  end

final_loss_avg = Enum.sum(Enum.take(final_trainer.metrics.loss, 10)) / 10

training_results = %{
  embed_size: embed_size,
  training_frames: if(train_dataset, do: train_dataset.size, else: :streaming),
  validation_frames: if(val_dataset, do: val_dataset.size, else: nil),
  total_time_seconds: total_time,
  final_training_loss: safe_round.(final_loss_avg, 4),
  epochs_completed: epochs_completed,
  stopped_early: stopped_early,
  # Replay provenance
  replay_count: length(replay_files),
  replay_files: replay_manifest,
  replay_manifest_hash: Config.compute_manifest_hash(replay_paths_for_hash),
  character_distribution: character_distribution
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
    e ->
      Output.puts("  ⚠ Loss plot generation failed:")
      Stacktrace.print_exception(e, __STACKTRACE__)
  end
end

# Step 6: Register model in registry
unless opts[:no_register] do
  Output.puts("\nStep 6: Registering model...", :cyan)

  # Determine parent model if resuming from checkpoint
  parent_id =
    if opts[:resume] do
      case Registry.list() do
        {:ok, models} ->
          Enum.find_value(models, fn m ->
            if m.checkpoint_path == opts[:resume], do: m.id
          end)

        _ ->
          nil
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
      final_loss: safe_round.(final_loss_avg, 4),
      epochs_completed: epochs_completed,
      training_frames: if(train_dataset, do: train_dataset.size, else: :streaming),
      validation_frames: if(val_dataset, do: val_dataset.size, else: nil),
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

  Output.terminal_loss_graph(Enum.reverse(training_history),
    title: "Training Loss",
    width: 70,
    height: 14
  )
end

# Summary with colors
final_loss = safe_round.(final_loss_avg, 4)

# Find best loss from training history
best_entry =
  Enum.min_by(training_history, & &1.train_loss, fn ->
    %{train_loss: final_loss_avg, epoch: epochs_completed}
  end)

best_loss = safe_round.(best_entry.train_loss, 4)
best_epoch = best_entry.epoch

Output.puts_raw("")

Output.puts_raw(
  Output.colorize("╔════════════════════════════════════════════════════════════════╗", :green)
)

Output.puts_raw(
  Output.colorize("║                      Training Complete!                        ║", :green)
)

Output.puts_raw(
  Output.colorize("╚════════════════════════════════════════════════════════════════╝", :green)
)

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

if train_dataset do
  Output.kv("Training frames", "#{train_dataset.size}")
else
  Output.kv("Training mode", "streaming (#{length(file_chunks)} chunks)")
end

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

Output.puts_raw(
  "  2. Continue training: mix run scripts/train_from_replays.exs --resume #{opts[:checkpoint]}"
)

Output.puts_raw(
  "  3. Self-play refinement: mix run scripts/train_self_play.exs --pretrained #{String.replace(opts[:checkpoint], ".axon", "_policy.bin")}"
)

Output.puts_raw("")

# Finish Wandb run if active
if Wandb.active?() do
  Wandb.finish_run()
  Output.puts("Wandb run finished.")
end

# Remove incomplete marker - training completed successfully
Recovery.mark_complete(opts[:checkpoint])
