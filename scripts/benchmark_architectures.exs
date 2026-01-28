#!/usr/bin/env elixir
# Benchmark different architectures on the same dataset
#
# Usage:
#   mix run scripts/benchmark_architectures.exs --replays /path/to/replays [options]
#
# Options:
#   --replays, --replay-dir PATH  Path to replay directory (default: ./replays)
#   --max-files N                 Max replay files to use (default: 30)
#   --epochs N                    Training epochs per architecture (default: 3)
#   --batch-size N                Batch size (default: 128 GPU, 64 CPU)
#   --only arch1,arch2            Only run specified architectures
#   --skip arch1,arch2            Skip specified architectures
#   --continue-on-error           Continue if an architecture fails
#   --cache-embeddings            Enable embedding disk cache (saves ~1hr on re-runs)
#   --cache-dir PATH              Cache directory (default: /workspace/cache/embeddings)
#   --no-cache                    Ignore cache and recompute embeddings
#   --gpu-prealloc [FRACTION]     Pre-allocate GPU memory (default: 0.7 = 70%)
#                                 Faster but can cause fragmentation OOM
#   --gpu-on-demand               Allocate GPU memory on-demand (default)
#                                 Slower (~5-10%) but prevents fragmentation
#   --quiet, -q                   Suppress XLA/CUDA warnings and Logger noise
#
# Available architectures: mlp, gated_ssm, mamba, jamba, lstm, gru, lstm_hybrid, sliding_window
# Note: mamba_nif excluded (inference-only, can't train - gradients don't flow through NIF)
# Note: "attention" is an alias for "sliding_window" (same implementation)
#
# Examples:
#   # Quick test with just MLP and Mamba
#   mix run scripts/benchmark_architectures.exs --replays /workspace/replays --only mlp,mamba
#
#   # Skip attention (if it keeps OOMing)
#   mix run scripts/benchmark_architectures.exs --replays /workspace/replays --skip attention
#
#   # Run all but continue if one fails
#   mix run scripts/benchmark_architectures.exs --replays /workspace/replays --continue-on-error
#
#   # Use embedding cache (saves ~1hr on subsequent runs)
#   mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings
#
#   # Force recompute even if cache exists
#   mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings --no-cache
#
#   # Use GPU pre-allocation for faster training (may fragment on multi-arch runs)
#   mix run scripts/benchmark_architectures.exs --replays /workspace/replays --gpu-prealloc 0.7
#
#   # Compare allocation modes (A/B test)
#   time mix run scripts/benchmark_architectures.exs --only mlp --epochs 1 --gpu-on-demand
#   time mix run scripts/benchmark_architectures.exs --only mlp --epochs 1 --gpu-prealloc
#
# This script compares:
# - Training loss convergence
# - Validation accuracy
# - Training speed (batches/sec)
# - GPU memory usage
#
# Results are saved to checkpoints/benchmark_results.json and benchmark_report.html
#
# Memory Tips:
#   If you get OOM errors on Mamba/Jamba, try:
#     1. Use --skip mamba,jamba to skip memory-heavy architectures
#     2. Reduce --max-files to use less data
#   Note: TF_GPU_ALLOCATOR=cuda_malloc_async is set automatically by this script

# Limit inspect output to prevent overwhelming logs with tensor dumps
Application.put_env(:elixir, :inspect, limit: 10, printable_limit: 100)

# Use async CUDA allocator to reduce memory fragmentation between architectures
# This helps prevent OOM when switching from MLP to Mamba/Jamba
System.put_env("TF_GPU_ALLOCATOR", "cuda_malloc_async")

# Quiet mode: suppress Logger warnings and XLA/CUDA noise
# Parse early before any modules are loaded
if "--quiet" in System.argv() or "-q" in System.argv() do
  # Suppress Elixir Logger warnings (XLA config warnings, etc.)
  Logger.configure(level: :error)
  # Suppress XLA/JAX warnings
  System.put_env("TF_CPP_MIN_LOG_LEVEL", "2")
  # Suppress ptxas warnings
  System.put_env("PTXAS_OPTIONS", "--warning-level 0")
end

# Parse GPU memory flags early (before EXLA initializes)
# --gpu-prealloc [0.7] = pre-allocate 70% (faster, but fragmentation risk)
# --gpu-on-demand = allocate on-demand (default, slower but no fragmentation)
early_args = System.argv()

gpu_config =
  cond do
    "--gpu-prealloc" in early_args ->
      # Check if a fraction was provided
      idx = Enum.find_index(early_args, &(&1 == "--gpu-prealloc"))
      next_arg = Enum.at(early_args, idx + 1)

      fraction =
        cond do
          is_nil(next_arg) -> 0.7
          String.starts_with?(next_arg || "", "--") -> 0.7
          Regex.match?(~r/^0\.\d+$/, next_arg || "") -> String.to_float(next_arg)
          true -> 0.7
        end

      IO.puts("[GPU] Pre-allocating #{round(fraction * 100)}% of VRAM (--gpu-prealloc)")
      [platform: :cuda, memory_fraction: fraction]

    "--gpu-on-demand" in early_args ->
      IO.puts("[GPU] On-demand allocation (--gpu-on-demand)")
      [platform: :cuda, preallocate: false]

    true ->
      # Default: on-demand to prevent fragmentation
      IO.puts("[GPU] On-demand allocation (default, use --gpu-prealloc for faster)")
      [platform: :cuda, preallocate: false]
  end

Application.put_env(:exla, :clients, cuda: gpu_config)

alias ExPhil.Data.Peppi
alias ExPhil.Training.{Data, GPUUtils, Imitation, Output}
alias ExPhil.Embeddings

# For timed macro
require Output

# Safe rounding that handles :nan, :infinity, and integers
# Nx.to_number() returns atoms for special values, which Float.round/2 rejects
safe_round = fn
  value, precision when is_float(value) -> Float.round(value, precision)
  value, precision when is_integer(value) -> Float.round(value / 1, precision)
  :nan, _precision -> :nan
  :infinity, _precision -> :infinity
  :neg_infinity, _precision -> :neg_infinity
  value, _precision -> value
end

# Detect GPU model
gpu_info =
  try do
    case System.cmd(
           "nvidia-smi",
           ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
           stderr_to_stdout: true
         ) do
      {output, 0} ->
        case String.split(String.trim(output), ", ") do
          [name, memory_mb] -> %{name: name, memory_gb: String.to_integer(memory_mb) / 1024}
          _ -> %{name: "Unknown GPU", memory_gb: 0.0}
        end

      _ ->
        %{name: "CPU only", memory_gb: 0.0}
    end
  rescue
    _ -> %{name: "CPU only", memory_gb: 0.0}
  end

# Parse args
args = System.argv()

replay_dir =
  case Enum.find_index(args, &(&1 in ["--replay-dir", "--replays"])) do
    nil -> "./replays"
    idx -> Enum.at(args, idx + 1) || "./replays"
  end

max_files =
  case Enum.find_index(args, &(&1 == "--max-files")) do
    nil -> 30
    idx -> String.to_integer(Enum.at(args, idx + 1) || "30")
  end

epochs =
  case Enum.find_index(args, &(&1 == "--epochs")) do
    nil -> 3
    idx -> String.to_integer(Enum.at(args, idx + 1) || "3")
  end

batch_size =
  case Enum.find_index(args, &(&1 == "--batch-size")) do
    nil ->
      # Auto-detect: use 128 for GPU (safer for temporal models), 64 for CPU
      if System.get_env("EXLA_TARGET") == "cuda", do: 128, else: 64

    idx ->
      String.to_integer(Enum.at(args, idx + 1) || "128")
  end

# Filter architectures
only_archs =
  case Enum.find_index(args, &(&1 == "--only")) do
    nil ->
      nil

    idx ->
      (Enum.at(args, idx + 1) || "")
      |> String.split(",")
      |> Enum.map(&String.to_atom/1)
  end

skip_archs =
  case Enum.find_index(args, &(&1 == "--skip")) do
    nil ->
      []

    idx ->
      (Enum.at(args, idx + 1) || "")
      |> String.split(",")
      |> Enum.map(&String.to_atom/1)
  end

# Continue on error? (useful for benchmarking when some archs might fail)
continue_on_error = "--continue-on-error" in args

# Embedding cache options (consistent with train_from_replays.exs)
cache_enabled = "--cache-embeddings" in args
force_recompute = "--no-cache" in args

cache_dir =
  case Enum.find_index(args, &(&1 == "--cache-dir")) do
    nil -> "/workspace/cache/embeddings"
    idx -> Enum.at(args, idx + 1) || "/workspace/cache/embeddings"
  end

# Architectures to benchmark
# Note: batch_size can be overridden per-architecture for memory-heavy models
# Mamba/Jamba need smaller batches due to selective scan memory requirements
all_architectures = [
  # Hidden sizes use 256 (multiple of GPU warp size 32) for better utilization
  {:mlp, "MLP (baseline)", [temporal: false, hidden_sizes: [256, 256], precompute: true]},
  {:gated_ssm, "GatedSSM (simplified)",
   [
     temporal: true,
     backbone: :gated_ssm,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64
   ]},
  {:mamba, "Mamba (parallel scan)",
   [
     temporal: true,
     backbone: :mamba,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64
   ]},
  # NOTE: mamba_nif excluded from training benchmark - gradients don't flow through NIF
  # Use for inference-only comparison: train with :mamba, infer with :mamba_nif
  # {:mamba_nif, "Mamba NIF (CUDA)", [temporal: true, backbone: :mamba_nif, ...]},
  {:jamba, "Jamba (Mamba+Attn)",
   [
     temporal: true,
     backbone: :jamba,
     window_size: 30,
     num_layers: 3,
     attention_every: 3,
     hidden_sizes: [256, 256],
     # Reduced from 32 - Jamba OOMs on validation after epoch with batch_size: 32
     batch_size: 16,
     # Jamba needs lower LR and gradient clipping to avoid NaN (attention + Mamba can explode)
     learning_rate: 3.0e-5,
     max_grad_norm: 1.0
   ]},
  {:lstm, "LSTM",
   [temporal: true, backbone: :lstm, window_size: 30, num_layers: 1, hidden_sizes: [256, 256]]},
  {:gru, "GRU",
   [temporal: true, backbone: :gru, window_size: 30, num_layers: 1, hidden_sizes: [256, 256]]},
  {:lstm_hybrid, "LSTM+Attention",
   [
     temporal: true,
     backbone: :lstm_hybrid,
     window_size: 30,
     num_layers: 1,
     num_heads: 4,
     hidden_sizes: [256, 256],
     # Attention models benefit from gradient clipping
     max_grad_norm: 1.0
   ]},
  {:sliding_window, "Sliding Window",
   [
     temporal: true,
     backbone: :sliding_window,
     window_size: 30,
     num_layers: 1,
     num_heads: 4,
     hidden_sizes: [256, 256],
     # Attention models benefit from gradient clipping
     max_grad_norm: 1.0
   ]}
]

# Apply filters
architectures =
  all_architectures
  |> Enum.filter(fn {id, _, _} ->
    (only_archs == nil || id in only_archs) && id not in skip_archs
  end)

if length(architectures) == 0 do
  Output.error("No architectures selected! Check --only/--skip flags.")

  Output.puts(
    "Available: #{Enum.map(all_architectures, fn {id, _, _} -> id end) |> Enum.join(", ")}"
  )

  System.halt(1)
end

Output.banner("ExPhil Architecture Benchmark")

Output.config([
  {"Replay dir", replay_dir},
  {"Max files", max_files},
  {"Epochs", epochs},
  {"Batch size", batch_size},
  {"Architectures",
   "#{length(architectures)} (#{Enum.map(architectures, fn {id, _, _} -> id end) |> Enum.join(", ")})"},
  {"Continue on error", continue_on_error},
  {"Embedding cache", if(cache_enabled, do: "enabled (#{cache_dir})", else: "disabled")},
  {"GPU", "#{gpu_info.name} (#{safe_round.(gpu_info.memory_gb, 1)} GB)"},
  {"Memory", GPUUtils.memory_status_string()}
])

# Step 1: Load replays
Output.step(1, 3, "Loading replays")
replay_files = Path.wildcard("#{replay_dir}/**/*.slp") |> Enum.take(max_files)
Output.puts("Found #{length(replay_files)} replay files")

if length(replay_files) == 0 do
  Output.error("No replay files found in #{replay_dir}")
  System.halt(1)
end

# Split by REPLAY FILES (not frames) to prevent data leakage for temporal models
# If we split frames, sequences near the boundary share frames between train/val
# Splitting by replay ensures train and val come from completely different games
{train_files, val_files} = Enum.split(replay_files, trunc(length(replay_files) * 0.9))
Output.puts("Split: #{length(train_files)} train replays, #{length(val_files)} val replays")

# Step 2: Parse replays (separately for train and val)
Output.step(2, 3, "Parsing replays")

Output.puts("Parsing train replays...")
train_frames =
  train_files
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {path, idx} ->
    Output.progress_bar(idx, length(train_files), label: "Train")

    case Peppi.parse(path) do
      {:ok, replay} -> Peppi.to_training_frames(replay)
      {:error, _} -> []
    end
  end)

Output.progress_done()

Output.puts("Parsing val replays...")
val_frames =
  val_files
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {path, idx} ->
    Output.progress_bar(idx, length(val_files), label: "Val")

    case Peppi.parse(path) do
      {:ok, replay} -> Peppi.to_training_frames(replay)
      {:error, _} -> []
    end
  end)

Output.progress_done()

Output.puts("Total frames: #{length(train_frames) + length(val_frames)}")

train_dataset = Data.from_frames(train_frames)
val_dataset = Data.from_frames(val_frames)

Output.puts("Train: #{train_dataset.size} frames, Val: #{val_dataset.size} frames")

# Precompute embeddings ONCE (reused for all architectures)
# This avoids re-embedding for every architecture which takes 9+ minutes each
embed_config = Embeddings.config()

# Check if any temporal architectures are selected
has_temporal = Enum.any?(architectures, fn {_id, _name, opts} -> opts[:temporal] end)
has_non_temporal = Enum.any?(architectures, fn {_id, _name, opts} -> !opts[:temporal] end)

# Always precompute frame embeddings first (needed for both MLP and temporal)
# Temporal architectures can build sequence embeddings from these (30x faster)
{precomputed_train_frames, precomputed_val_frames} =
  if has_non_temporal or has_temporal do
    Output.puts("Precomputing frame embeddings (reused for ALL architectures)...")

    # Use separate cache keys for train and val (different replay files)
    train_cache_opts = [
      cache: cache_enabled,
      cache_dir: cache_dir,
      force_recompute: force_recompute,
      replay_files: train_files,
      show_progress: true
    ]

    val_cache_opts = [
      cache: cache_enabled,
      cache_dir: cache_dir,
      force_recompute: force_recompute,
      replay_files: val_files,
      show_progress: false
    ]

    train_emb =
      Output.timed "Embedding train frames" do
        Data.precompute_frame_embeddings_cached(train_dataset, train_cache_opts)
      end

    val_emb =
      Output.timed "Embedding val frames" do
        Data.precompute_frame_embeddings_cached(val_dataset, val_cache_opts)
      end

    {train_emb, val_emb}
  else
    {nil, nil}
  end

# Build sequence embeddings from frame embeddings (30x faster than re-embedding!)
# This reuses the frame embeddings computed above via tensor slicing
{precomputed_train_seqs, precomputed_val_seqs} =
  if has_temporal do
    Output.puts("Building sequence embeddings from frame embeddings (30x faster)...")

    window_size = 30
    stride = 1

    train_seq =
      Output.timed "Building train sequences" do
        seq_ds = Data.to_sequences(train_dataset, window_size: window_size, stride: stride)
        Data.sequences_from_frame_embeddings(
          seq_ds,
          precomputed_train_frames.embedded_frames,
          window_size: window_size,
          show_progress: true
        )
      end

    val_seq =
      Output.timed "Building val sequences" do
        seq_ds = Data.to_sequences(val_dataset, window_size: window_size, stride: stride)
        Data.sequences_from_frame_embeddings(
          seq_ds,
          precomputed_val_frames.embedded_frames,
          window_size: window_size,
          show_progress: false
        )
      end

    {train_seq, val_seq}
  else
    {nil, nil}
  end

# Step 3: Run benchmarks
Output.step(3, 3, "Running benchmarks")
Output.divider()

num_archs = length(architectures)

results =
  architectures
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {{arch_id, arch_name, arch_opts}, arch_idx} ->
    # Force garbage collection between architectures to free GPU memory
    # With preallocate: false, this actually releases memory back to CUDA
    if arch_idx > 1 do
      Output.puts("Clearing memory before next architecture...")
      # GC all processes to release any tensor references
      for pid <- Process.list(), do: :erlang.garbage_collect(pid)
      # Give CUDA async allocator time to actually free memory
      Process.sleep(2000)
    end

    Output.section("[#{arch_idx}/#{num_archs}] #{arch_name}")
    Output.puts("#{GPUUtils.memory_status_string()}")

    # Wrap entire architecture benchmark in try/rescue
    try do
      # Merge with base options (hidden_sizes specified per-architecture for GPU optimization)
      opts =
        Keyword.merge(
          [
            epochs: epochs,
            batch_size: batch_size,
            # Default, but each arch overrides
            hidden_sizes: [256, 256],
            learning_rate: 1.0e-4,
            warmup_steps: 10,
            # We handle split ourselves
            val_split: 0.0,
            checkpoint: "checkpoints/benchmark_#{arch_id}.axon"
          ],
          arch_opts
        )

      # Use precomputed embeddings (computed once before the loop)
      {prepared_train, prepared_val} =
        if opts[:temporal] do
          # Temporal architectures use precomputed sequence embeddings
          Output.puts("  Using precomputed sequence embeddings")
          {precomputed_train_seqs, precomputed_val_seqs}
        else
          # Non-temporal architectures use precomputed frame embeddings
          Output.puts("  Using precomputed frame embeddings")
          {precomputed_train_frames, precomputed_val_frames}
        end

      # Create trainer
      # Filter out nil values so defaults are used instead of being overridden with nil
      trainer_opts =
        [
          hidden_sizes: opts[:hidden_sizes],
          embed_config: embed_config,
          temporal: opts[:temporal],
          backbone: opts[:backbone],
          window_size: opts[:window_size],
          num_layers: opts[:num_layers],
          num_heads: opts[:num_heads],
          attention_every: opts[:attention_every],
          # Training hyperparameters (per-architecture overrides)
          learning_rate: opts[:learning_rate],
          max_grad_norm: opts[:max_grad_norm]
        ]
        |> Enum.reject(fn {_k, v} -> is_nil(v) end)

      trainer =
        Output.timed "Creating model" do
          Imitation.new(trainer_opts)
        end

      # Calculate batch count without materializing (memory efficient)
      num_train_samples = if opts[:temporal], do: prepared_train.size, else: prepared_train.size
      num_batches = div(num_train_samples, opts[:batch_size])
      Output.puts("#{num_batches} train batches (streaming, not pre-materialized)")

      Output.warning("First batch triggers JIT compilation (may take 2-5 min)")

      # Verify batch shape before training (diagnose data issues early)
      test_batch =
        if opts[:temporal] do
          Data.batched_sequences(prepared_train,
            batch_size: min(4, opts[:batch_size]),
            shuffle: false
          )
          |> Enum.take(1)
          |> List.first()
        else
          Data.batched(prepared_train, batch_size: min(4, opts[:batch_size]), shuffle: false)
          |> Enum.take(1)
          |> List.first()
        end

      if test_batch do
        Output.puts("  Batch states shape: #{inspect(Nx.shape(test_batch.states))}")
        Output.puts("  Batch actions keys: #{inspect(Map.keys(test_batch.actions))}")
      else
        Output.error("  Failed to create test batch!")
      end

      # Training loop with timing
      start_time = System.monotonic_time(:millisecond)
      num_epochs = opts[:epochs]

      {_final_trainer, epoch_metrics} =
        Enum.reduce(1..num_epochs, {trainer, []}, fn epoch, {t, metrics} ->
          epoch_start = System.monotonic_time(:millisecond)

          # Create fresh batch stream each epoch (lazy, memory efficient)
          # Shuffle happens inside the batch creator via seed
          batches =
            if opts[:temporal] do
              Data.batched_sequences(prepared_train,
                batch_size: opts[:batch_size],
                shuffle: true,
                seed: epoch
              )
            else
              Data.batched(prepared_train,
                batch_size: opts[:batch_size],
                shuffle: true,
                seed: epoch
              )
            end

          # Train epoch with progress
          # Losses are now tensors (not numbers) to avoid blocking GPU→CPU transfers
          {updated_t, losses} =
            batches
            |> Enum.with_index(1)
            |> Enum.reduce({t, []}, fn {batch, batch_idx}, {tr, ls} ->
              # Update progress bar
              Output.progress_bar(batch_idx, num_batches, label: "Epoch #{epoch}/#{num_epochs}")

              # Wrap train_step with error handling to see actual error
              try do
                {new_tr, m} = Imitation.train_step(tr, batch, nil)
                {new_tr, [m.loss | ls]}
              rescue
                e ->
                  Output.progress_done()
                  Output.error("Train step failed at batch #{batch_idx}")
                  Output.error("Batch states shape: #{inspect(Nx.shape(batch.states))}")
                  Output.error("Error: #{Exception.message(e)}")
                  reraise e, __STACKTRACE__
              end
            end)

          Output.progress_done()

          epoch_time = System.monotonic_time(:millisecond) - epoch_start
          num_losses = length(losses)
          # Single GPU→CPU transfer at epoch end (not per-batch)
          avg_loss = losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()
          batches_per_sec = num_losses / (epoch_time / 1000)

          # Detect NaN/infinity early and warn
          if avg_loss in [:nan, :infinity, :neg_infinity] do
            Output.warning("Loss became #{avg_loss} - numeric instability detected")
          end

          # Validation (create batches lazily, don't materialize all at once)
          # GC before validation to release training batch memory (helps with Jamba OOM)
          :erlang.garbage_collect()

          val_batches =
            if opts[:temporal] do
              Data.batched_sequences(prepared_val, batch_size: opts[:batch_size], shuffle: false)
            else
              Data.batched(prepared_val, batch_size: opts[:batch_size], shuffle: false)
            end

          # Compute validation loss with streaming mean to avoid accumulating all tensors
          val_losses =
            Enum.map(val_batches, fn batch ->
              Imitation.evaluate_batch(updated_t, batch).loss
            end)

          val_loss =
            if length(val_losses) > 0 do
              val_losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()
            else
              avg_loss
            end

          Output.puts(
            "  Epoch #{epoch}: loss=#{safe_round.(avg_loss, 4)} val=#{safe_round.(val_loss, 4)} (#{safe_round.(batches_per_sec, 1)} batch/s)"
          )

          epoch_entry = %{
            epoch: epoch,
            train_loss: avg_loss,
            val_loss: val_loss,
            batches_per_sec: batches_per_sec,
            time_ms: epoch_time
          }

          {updated_t, [epoch_entry | metrics]}
        end)

      total_time = System.monotonic_time(:millisecond) - start_time
      epoch_metrics = Enum.reverse(epoch_metrics)

      # Final metrics
      final_train = List.last(epoch_metrics).train_loss
      final_val = List.last(epoch_metrics).val_loss
      avg_speed = Enum.sum(Enum.map(epoch_metrics, & &1.batches_per_sec)) / length(epoch_metrics)

      Output.success(
        "Complete: val=#{safe_round.(final_val, 4)}, speed=#{safe_round.(avg_speed, 1)} batch/s, time=#{safe_round.(total_time / 1000, 1)}s"
      )

      # Inference benchmarking - measure single batch latency
      Output.puts("  Measuring inference latency...")
      inference_batch = test_batch

      # Warmup inference (JIT compile)
      _ = Imitation.evaluate_batch(trainer, inference_batch)

      # Measure 10 inference passes
      inference_times =
        for _ <- 1..10 do
          start = System.monotonic_time(:microsecond)
          _ = Imitation.evaluate_batch(trainer, inference_batch)
          System.monotonic_time(:microsecond) - start
        end

      avg_inference_us = Enum.sum(inference_times) / length(inference_times)
      inference_batch_size = Nx.axis_size(inference_batch.states, 0)
      per_sample_us = avg_inference_us / inference_batch_size

      Output.puts(
        "  Inference: #{safe_round.(avg_inference_us / 1000, 2)}ms/batch, #{safe_round.(per_sample_us, 1)}μs/sample"
      )

      # Theoretical complexity (for documentation)
      theoretical_complexity =
        case arch_id do
          :mlp -> "O(1)"
          :lstm -> "O(L) sequential"
          :gru -> "O(L) sequential"
          :gated_ssm -> "O(L) sequential (simplified)"
          :mamba -> "O(L) work, O(log L) depth (parallel scan)"
          :mamba_nif -> "O(L) CUDA kernel (5x faster)"
          :jamba -> "O(L) + O(L²) hybrid"
          :attention -> "O(L²)"
          :sliding_window -> "O(L²) parallel"
          :lstm_hybrid -> "O(L) + O(L²)"
          _ -> "unknown"
        end

      # Return as list for flat_map
      [
        %{
          id: arch_id,
          name: arch_name,
          final_train_loss: final_train,
          final_val_loss: final_val,
          avg_batches_per_sec: avg_speed,
          total_time_ms: total_time,
          inference_us_per_batch: avg_inference_us,
          inference_us_per_sample: per_sample_us,
          theoretical_complexity: theoretical_complexity,
          epochs: epoch_metrics,
          config:
            Map.new(
              Keyword.take(opts, [:temporal, :backbone, :window_size, :num_layers, :hidden_sizes])
            )
        }
      ]
    rescue
      e ->
        Output.error("Architecture #{arch_name} failed: #{Exception.message(e)}")

        if continue_on_error do
          Output.warning("Continuing with next architecture (--continue-on-error)")
          []
        else
          reraise e, __STACKTRACE__
        end
    end
  end)

Output.divider()

# Handle case where all architectures failed
if length(results) == 0 do
  Output.error("All architectures failed! No results to display.")
  System.halt(1)
end

# Sort by validation loss
sorted_results = Enum.sort_by(results, & &1.final_val_loss)

# Print comparison table
Output.section("Benchmark Results")
Output.puts("Ranked by validation loss (lower is better):\n")

Output.puts(
  "  Rank | Architecture    | Val Loss | Train Loss | Speed (b/s) | Inference | Complexity"
)

Output.puts(
  "  -----+-----------------+----------+------------+-------------+-----------+-----------"
)

sorted_results
|> Enum.with_index(1)
|> Enum.each(fn {r, rank} ->
  name = String.pad_trailing(r.name, 15)
  val = safe_round.(r.final_val_loss, 4) |> to_string() |> String.pad_leading(8)
  train = safe_round.(r.final_train_loss, 4) |> to_string() |> String.pad_leading(10)
  speed = safe_round.(r.avg_batches_per_sec, 1) |> to_string() |> String.pad_leading(11)
  inference = "#{safe_round.(r.inference_us_per_batch / 1000, 1)}ms" |> String.pad_leading(9)
  complexity = String.pad_trailing(r.theoretical_complexity, 10)

  Output.puts(
    "  #{rank}    | #{name} | #{val} | #{train} | #{speed} | #{inference} | #{complexity}"
  )
end)

# Best architecture
best = List.first(sorted_results)
Output.puts("")

Output.success(
  "Best architecture: #{best.name} (val_loss=#{safe_round.(best.final_val_loss, 4)})"
)

# Save results with timestamp in filename
timestamp = DateTime.utc_now()
timestamp_str = Calendar.strftime(timestamp, "%Y%m%d_%H%M%S")
results_path = "checkpoints/benchmark_results_#{timestamp_str}.json"
report_path = "checkpoints/benchmark_report_#{timestamp_str}.html"
File.mkdir_p!("checkpoints")

json_results = %{
  timestamp: DateTime.to_iso8601(timestamp),
  machine: %{
    gpu: gpu_info.name,
    gpu_memory_gb: safe_round.(gpu_info.memory_gb, 1)
  },
  config: %{
    replay_dir: replay_dir,
    max_files: max_files,
    epochs: epochs,
    batch_size: batch_size,
    train_frames: train_dataset.size,
    val_frames: val_dataset.size
  },
  results: sorted_results,
  best: best.id
}

File.write!(results_path, Jason.encode!(json_results, pretty: true))
Output.puts("Results saved to #{results_path}")

# Generate comparison plot

# Build loss comparison data
plot_data =
  results
  |> Enum.flat_map(fn r ->
    Enum.map(r.epochs, fn e ->
      %{architecture: r.name, epoch: e.epoch, loss: e.val_loss, type: "val"}
    end)
  end)

comparison_plot =
  VegaLite.new(width: 700, height: 400, title: "Architecture Comparison - Validation Loss")
  |> VegaLite.data_from_values(plot_data)
  |> VegaLite.mark(:line, point: true)
  |> VegaLite.encode_field(:x, "epoch", type: :quantitative, title: "Epoch")
  |> VegaLite.encode_field(:y, "loss",
    type: :quantitative,
    title: "Validation Loss",
    scale: [zero: false]
  )
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, title: "Architecture")

# Build training speed bar chart data
speed_data =
  sorted_results
  |> Enum.map(fn r -> %{architecture: r.name, speed: r.avg_batches_per_sec} end)

speed_plot =
  VegaLite.new(width: 700, height: 300, title: "Training Speed (batches/sec)")
  |> VegaLite.data_from_values(speed_data)
  |> VegaLite.mark(:bar)
  |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "-y")
  |> VegaLite.encode_field(:y, "speed", type: :quantitative, title: "Batches/sec")
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)

# Build inference speed bar chart data
inference_data =
  sorted_results
  |> Enum.map(fn r -> %{architecture: r.name, latency_ms: r.inference_us_per_batch / 1000} end)

inference_plot =
  VegaLite.new(width: 700, height: 300, title: "Inference Latency (ms/batch)")
  |> VegaLite.data_from_values(inference_data)
  |> VegaLite.mark(:bar)
  |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "y")
  |> VegaLite.encode_field(:y, "latency_ms", type: :quantitative, title: "Latency (ms)")
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)

# Build theoretical complexity chart data
# For training: includes backward pass, so multiply by ~3x
# For inference: just forward pass
# Normalized to MLP = 1 (baseline)
window_size = 30

theoretical_data =
  sorted_results
  |> Enum.flat_map(fn r ->
    {train_ops, inference_ops} =
      case r.id do
        # O(1) baseline
        :mlp ->
          {1, 1}

        # O(L) sequential, 3x for backprop
        :lstm ->
          {window_size * 3, window_size}

        # O(L) sequential, 3x for backprop
        :gru ->
          {window_size * 3, window_size}

        # O(L) sequential (simplified SSM, no parallel scan)
        :gated_ssm ->
          {window_size * 3, window_size}

        # O(L) work, O(log L) depth - parallel scan reduces effective latency
        :mamba ->
          {round(:math.log2(window_size)) * 3, round(:math.log2(window_size))}

        # O(L) CUDA kernel - faster than XLA parallel scan
        :mamba_nif ->
          {round(:math.log2(window_size)), round(:math.log2(window_size)) / 2}

        # O(L²) - pure attention
        :attention ->
          {window_size * window_size * 3, window_size * window_size}

        # O(L²) - sliding window attention (same as pure attention)
        :sliding_window ->
          {window_size * window_size * 3, window_size * window_size}

        # O(L) + O(L²) - LSTM sequential + attention quadratic
        :lstm_hybrid ->
          {window_size * 3 + window_size * window_size * 3,
           window_size + window_size * window_size}

        # O(L) + O(L²/3) - Mamba + sparse attention
        :jamba ->
          {window_size + div(window_size * window_size, 3),
           window_size + div(window_size * window_size, 3)}

        _ ->
          {1, 1}
      end

    [
      %{architecture: r.name, type: "Training (relative)", ops: train_ops},
      %{architecture: r.name, type: "Inference (relative)", ops: inference_ops}
    ]
  end)

# Use log scale for theoretical complexity comparison
# Faceted bar chart - one column per phase (Training/Inference)
theoretical_plot =
  VegaLite.new(
    width: 300,
    height: 300,
    title: "Theoretical Complexity (relative to MLP baseline, L=30)"
  )
  |> VegaLite.data_from_values(theoretical_data)
  |> VegaLite.mark(:bar, tooltip: true)
  |> VegaLite.encode_field(:x, "architecture",
    type: :nominal,
    title: "Architecture",
    axis: [label_angle: -45]
  )
  |> VegaLite.encode_field(:y, "ops",
    type: :quantitative,
    title: "Relative Operations (log scale)",
    scale: [type: :log]
  )
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)
  |> VegaLite.encode_field(:column, "type", type: :nominal, title: "Phase")

# Build training loss bar chart (final val loss comparison)
loss_bar_data =
  sorted_results
  |> Enum.map(fn r -> %{architecture: r.name, loss: r.final_val_loss} end)

loss_bar_plot =
  VegaLite.new(width: 700, height: 300, title: "Final Validation Loss (lower is better)")
  |> VegaLite.data_from_values(loss_bar_data)
  |> VegaLite.mark(:bar)
  |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "y")
  |> VegaLite.encode_field(:y, "loss",
    type: :quantitative,
    title: "Validation Loss",
    scale: [zero: false]
  )
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)

# Save report (use to_spec + Jason instead of deprecated Export.to_json)
spec = comparison_plot |> VegaLite.to_spec() |> Jason.encode!()
speed_spec = speed_plot |> VegaLite.to_spec() |> Jason.encode!()
inference_spec = inference_plot |> VegaLite.to_spec() |> Jason.encode!()
theoretical_spec = theoretical_plot |> VegaLite.to_spec() |> Jason.encode!()
loss_bar_spec = loss_bar_plot |> VegaLite.to_spec() |> Jason.encode!()

html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Architecture Benchmark Report</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 40px; max-width: 900px; margin: 0 auto; }
    h1 { color: #333; }
    .summary { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .notes { background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #333; color: white; }
    tr:nth-child(1) td { background: #d4edda; font-weight: bold; }
    .winner { color: #28a745; font-weight: bold; }
    .plot { margin: 30px 0; }
    .complexity { font-family: monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
  </style>
</head>
<body>
  <h1>Architecture Benchmark Report</h1>

  <div class="summary">
    <h3>Configuration</h3>
    <p><strong>Machine:</strong> #{gpu_info.name} (#{safe_round.(gpu_info.memory_gb, 1)} GB)</p>
    <p><strong>Replays:</strong> #{max_files} files (#{train_dataset.size} train / #{val_dataset.size} val frames)</p>
    <p><strong>Epochs:</strong> #{epochs}</p>
    <p><strong>Batch size:</strong> #{batch_size}</p>
    <p><strong>Generated:</strong> #{DateTime.utc_now() |> Calendar.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
  </div>

  <h2>Results (ranked by validation loss)</h2>
  <table>
    <tr><th>Rank</th><th>Architecture</th><th>Val Loss</th><th>Train Loss</th><th>Speed</th><th>Inference</th><th>Complexity</th><th>Time</th></tr>
    #{sorted_results |> Enum.with_index(1) |> Enum.map(fn {r, rank} -> "<tr><td>#{rank}</td><td>#{r.name}</td><td>#{safe_round.(r.final_val_loss, 4)}</td><td>#{safe_round.(r.final_train_loss, 4)}</td><td>#{safe_round.(r.avg_batches_per_sec, 1)} b/s</td><td>#{safe_round.(r.inference_us_per_batch / 1000, 2)} ms</td><td><span class=\"complexity\">#{r.theoretical_complexity}</span></td><td>#{safe_round.(r.total_time_ms / 1000, 1)}s</td></tr>" end) |> Enum.join("\n")}
  </table>

  <p class="winner">Best: #{best.name}</p>

  <div class="notes">
    <h3>Notes</h3>
    <p><em>Add your observations here after reviewing the results.</em></p>
    <ul>
      <li>Loss curves: Do the temporal architectures show improvement over epochs?</li>
      <li>Mamba/Jamba may need more epochs or different hyperparameters to converge</li>
      <li>Consider training speed vs accuracy tradeoffs for your use case</li>
    </ul>
  </div>

  <h2>Validation Loss Comparison</h2>
  <div id="loss_bar_plot" class="plot"></div>

  <h2>Loss Curves (over epochs)</h2>
  <div id="loss_plot" class="plot"></div>

  <h2>Measured Training Speed</h2>
  <p>Batches per second during training. Higher is better.</p>
  <div id="speed_plot" class="plot"></div>

  <h2>Measured Inference Latency</h2>
  <p>Lower is better. Target for 60 FPS gameplay: &lt;16.7ms per frame.</p>
  <div id="inference_plot" class="plot"></div>

  <h2>Theoretical Complexity</h2>
  <p>Big-O complexity relative to MLP baseline (L = window size = 30). Log scale. Lower is better.</p>
  <ul>
    <li><strong>MLP:</strong> O(1) - constant time, no temporal context</li>
    <li><strong>LSTM/GRU:</strong> O(L) - sequential, cannot parallelize</li>
    <li><strong>Mamba:</strong> O(L) - parallel scan, GPU-friendly</li>
    <li><strong>Mamba NIF:</strong> O(L) - CUDA kernel, 5x faster than XLA</li>
    <li><strong>Attention:</strong> O(L²) - quadratic, but highly parallel</li>
    <li><strong>Jamba:</strong> O(L) + O(L²/3) - hybrid, attention every 3 layers</li>
  </ul>
  <div id="theoretical_plot" class="plot"></div>

  <script>
    vegaEmbed('#loss_bar_plot', #{loss_bar_spec});
    vegaEmbed('#loss_plot', #{spec});
    vegaEmbed('#speed_plot', #{speed_spec});
    vegaEmbed('#inference_plot', #{inference_spec});
    vegaEmbed('#theoretical_plot', #{theoretical_spec});
  </script>
</body>
</html>
"""

File.write!(report_path, html)
Output.success("Report saved to #{report_path}")

# Create symlinks to latest reports for easy access
latest_json = "checkpoints/benchmark_results_latest.json"
latest_html = "checkpoints/benchmark_report_latest.html"
File.rm(latest_json)
File.rm(latest_html)
File.ln_s!(Path.basename(results_path), latest_json)
File.ln_s!(Path.basename(report_path), latest_html)
Output.puts("Symlinked to #{latest_json} and #{latest_html}")

Output.puts("")
Output.puts("Open #{report_path} in a browser to see the comparison.")
