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
#   --lazy-sequences              Slice sequences on-the-fly (150 MB RAM vs 13 GB)
#                                 Slightly slower batching but enables low-RAM training
#   --eager-sequences             Pre-build all sequences (default, faster but 13+ GB RAM)
#   --grad-norms                   Log per-layer gradient norms (diagnose NaN)
#   --quiet, -q                   Suppress XLA/CUDA warnings and Logger noise
#
# Available architectures:
#   Basic: mlp, gated_ssm, kan
#   Recurrent: lstm, gru, lstm_hybrid, reservoir, deltanet, ttt
#   SSM: mamba, mamba_ssd, s4, s4d, s5, h3
#   Linear Attention: rwkv, gla, hgrn, retnet, performer, fnet
#   Hybrid: jamba, zamba, griffin, hawk, xlstm
#   Transformer: sliding_window (also: attention), perceiver
#   Memory: hopfield, ntm
#   Other: liquid, decision_transformer, snn, bayesian
#
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
#
# Quiet Mode:
#   --quiet suppresses XLA/CUDA warnings but keeps errors visible.
#   To capture warnings to a file while keeping terminal clean:
#     mix run scripts/benchmark_architectures.exs --quiet ... 2>warnings.log
#   This lets you review warnings later if something goes wrong.

# Limit inspect output to prevent overwhelming logs with tensor dumps
Application.put_env(:elixir, :inspect, limit: 10, printable_limit: 100)

# Use async CUDA allocator to reduce memory fragmentation between architectures
# This helps prevent OOM when switching from MLP to Mamba/Jamba
System.put_env("TF_GPU_ALLOCATOR", "cuda_malloc_async")

# Use platform allocator so CUDA actually frees memory between architectures.
# The default BFC allocator grows a pool that never shrinks, so by architecture #25+
# the GPU is saturated with stale JIT caches. Platform allocator uses cudaMalloc/cudaFree
# directly — slightly slower per-allocation but essential for multi-architecture runs.
System.put_env("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

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

# Only override EXLA config if CUDA target is set; otherwise use app default (:host CPU)
if System.get_env("EXLA_TARGET") == "cuda" do
  Application.put_env(:exla, :clients, cuda: gpu_config)
else
  IO.puts("[GPU] No EXLA_TARGET=cuda, using CPU (host) backend")
end

alias ExPhil.CLI
alias ExPhil.Data.Peppi
alias ExPhil.Training.{Data, GPUUtils, Imitation, Output, Prefetcher}
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

# Parse args using CLI module
# Note: GPU config and quiet mode already parsed early (before EXLA loads)
opts = CLI.parse_args(System.argv(),
  flags: [:verbosity, :replay, :training, :benchmark, :common],
  defaults: [
    replays: "./replays",
    max_files: 30,
    epochs: 3,
    # Auto-detect batch size based on GPU availability
    batch_size: (if System.get_env("EXLA_TARGET") == "cuda", do: 128, else: 64)
  ]
)

# Extract options to variables for easier use in rest of script
replay_dir = opts[:replays]
max_files = opts[:max_files]
epochs = opts[:epochs]
batch_size = opts[:batch_size]
continue_on_error = opts[:continue_on_error]
grad_norms_enabled = opts[:grad_norms]
cache_enabled = opts[:cache_embeddings]
force_recompute = opts[:no_cache]
cache_dir = opts[:cache_dir]

# Parse comma-separated architecture lists
only_archs =
  case opts[:only] do
    nil -> nil
    str -> str |> String.split(",") |> Enum.map(&String.to_atom(String.trim(&1)))
  end

skip_archs =
  case opts[:skip] do
    nil -> []
    str -> str |> String.split(",") |> Enum.map(&String.to_atom(String.trim(&1)))
  end

# Lazy vs eager sequence loading
# Lazy: 150 MB RAM, slices on-the-fly (slightly slower batching)
# Eager: 13+ GB RAM, pre-builds all sequences (faster batching)
# Default to eager for backwards compatibility; use --lazy-sequences for low RAM
use_lazy_sequences = opts[:lazy_sequences] and not opts[:eager_sequences]

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
     batch_size: 64,
     dropout: 0.1
   ]},
  {:mamba, "Mamba (parallel scan)",
   [
     temporal: true,
     backbone: :mamba,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     # NOTE: Mamba defaults to dropout: 0.0 which causes severe overfitting!
     # Must explicitly set dropout for fair comparison with MLP (which uses 0.1)
     dropout: 0.1
   ]},
  # NOTE: mamba_nif excluded from training benchmark - gradients don't flow through NIF
  # Use for inference-only comparison: train with :mamba, infer with :mamba_nif
  # {:mamba_nif, "Mamba NIF (CUDA)", [temporal: true, backbone: :mamba_nif, ...]},
  {:jamba, "Jamba (Mamba+Attn)",
   [
     temporal: true,
     backbone: :jamba,
     window_size: 30,
     num_layers: 2,
     attention_every: 2,
     hidden_sizes: [256, 256],
     # Reduced from 32 - Jamba OOMs on validation after epoch with batch_size: 32
     batch_size: 16,
     # Jamba needs very conservative settings to avoid NaN (attention + Mamba can explode)
     # - Ultra-low LR (5e-6) - diverged at epoch 3 with 1e-5
     # - Strict gradient clipping (0.25)
     # - Reduced layers (2 instead of 3)
     learning_rate: 5.0e-6,
     max_grad_norm: 0.25
   ]},
  {:lstm, "LSTM",
   [temporal: true, backbone: :lstm, window_size: 30, num_layers: 1, hidden_sizes: [256, 256], dropout: 0.1]},
  {:gru, "GRU",
   [temporal: true, backbone: :gru, window_size: 30, num_layers: 1, hidden_sizes: [256, 256], dropout: 0.1]},
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
   ]},

  # === New SSM Architectures ===
  {:zamba, "Zamba (Shared Attn)",
   [
     temporal: true,
     backbone: :zamba,
     window_size: 30,
     num_layers: 4,
     # Shared attention every 2 layers (applied 2x total)
     attention_every: 2,
     hidden_sizes: [256, 256],
     batch_size: 32,
     dropout: 0.1,
     # Shared attention + Mamba needs conservative settings to avoid NaN
     learning_rate: 1.0e-5,
     max_grad_norm: 0.5
   ]},
  {:mamba_ssd, "Mamba-2 (SSD)",
   [
     temporal: true,
     backbone: :mamba_ssd,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:s5, "S5 (Simplified SSM)",
   [
     temporal: true,
     backbone: :s5,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},

  # === Linear Attention Architectures ===
  {:rwkv, "RWKV (Linear RNN)",
   [
     temporal: true,
     backbone: :rwkv,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:gla, "GLA (Gated Linear)",
   [
     temporal: true,
     backbone: :gla,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:hgrn, "HGRN (Hierarchical)",
   [
     temporal: true,
     backbone: :hgrn,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},

  # === SSM Architectures (classic) ===
  {:s4, "S4 (Blelloch Scan)",
   [
     temporal: true,
     backbone: :s4,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:s4d, "S4D (Diagonal SSM)",
   [
     temporal: true,
     backbone: :s4d,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:h3, "H3 (Hungry Hippos)",
   [
     temporal: true,
     backbone: :h3,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1,
     # H3 SSM learnable a_log/dt_log are numerically sensitive
     learning_rate: 1.0e-5,
     max_grad_norm: 0.5
   ]},

  # === Hybrid RNN+Attention ===
  {:griffin, "Griffin (RG-LRU+Attn)",
   [
     temporal: true,
     backbone: :griffin,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:hawk, "Hawk (Pure RG-LRU)",
   [
     temporal: true,
     backbone: :hawk,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:xlstm, "xLSTM (Mixed sLSTM+mLSTM)",
   [
     temporal: true,
     backbone: :xlstm,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},

  # === Linear Attention (additional) ===
  {:retnet, "RetNet (Multi-Scale Decay)",
   [
     temporal: true,
     backbone: :retnet,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:performer, "Performer (FAVOR+)",
   [
     temporal: true,
     backbone: :performer,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:deltanet, "DeltaNet (Delta Rule)",
   [
     temporal: true,
     backbone: :deltanet,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},
  {:fnet, "FNet (Fourier Mixing)",
   [
     temporal: true,
     backbone: :fnet,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},

  # === Attention Variants ===
  {:perceiver, "Perceiver IO (Latent Bottleneck)",
   [
     temporal: true,
     backbone: :perceiver,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1
   ]},

  # === Recurrent (additional) ===
  {:ttt, "TTT (Test-Time Training)",
   [
     temporal: true,
     backbone: :ttt,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [256, 256],
     batch_size: 64,
     dropout: 0.1,
     # TTT inner-loop weight updates are numerically sensitive
     learning_rate: 1.0e-5,
     max_grad_norm: 0.5
   ]},
  {:reservoir, "Reservoir (Echo State)",
   [
     temporal: true,
     backbone: :reservoir,
     window_size: 30,
     hidden_sizes: [256, 256],
     batch_size: 64
   ]},

  # === Memory Architectures ===
  {:hopfield, "Hopfield (Associative Memory)",
   [
     temporal: true,
     backbone: :hopfield,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [256, 256],
     batch_size: 64,
     max_grad_norm: 1.0
   ]},
  {:ntm, "NTM (Neural Turing Machine)",
   [
     temporal: true,
     backbone: :ntm,
     window_size: 30,
     hidden_sizes: [256, 256],
     batch_size: 64,
     max_grad_norm: 1.0
   ]},

  # === Other Architectures ===
  {:liquid, "Liquid (Neural ODE)",
   [
     temporal: true,
     backbone: :liquid,
     window_size: 30,
     num_layers: 1,
     hidden_sizes: [128, 128],
     hidden_size: 64,
     batch_size: 4,
     dropout: 0.1,
     # Liquid ODE creates massive JIT graphs — minimize everything to fit GPU
     # Uses hidden_size: 64 (backbone) vs hidden_sizes: [128,128] (policy heads)
     integration_steps: 1,
     max_grad_norm: 1.0
   ]},
  {:decision_transformer, "Decision Transformer",
   [
     temporal: true,
     backbone: :decision_transformer,
     window_size: 30,
     num_layers: 2,
     num_heads: 4,
     hidden_sizes: [128, 128],
     batch_size: 16,
     # Transformer-based, massive JIT graph — reduce batch/hidden to fit GPU
     max_grad_norm: 1.0,
     learning_rate: 1.0e-4
   ]},
  {:kan, "KAN (Kolmogorov-Arnold)",
   [
     temporal: true,
     backbone: :kan,
     window_size: 30,
     num_layers: 2,
     hidden_sizes: [128, 128],
     batch_size: 16,
     dropout: 0.1,
     # KAN basis expansions create large intermediate tensors — reduce to fit GPU
     max_grad_norm: 1.0
   ]},
  {:snn, "SNN (Spiking Neural Network)",
   [
     temporal: true,
     backbone: :snn,
     window_size: 30,
     hidden_sizes: [256, 256],
     batch_size: 64,
     max_grad_norm: 1.0
   ]},
  {:bayesian, "Bayesian NN (Weight Uncertainty)",
   [
     temporal: true,
     backbone: :bayesian,
     window_size: 30,
     hidden_sizes: [256, 256],
     batch_size: 64,
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
  {"Gradient norms", if(grad_norms_enabled, do: "enabled (per-epoch)", else: "disabled")},
  {"Embedding cache", if(cache_enabled, do: "enabled (#{cache_dir})", else: "disabled")},
  {"Sequence mode", if(use_lazy_sequences, do: "lazy (150 MB RAM)", else: "eager (13+ GB RAM)")},
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

    # Transfer embeddings to GPU once (cache loads as CPU/BinaryBackend)
    # This avoids the "Embeddings on CPU" warning during training
    # Only transfer if under 2GB threshold - larger tensors transfer batch-wise
    gpu_threshold_mb = 2000

    # Handle empty val set (0 frames parsed from val replays)
    has_val_embeddings = val_emb.embedded_frames != nil
    train_size_mb = Nx.byte_size(train_emb.embedded_frames) / 1_000_000
    val_size_mb = if has_val_embeddings, do: Nx.byte_size(val_emb.embedded_frames) / 1_000_000, else: 0.0

    {train_emb, val_emb} =
      if train_size_mb <= gpu_threshold_mb do
        Output.puts("Transferring embeddings to GPU (#{Float.round(train_size_mb, 1)} MB train, #{Float.round(val_size_mb, 1)} MB val)...")
        train_emb = Map.update!(train_emb, :embedded_frames, &Nx.backend_transfer(&1, EXLA.Backend))
        val_emb = if has_val_embeddings do
          Map.update!(val_emb, :embedded_frames, &Nx.backend_transfer(&1, EXLA.Backend))
        else
          val_emb
        end
        Output.success("Embeddings on GPU")
        {train_emb, val_emb}
      else
        Output.puts("Keeping embeddings on CPU (#{Float.round(train_size_mb, 1)} MB > 2GB threshold)")
        Output.puts("  Batches will be transferred during training (expect per-epoch warning)")
        {train_emb, val_emb}
      end

    {train_emb, val_emb}
  else
    {nil, nil}
  end

# Build sequence embeddings from frame embeddings (30x faster than re-embedding!)
# This reuses the frame embeddings computed above via tensor slicing
# In lazy mode, we skip this and slice on-the-fly during batching (150 MB vs 13 GB RAM)
{precomputed_train_seqs, precomputed_val_seqs} =
  if has_temporal do
    window_size = 30
    stride = 1

    if use_lazy_sequences do
      # Lazy mode: just create sequence structure, don't build embeddings
      # Batching will slice from frame embeddings on-the-fly
      Output.puts("Using lazy sequence mode (150 MB RAM vs 13 GB)")
      Output.puts("  Sequences will be sliced on-the-fly during batching")

      train_seq =
        Data.to_sequences(train_dataset, window_size: window_size, stride: stride)
        |> Map.put(:embedded_frames, precomputed_train_frames.embedded_frames)

      val_seq =
        Data.to_sequences(val_dataset, window_size: window_size, stride: stride)
        |> Map.put(:embedded_frames, precomputed_val_frames.embedded_frames)

      {train_seq, val_seq}
    else
      # Eager mode: pre-build all sequence embeddings (faster batching but 13+ GB RAM)
      Output.puts("Building sequence embeddings from frame embeddings (30x faster)...")
      Output.puts("  (Use --lazy-sequences for 150 MB RAM and better GPU efficiency)")

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

      # NOTE: Eager sequence mode stores embeddings as an array of individual tensors.
      # GPU transfer happens per-batch during training (stack + transfer).
      # For better GPU efficiency, use --lazy-sequences which keeps frame embeddings
      # as a single GPU tensor and slices on-the-fly.

      {train_seq, val_seq}
    end
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
            shuffle: false,
            lazy: use_lazy_sequences,
            window_size: opts[:window_size] || 30,
            stride: opts[:stride] || 1
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
      # Uses reduce_while for NaN early stopping — no point training more epochs
      # once gradients have exploded
      start_time = System.monotonic_time(:millisecond)
      num_epochs = opts[:epochs]

      {_final_trainer, epoch_metrics} =
        Enum.reduce_while(1..num_epochs, {trainer, []}, fn epoch, {t, metrics} ->
          epoch_start = System.monotonic_time(:millisecond)

          # Create fresh batch stream each epoch (lazy, memory efficient)
          # Shuffle happens inside the batch creator via seed
          batches =
            if opts[:temporal] do
              Data.batched_sequences(prepared_train,
                batch_size: opts[:batch_size],
                shuffle: true,
                seed: epoch,
                lazy: use_lazy_sequences,
                window_size: opts[:window_size] || 30,
                stride: opts[:stride] || 1
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
          # Use Prefetcher to overlap CPU→GPU transfer with GPU compute (~10-20% speedup)
          {updated_t, losses} =
            Prefetcher.reduce_stream_indexed(
              batches,
              {t, []},
              fn batch, batch_idx, {tr, ls} ->
                # batch_idx is 0-based from Prefetcher, add 1 for display
                display_idx = batch_idx + 1

                # Update progress bar
                Output.progress_bar(display_idx, num_batches, label: "Epoch #{epoch}/#{num_epochs}")

                # Wrap train_step with error handling to see actual error
                try do
                  {new_tr, m} = Imitation.train_step(tr, batch, nil)
                  {new_tr, [m.loss | ls]}
                rescue
                  e ->
                    Output.progress_done()
                    Output.error("Train step failed at batch #{display_idx}")
                    Output.error("Batch states shape: #{inspect(Nx.shape(batch.states))}")
                    Output.error("Error: #{Exception.message(e)}")
                    reraise e, __STACKTRACE__
                end
              end,
              buffer_size: 2
            )

          Output.progress_done()

          epoch_time = System.monotonic_time(:millisecond) - epoch_start
          num_losses = length(losses)
          # Single GPU→CPU transfer at epoch end (not per-batch)
          avg_loss = losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()
          batches_per_sec = num_losses / (epoch_time / 1000)

          # NaN early stopping — detect and skip remaining epochs
          nan_detected = avg_loss in [:nan, :infinity, :neg_infinity]

          if nan_detected do
            Output.warning("Loss became #{avg_loss} - numeric instability detected")

            # Log gradient norms on NaN to diagnose which layers exploded
            if grad_norms_enabled do
              Output.puts("  Computing gradient norms to diagnose instability...")
              try do
                sample_batch = test_batch
                norms = Imitation.compute_grad_norms(updated_t, sample_batch)
                top_norms = Enum.take(norms, 10)
                Output.puts("  Top gradient norms (L2):")
                Enum.each(top_norms, fn {layer, norm} ->
                  marker = if norm > 100.0, do: " <<<", else: ""
                  Output.puts("    #{safe_round.(norm, 4)}\t#{layer}#{marker}")
                end)
              rescue
                _ -> Output.puts("  (gradient norm computation failed — model may be in bad state)")
              end
            end
          end

          # Validation (create batches lazily, don't materialize all at once)
          # GC before validation to release training batch memory (helps with Jamba OOM)
          :erlang.garbage_collect()

          # Skip validation if loss is NaN (pointless and wastes time)
          val_loss =
            if nan_detected do
              :nan
            else
              val_batches =
                if opts[:temporal] do
                  Data.batched_sequences(prepared_val,
                    batch_size: opts[:batch_size],
                    shuffle: false,
                    lazy: use_lazy_sequences,
                    window_size: opts[:window_size] || 30,
                    stride: opts[:stride] || 1
                  )
                else
                  Data.batched(prepared_val, batch_size: opts[:batch_size], shuffle: false)
                end

              val_losses =
                Enum.map(val_batches, fn batch ->
                  Imitation.evaluate_batch(updated_t, batch).loss
                end)

              if length(val_losses) > 0 do
                val_losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()
              else
                avg_loss
              end
            end

          Output.puts(
            "  Epoch #{epoch}: loss=#{safe_round.(avg_loss, 4)} val=#{safe_round.(val_loss, 4)} (#{safe_round.(batches_per_sec, 1)} batch/s)"
          )

          # Log gradient norms at end of epoch 1 (if enabled and not NaN)
          if grad_norms_enabled and epoch == 1 and not nan_detected do
            Output.puts("  Gradient norms (L2, top 5 layers):")
            try do
              sample_batch = test_batch
              norms = Imitation.compute_grad_norms(updated_t, sample_batch)
              Enum.take(norms, 5)
              |> Enum.each(fn {layer, norm} ->
                Output.puts("    #{safe_round.(norm, 4)}\t#{layer}")
              end)
            rescue
              _ -> Output.puts("    (gradient norm computation failed)")
            end
          end

          epoch_entry = %{
            epoch: epoch,
            train_loss: avg_loss,
            val_loss: val_loss,
            batches_per_sec: batches_per_sec,
            time_ms: epoch_time
          }

          # Early stop on NaN — no point training further
          if nan_detected do
            Output.warning("Stopping early — loss is #{avg_loss}, remaining epochs would be wasted")
            {:halt, {updated_t, [epoch_entry | metrics]}}
          else
            {:cont, {updated_t, [epoch_entry | metrics]}}
          end
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
          # Basic
          :mlp -> "O(1)"
          :gated_ssm -> "O(L) sequential"
          # Recurrent
          :lstm -> "O(L) sequential"
          :gru -> "O(L) sequential"
          :lstm_hybrid -> "O(L) + O(L²)"
          # SSM
          :mamba -> "O(L) work, O(log L) depth"
          :mamba_nif -> "O(L) CUDA kernel"
          :mamba_ssd -> "O(L) work, O(log L) depth"
          :s5 -> "O(L) work, O(log L) depth"
          # Linear Attention
          :rwkv -> "O(L) linear RNN"
          :gla -> "O(L) linear attn"
          :hgrn -> "O(L) hierarchical"
          # Hybrid
          :jamba -> "O(L) + O(L²) hybrid"
          :zamba -> "O(L) + O(L²/N) shared"
          # Transformer
          :attention -> "O(L²)"
          :sliding_window -> "O(L²) parallel"
          :decision_transformer -> "O(L²) goal-cond"
          # Other
          :liquid -> "O(L) adaptive ODE"
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
# Use absolute operation counts (not normalized) for clearer comparison
window_size = 30

theoretical_data =
  sorted_results
  |> Enum.flat_map(fn r ->
    {train_ops, inference_ops} =
      case r.id do
        # === Basic ===
        # O(1) baseline - constant, but still some ops
        :mlp ->
          {10, 10}

        # O(L) sequential (simplified SSM, no parallel scan)
        :gated_ssm ->
          {window_size * 3, window_size}

        # === Recurrent ===
        # O(L) sequential, 3x for backprop
        :lstm ->
          {window_size * 3, window_size}

        # O(L) sequential, 3x for backprop
        :gru ->
          {window_size * 3, window_size}

        # O(L) + O(L²) - LSTM sequential + attention quadratic
        :lstm_hybrid ->
          {window_size * 3 + window_size * window_size * 3,
           window_size + window_size * window_size}

        # === SSM ===
        # O(L) work, O(log L) depth - parallel scan reduces effective latency
        :mamba ->
          {round(:math.log2(window_size)) * 3, round(:math.log2(window_size))}

        # O(L) CUDA kernel - faster than XLA parallel scan
        :mamba_nif ->
          {round(:math.log2(window_size)), round(:math.log2(window_size)) / 2}

        # O(L) work, O(log L) depth - Mamba-2 with SSD algorithm
        :mamba_ssd ->
          {round(:math.log2(window_size)) * 3, round(:math.log2(window_size))}

        # O(L) work, O(log L) depth - simplified state space
        :s5 ->
          {round(:math.log2(window_size)) * 3, round(:math.log2(window_size))}

        # === Linear Attention ===
        # O(L) - linear RNN with channel mixing
        :rwkv ->
          {window_size * 2, window_size}

        # O(L) - gated linear attention
        :gla ->
          {window_size * 2, window_size}

        # O(L) - hierarchical gated RNN
        :hgrn ->
          {window_size * 2, window_size}

        # === Hybrid ===
        # O(L) + O(L²/3) - Mamba + sparse attention
        :jamba ->
          {window_size + div(window_size * window_size, 3),
           window_size + div(window_size * window_size, 3)}

        # O(L) + O(L²/N) - Mamba + shared attention (N = attention_every)
        # Much cheaper than Jamba since attention weights are reused
        :zamba ->
          {window_size + div(window_size * window_size, 6),
           window_size + div(window_size * window_size, 6)}

        # === Transformer ===
        # O(L²) - pure attention
        :attention ->
          {window_size * window_size * 3, window_size * window_size}

        # O(L²) - sliding window attention (same as pure attention)
        :sliding_window ->
          {window_size * window_size * 3, window_size * window_size}

        # O(L²) - goal-conditioned transformer
        :decision_transformer ->
          {window_size * window_size * 3, window_size * window_size}

        # === Other ===
        # O(L) - adaptive ODE dynamics
        :liquid ->
          {window_size * 4, window_size * 2}

        _ ->
          {10, 10}
      end

    [
      %{architecture: r.name, type: "Training", ops: train_ops},
      %{architecture: r.name, type: "Inference", ops: inference_ops}
    ]
  end)

# Grouped bar chart with color encoding for phase
# Using xOffset for side-by-side bars within each architecture
# Use rect mark with explicit y2 baseline - bar mark doesn't work with log scale
# because bar tries to draw from 0, but log(0) = -∞
theoretical_plot =
  VegaLite.new(
    width: 600,
    height: 350,
    title: "Theoretical Complexity (L=30)"
  )
  |> VegaLite.data_from_values(theoretical_data)
  |> VegaLite.mark(:rect, tooltip: true)
  |> VegaLite.encode_field(:x, "architecture",
    type: :nominal,
    title: "Architecture",
    axis: [label_angle: -45]
  )
  |> VegaLite.encode_field(:y, "ops",
    type: :quantitative,
    title: "Relative Operations (log scale)",
    scale: [type: :log, domain: [1, 5000]]
  )
  |> VegaLite.encode(:y2, datum: 1)
  |> VegaLite.encode_field(:color, "type", type: :nominal, title: "Phase")
  |> VegaLite.encode_field(:x_offset, "type", type: :nominal)

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
    <p><strong>Generated:</strong> #{DateTime.utc_now() |> DateTime.add(-6 * 3600, :second) |> Calendar.strftime("%Y-%m-%d %H:%M:%S")} CT</p>
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
    <li><strong>Mamba/S5:</strong> O(L) work, O(log L) depth - parallel scan, GPU-friendly</li>
    <li><strong>RWKV/GLA/HGRN:</strong> O(L) - linear attention variants, O(1) inference memory</li>
    <li><strong>Attention:</strong> O(L²) - quadratic, but highly parallel</li>
    <li><strong>Zamba:</strong> O(L) + O(L²/N) - Mamba + shared attention (10x KV cache reduction vs Jamba)</li>
    <li><strong>Jamba:</strong> O(L) + O(L²/3) - Mamba + multiple attention layers</li>
    <li><strong>Liquid:</strong> O(L) - adaptive ODE dynamics</li>
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
