# GPU Optimizations

This document covers GPU-specific optimizations for ExPhil training and inference.

## Current GPU Performance (RTX 4090)

| Operation | Time | Notes |
|-----------|------|-------|
| Policy inference (single) | 1.6ms | 156x faster than CPU |
| Policy inference (batch 32) | 1.9ms | 131x faster than CPU |
| Mamba temporal (30 frames) | 40ms | Impossible on CPU |
| Embedding (player) | 1.9ms | GPU overhead > CPU |
| Embedding (game state) | 4.5ms | GPU overhead > CPU |

## Implemented Optimizations

### 1. XLA Compilation Caching

XLA JIT-compiles operations on first use. Caching avoids recompilation on restart.

```elixir
# config/runtime.exs
config :exla, :cache_path, System.get_env("XLA_CACHE_PATH", ".xla_cache")
```

**Impact:** Faster startup after first run (~30s saved)

### 2. GPU-Optimized Batch Sizes

GPU presets use larger batch sizes to better utilize parallelism.

| Preset | Batch Size | VRAM Usage |
|--------|------------|------------|
| `gpu_quick` | 256 | ~4GB |
| `gpu_standard` | 256 | ~6GB |
| `production` | 256 | ~8GB |

RTX 4090 (24GB) can handle batch sizes up to 512 for single-frame, 256 for temporal.

### 3. Tensor Dimension Alignment

Tensor core efficiency requires dimensions aligned to 8 (FP16) or 16 (INT8).

| Parameter | Old | Aligned | Reason |
|-----------|-----|---------|--------|
| `hidden_size` | 256 | 256 | Already aligned |
| `embed_size` | varies | padded | Pad to multiple of 8 |
| `num_heads` | 4/8 | 8 | Multiple of 8 |

### 4. GPU Memory Monitoring

Track VRAM usage during training to avoid OOM:

```elixir
ExPhil.Training.GPUUtils.memory_status()
# => %{used_mb: 4521, total_mb: 24564, utilization: 0.18}
```

## Planned Optimizations

### Mixed Precision Training (BF16)

RTX 4090 has excellent BF16 tensor cores. Benefits:
- ~2x throughput for matrix operations
- ~50% memory reduction
- Minimal accuracy loss for neural networks

```elixir
# Future API
config :exphil, :training_dtype, :bf16
config :exphil, :accumulation_dtype, :f32  # Keep gradients in FP32
```

**Status:** Planned - requires careful handling of gradient scaling

### Flash Attention

Memory-efficient attention that computes in blocks:
- O(n) memory instead of O(n²)
- Enables longer sequences (120+ frames)
- Better cache utilization

**Status:** Planned - significant implementation effort

### Async Data Prefetching

Overlap data loading with GPU compute:

```elixir
# Pipeline: load batch N+1 while GPU processes batch N
defmodule ExPhil.Training.AsyncLoader do
  def stream(dataset, batch_size) do
    dataset
    |> Stream.chunk_every(batch_size)
    |> Stream.transform(nil, fn batch, prefetched ->
      next = Task.async(fn -> prepare_batch(batch) end)
      result = if prefetched, do: Task.await(prefetched), else: prepare_batch(batch)
      {[result], next}
    end)
  end
end
```

**Status:** Planned - moderate implementation effort

### Multi-GPU Training

Distribute training across multiple GPUs using data parallelism:

```elixir
# Future API
config :exphil, :devices, [:cuda0, :cuda1]
config :exphil, :distribution, :data_parallel
```

**Status:** Future - requires significant infrastructure

### Quantization-Aware Training

Train with INT8 quantization noise for better inference quantization:

```elixir
def fake_quantize(tensor, bits \\ 8) do
  scale = Nx.reduce_max(Nx.abs(tensor)) / (2 ** (bits - 1) - 1)
  Nx.round(tensor / scale) * scale
end
```

**Status:** Future - useful for ONNX INT8 export

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXLA_TARGET` | `cuda` | EXLA backend target |
| `XLA_TARGET` | `cuda12` | XLA platform |
| `XLA_CACHE_PATH` | `.xla_cache` | XLA compilation cache |
| `CUDA_VISIBLE_DEVICES` | `0` | Which GPU to use |
| `EXPHIL_GPU` | auto | Force GPU (`1`) or CPU (`0`) in tests |

### Recommended Settings by GPU

| GPU | VRAM | Batch Size | Window Size | Notes |
|-----|------|------------|-------------|-------|
| RTX 3090 | 24GB | 256 | 60 | Good value |
| RTX 4090 | 24GB | 256-512 | 90 | Best single-GPU option |
| RTX 5090 | 32GB | 512 | 120 | Requires CUDA 12.6+ |
| A100 40GB | 40GB | 512 | 120 | Overkill for current model |
| A100 80GB | 80GB | 1024 | 180 | Way overkill |

## Profiling GPU Performance

### Using nvidia-smi

```bash
# Watch GPU utilization during training
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > gpu_log.csv
```

### Using EXLA Profiling

```elixir
# Enable XLA profiling
EXLA.Client.default_name()
|> EXLA.Client.get_client()
|> EXLA.Client.profile(fn ->
  # Your computation here
  Axon.predict(model, params, input)
end)
```

## Troubleshooting

### OOM (Out of Memory)

1. Reduce batch size
2. Reduce window size for temporal models
3. Enable gradient checkpointing (future)
4. Use mixed precision (future)

### Slow First Iteration

XLA JIT compilation. Solutions:
1. Enable XLA cache (`XLA_CACHE_PATH`)
2. Use warmup iterations in benchmarks
3. Pre-compile common shapes at startup

### GPU Not Being Used

Check:
```bash
# Verify CUDA is available
nvidia-smi

# Check EXLA target
echo $EXLA_TARGET  # Should be "cuda"

# Check Nx backend
MIX_ENV=test mix run -e 'IO.inspect(Nx.default_backend())'
```

### Tensor Core Not Utilized

Ensure dimensions are multiples of 8:
```elixir
# Bad
hidden_size: 250

# Good
hidden_size: 256
```
