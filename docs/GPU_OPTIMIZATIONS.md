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

### 5. Precision: FP32 is Actually Faster

**Counterintuitive finding:** Benchmarks on RTX 4090 show FP32 is 2x faster than BF16.

| Precision | ms/batch | Relative |
|-----------|----------|----------|
| FP32 | 123 ms | 1.0x |
| BF16 | 256 ms | 2.1x slower |
| Mixed | 223 ms | 1.8x slower |

**Why BF16 is slower on EXLA/XLA:**
- Embedding dimensions (287) not aligned to 16 (tensor cores need alignment)
- Type casting overhead (FP32→BF16→FP32 every batch)
- XLA may fall back to FP32 kernels internally
- Non-matmul ops don't benefit from tensor cores
- See: https://github.com/openxla/xla/issues/12429

```bash
# FP32 is the default (fastest)
mix run scripts/train_from_replays.exs --temporal --backbone mamba

# BF16 uses 50% less VRAM but is 2x slower
# Only use if VRAM constrained
mix run scripts/train_from_replays.exs --precision bf16
```

**Status:** ✅ Benchmarked - FP32 is default (Jan 2026)

### 6. JIT-Wrapped Training Step (300x Speedup)

**This is the most important optimization.** Without it, training runs at 0% GPU utilization.

The problem: `Nx.Defn.value_and_grad` traces through its function argument. If that function
captures EXLA tensors in a closure, you get an EXLA/Defn.Expr conflict error. The naive fix
(copying to BinaryBackend) causes expensive CPU→GPU transfers every batch (4-5s/batch).

The solution: Wrap the entire loss+grad function with `Nx.Defn.jit`. This makes batch tensors
flow as Defn.Expr during tracing, avoiding the conflict while keeping tensors on GPU.

```elixir
# SLOW (4-5s/batch) - BinaryBackend copy
fn params, states, actions ->
  states = Nx.backend_copy(states, Nx.BinaryBackend)  # CPU→GPU every batch!
  Nx.Defn.value_and_grad(fn p -> loss(p, states, actions) end).(params)
end

# FAST (~200ms/batch) - JIT outer function
inner_fn = fn params, states, actions ->
  Nx.Defn.value_and_grad(fn p -> loss(p, states, actions) end).(params)
end
Nx.Defn.jit(inner_fn, compiler: EXLA)  # Tensors become Defn.Expr during tracing
```

**Impact:** 71s → 200ms per batch (300x speedup)

**Sources:**
- [Nx.Defn docs](https://hexdocs.pm/nx/Nx.Defn.html) - JIT compilation
- [Axon Loop source](https://github.com/elixir-nx/axon/blob/v0.7.0/lib/axon/loop.ex) - uses `defnp` for train_step
- [GitHub issue #776](https://github.com/elixir-nx/nx/issues/776) - EXLA/Defn.Expr conflict

**Status:** ✅ Implemented - automatic in all training

### 7. Gradient Accumulation

Simulate larger batch sizes without increasing memory by accumulating gradients over multiple
mini-batches before updating weights:

```
Effective batch = batch_size × accumulation_steps
Memory usage = batch_size (not effective batch)
```

```bash
# Effective batch of 2048 using 512 actual batch
mix run scripts/train_from_replays.exs \
  --batch-size 512 \
  --accumulation-steps 4
```

**When to use:**
- GPU memory limited but want larger effective batch
- Stabilize training with very small physical batch sizes
- Match published paper batch sizes on smaller GPUs

**Impact:** Same convergence as larger batch with less memory

**Sources:**
- [Gradient Accumulation in PyTorch](https://kozodoi.me/blog/20210219/gradient-accumulation)
- [Why Large Batches Train Better](https://arxiv.org/abs/1711.00489)

**Status:** ✅ Implemented - use `--accumulation-steps N`

### 9. Async Data Prefetching

Overlap data loading with GPU compute:

```
Time:     |--GPU train 1--|--GPU train 2--|--GPU train 3--|
CPU:      |--compute 1--|--compute 2--|--compute 3--|--compute 4--|
                        ↑              ↑
                     batch 2        batch 3
                     ready          ready
```

```bash
# Enabled by default with 2-buffer prefetch
mix run scripts/train_from_replays.exs --prefetch --prefetch-buffer 2

# Disable for debugging
mix run scripts/train_from_replays.exs --no-prefetch
```

**Sources:**
- [NVIDIA Data Loading Best Practices](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

**Status:** ✅ Implemented - uses streaming prefetcher with configurable buffer

### 10. Gradient Checkpointing

Trade compute for memory by recomputing activations during backward pass:

- ~60% memory reduction for activations
- ~30% slower training (recomputation overhead)
- Enables larger batch sizes or longer sequences

```bash
# Enable gradient checkpointing for Mamba backbone
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --gradient-checkpoint

# Checkpoint every other layer (less memory savings, less overhead)
mix run scripts/train_from_replays.exs \
  --gradient-checkpoint --checkpoint-every 2
```

**Status:** ✅ Implemented - supports per-layer checkpointing

## Planned Optimizations

### Flash Attention

Memory-efficient attention that computes in blocks:
- O(n) memory instead of O(n²)
- Enables longer sequences (120+ frames)
- Better cache utilization

**Status:** Planned - significant implementation effort

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
