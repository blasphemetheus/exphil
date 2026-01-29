# RAM Optimization Plan for Sequence Embeddings

## Problem

Temporal architectures (LSTM, GRU, Mamba, Attention, Jamba) require **13+ GB RAM** for sequence embeddings:

```
380K sequences × 30 frames × 287 dims × 4 bytes = 13.1 GB
```

This causes OOM on machines with <16 GB RAM.

## Current Architecture

```
Frame Embeddings (140 MB)          Sequence Embeddings (13 GB)
┌─────────────────────────┐        ┌──────────────────────────────┐
│ Tensor: [380K, 287]     │   →    │ Array of 380K tensors        │
│ - Stacked, efficient    │        │ - Each: [30, 287]            │
│ - GPU-friendly          │        │ - Erlang array overhead      │
└─────────────────────────┘        │ - Built upfront              │
                                   └──────────────────────────────┘
```

The `sequences_from_frame_embeddings` function pre-builds ALL sequences by slicing from frame embeddings. This is 30x faster than re-embedding, but still allocates 13 GB.

## Proposed Solution: Lazy Sequence Slicing

**Keep frame embeddings, build sequences on-demand during batching.**

```
Frame Embeddings (140 MB)          Batch Time
┌─────────────────────────┐        ┌──────────────────────────────┐
│ Tensor: [380K, 287]     │   →    │ Slice 256 windows per batch  │
│ - Stays in RAM          │        │ - [256, 30, 287] = 8.8 MB    │
│ - Or on GPU             │        │ - Fast Nx.slice operation    │
└─────────────────────────┘        └──────────────────────────────┘

RAM: 140 MB (frame embeddings) + 8.8 MB (current batch) = ~150 MB
```

### Implementation

1. **New batching mode in `Data.batched_sequences/3`:**

```elixir
def batched_sequences(dataset, opts) do
  batch_size = opts[:batch_size] || 64
  window_size = opts[:window_size] || 30
  stride = opts[:stride] || 1
  lazy = opts[:lazy] || false  # NEW FLAG

  if lazy and dataset.embedded_frames do
    # Lazy mode: slice from frame embeddings on-the-fly
    batched_sequences_lazy(dataset, batch_size, window_size, stride)
  else
    # Eager mode: use pre-built sequence embeddings (current behavior)
    batched_sequences_eager(dataset, batch_size)
  end
end

defp batched_sequences_lazy(dataset, batch_size, window_size, stride) do
  frame_embeddings = dataset.embedded_frames
  num_frames = Nx.axis_size(frame_embeddings, 0)
  num_sequences = div(num_frames - window_size, stride) + 1

  indices = Enum.shuffle(0..(num_sequences - 1))

  indices
  |> Enum.chunk_every(batch_size)
  |> Stream.map(fn batch_indices ->
    # Slice windows from frame embeddings
    sequences = Enum.map(batch_indices, fn seq_idx ->
      frame_start = seq_idx * stride
      Nx.slice(frame_embeddings, [frame_start, 0], [window_size, Nx.axis_size(frame_embeddings, 1)])
    end)

    # Stack into batch tensor
    states = Nx.stack(sequences)
    actions = get_actions_for_indices(dataset, batch_indices, window_size, stride)

    %{states: states, actions: actions}
  end)
end
```

2. **Benchmark script flag:**

```bash
# Current (high RAM, fast batching)
mix run scripts/benchmark_architectures.exs --eager-sequences

# New default (low RAM, slightly slower batching)
mix run scripts/benchmark_architectures.exs --lazy-sequences
```

3. **GPU-accelerated slicing (optional optimization):**

```elixir
# Keep frame embeddings on GPU, slice there
frame_embeddings_gpu = Nx.backend_transfer(frame_embeddings, EXLA.Backend)

# Batch gather is faster than individual slices
indices_tensor = Nx.tensor(batch_indices)
# Use Nx.take with computed indices for each window position
```

## Performance Analysis

| Mode | RAM Usage | Batch Overhead | Training Speed |
|------|-----------|----------------|----------------|
| Eager (current) | 13 GB | ~0ms | 7.7 b/s |
| Lazy (CPU slicing) | 150 MB | ~5-10ms | ~6.5 b/s (est.) |
| Lazy (GPU slicing) | 150 MB | ~1-2ms | ~7.3 b/s (est.) |

**Expected slowdown: 5-15%** for lazy mode, but enables training on 8 GB RAM machines.

## Implementation Plan

### Phase 1: Core Implementation (2-3 hours)

1. Add `batched_sequences_lazy/4` to `lib/exphil/training/data.ex`
2. Add `lazy: true` option to `batched_sequences/3`
3. Add `--lazy-sequences` / `--eager-sequences` flags to benchmark script
4. Test with small dataset

### Phase 2: Optimization (1-2 hours)

1. GPU-accelerated batch slicing (keep frame embeddings on GPU)
2. Prefetch next batch while training current batch
3. Benchmark lazy vs eager to measure actual overhead

### Phase 3: Integration (1 hour)

1. Make lazy default for benchmark script
2. Update `train_from_replays.exs` to support lazy mode
3. Add auto-detection: use lazy if RAM < 16 GB

## Alternative Approaches Considered

### Memory-Mapped Files
- Pro: Zero RAM usage
- Con: Disk I/O latency, complex serialization
- Verdict: Overkill for this use case

### Chunked Processing
- Pro: Bounded RAM
- Con: Can't shuffle across full dataset
- Verdict: Hurts training quality

### Compressed Sequences
- Pro: 2-4x RAM reduction with BF16/FP16
- Con: Precision loss, still 3-6 GB
- Verdict: Good complement to lazy mode

## Decision

**Recommended: Implement lazy sequence slicing (Phase 1-2)**

- 100x RAM reduction (13 GB → 150 MB)
- Minimal speed impact with GPU slicing (~5%)
- Enables training on consumer hardware
- Backwards compatible (eager mode still available)
