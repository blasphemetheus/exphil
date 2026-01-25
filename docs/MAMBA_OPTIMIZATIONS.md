# Mamba Optimizations

This document covers potential optimizations for the Mamba SSM implementation in ExPhil.

## Current Status

The current Mamba implementation (`lib/exphil/networks/mamba.ex`) uses a **simplified approximation**
of the selective scan algorithm. This works but is not as accurate or efficient as the true algorithm.

### Current Implementation (Approximation)

```elixir
# Current: Simplified gated combination
y = sigmoid(dt) * (B * C * x)  # No true recurrence
```

### True Mamba Algorithm

```
h(t) = A(x) * h(t-1) + B(x) * x(t)   # State update (selective, input-dependent)
y(t) = C(x) * h(t)                    # Output projection
```

Where A, B, C, and dt (discretization step) are all **input-dependent**, computed from x(t).

## Optimization Opportunities

### 1. Parallel Associative Scan (Priority: HIGH)

**What:** Implement the true selective scan using parallel prefix sum.

**Why:** The current approximation loses the recurrent memory that makes Mamba powerful for sequences.

**How it works:**

Sequential SSM appears impossible to parallelize:
```
h₁ = A₁·h₀ + B₁·x₁
h₂ = A₂·h₁ + B₂·x₂  ← depends on h₁
h₃ = A₃·h₂ + B₃·x₃  ← depends on h₂
```

But using the **associative property**, we can rewrite as:
```
(A, B) ⊗ (A', B') = (A·A', A·B' + B)
```

This operator is associative, enabling parallel computation via tree reduction:

```
Level 0:  [h₁]   [h₂]   [h₃]   [h₄]   (compute pairs in parallel)
           ↘    ↙       ↘    ↙
Level 1:    [h₁₂]        [h₃₄]        (combine pairs)
               ↘        ↙
Level 2:         [h₁₂₃₄]              (final result)
```

**Depth:** O(log L) instead of O(L)
**Work:** O(L) total operations

**Implementation in Nx:**

```elixir
defn parallel_scan(x, a, b, c, dt) do
  # x: [batch, seq_len, hidden_size]
  # a: [batch, seq_len, state_size]  (discretized state transition)

  batch_size = Nx.axis_size(x, 0)
  seq_len = Nx.axis_size(x, 1)
  state_size = Nx.axis_size(a, 2)

  # Discretize: A_bar = exp(dt * A), B_bar = dt * B
  a_bar = Nx.exp(Nx.multiply(dt, a))
  b_bar = Nx.multiply(dt, b)

  # Initialize states
  # Each element is (A_cumulative, Bu_cumulative) tuple encoded as tensor

  # Up-sweep (reduce phase)
  states = up_sweep(a_bar, Nx.multiply(b_bar, x))

  # Down-sweep (distribution phase)
  outputs = down_sweep(states, c)

  outputs
end

defnp up_sweep(a, bu) do
  # Parallel prefix computation
  # For each level, combine adjacent pairs
  # (a₁, bu₁) ⊗ (a₂, bu₂) = (a₁·a₂, a₁·bu₂ + bu₁)

  seq_len = Nx.axis_size(a, 1)
  levels = :math.ceil(:math.log2(seq_len)) |> trunc()

  {final_a, final_bu} = while {a, bu}, i <- 0..(levels - 1) do
    stride = Nx.pow(2, i + 1) |> Nx.as_type(:s32)
    offset = Nx.pow(2, i) |> Nx.as_type(:s32)

    # Combine pairs at distance `offset`
    # New implementation needed here...
    {a, bu}
  end

  {final_a, final_bu}
end
```

**Estimated speedup:** 2-5x for sequences > 30 frames

**Sources:**
- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Section 3.3, Algorithm 2
- [Parallel Scan in CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [Visual Guide to Mamba](https://www.maartengrootendorst.com/blog/mamba/)

---

### 2. State Space Duality (SSD) Algorithm (Priority: MEDIUM)

**What:** Use Mamba-2's SSD algorithm for even faster computation.

**Why:** SSD leverages matrix multiplication (tensor cores) instead of scan operations.

**How it works:**

SSD decomposes the computation into 4 steps:
1. **Intra-chunk outputs** - Dense matmul within chunks (parallel, uses tensor cores)
2. **Chunk states** - Compute final state per chunk (parallel)
3. **Inter-chunk recurrence** - Short scan over chunk boundaries only (sequential but tiny)
4. **State-to-output** - Transform states to outputs (parallel)

```
Sequence length: 1024
Chunk size: 64
Scan length: 1024/64 = 16 (vs 1024 for naive scan)
```

**Estimated speedup:** 3-10x over parallel scan for large sequences

**Sources:**
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)
- [SSD Algorithm Explanation](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/)

---

### 3. Inference State Caching (Priority: HIGH for real-time play) ✅ IMPLEMENTED

**What:** Cache SSM hidden state between frames during inference.

**Why:** Currently, each prediction recomputes the entire sequence. For real-time play,
we only need to update the state with the new frame.

**Current (slow):**
```
Frame 1: Compute h₁ from x₁
Frame 2: Compute h₁, h₂ from x₁, x₂  (recomputes h₁!)
Frame 3: Compute h₁, h₂, h₃ from x₁, x₂, x₃  (recomputes h₁, h₂!)
```

**With caching (fast):**
```
Frame 1: Compute h₁ from x₁, cache h₁
Frame 2: Compute h₂ from h₁, x₂, cache h₂
Frame 3: Compute h₃ from h₂, x₃, cache h₃
```

**Implementation:** See `lib/exphil/networks/mamba.ex`

```elixir
# Initialize cache
cache = Mamba.init_cache(
  batch_size: 1,
  hidden_size: 256,
  num_layers: 2
)

# Single-frame inference (O(1))
{output, new_cache} = Mamba.step(frame, params, cache)
```

The Agent automatically uses incremental inference when:
- Backbone is `:mamba`
- `use_incremental: true` (default)

```elixir
# Agent automatically uses O(1) inference
{:ok, agent} = Agent.start_link(
  policy_path: "checkpoints/mamba_policy.bin",
  use_incremental: true  # default
)
```

**Speedup:** 10-60x for inference (O(1) per frame vs O(window_size))

---

### 4. Fused CUDA Kernel (Priority: LOW - requires custom XLA op)

**What:** Write a custom fused kernel for the selective scan.

**Why:** Avoid memory round-trips between operations.

**Implementation:** Would require writing a custom XLA operation in C++/CUDA and
registering it with EXLA. Significant effort but maximum performance.

**Sources:**
- [Mamba CUDA Implementation](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/)
- [EXLA Custom Operations](https://github.com/elixir-nx/nx/tree/main/exla)

---

### 5. Chunked Sequence Processing (Priority: MEDIUM)

**What:** Process very long sequences in chunks to reduce memory.

**Why:** For sequences > 120 frames, memory can become an issue.

**Implementation:**

```elixir
def process_chunked(x, chunk_size \\ 60) do
  seq_len = Nx.axis_size(x, 1)

  x
  |> chunk_sequence(chunk_size)
  |> Enum.reduce(init_state(), fn chunk, state ->
    {output, new_state} = process_chunk(chunk, state)
    {output, new_state}
  end)
  |> combine_outputs()
end
```

---

## Implementation Roadmap

| Phase | Optimization | Effort | Impact | Status |
|-------|--------------|--------|--------|--------|
| 1 | Inference state caching | Medium | 10-60x inference | ✅ Done |
| 2 | Parallel associative scan | High | 2-5x training | Planned |
| 3 | Chunked processing | Low | Memory savings | Planned |
| 4 | SSD algorithm (Mamba-2) | High | 3-10x training | Planned |
| 5 | Custom CUDA kernel | Very High | 2-3x additional | Future |

## Testing the Improvements

After implementing, benchmark with:

```bash
# Compare current vs optimized
mix run scripts/benchmark_architectures.exs \
  --replays /workspace/replays/mewtwo \
  --max-files 20 \
  --epochs 3 \
  --only mamba

# Test inference speed
mix run scripts/test_gpu_speed.exs
```

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Original Mamba architecture
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060) - SSD algorithm
- [Visual Guide to Mamba](https://www.maartengrootendorst.com/blog/mamba/) - Intuitive explanations
- [SSD Algorithm Deep Dive](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/) - Implementation details
- [Official Mamba Repo](https://github.com/state-spaces/mamba) - Reference CUDA implementation
- [Parallel Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) - GPU parallel scan
