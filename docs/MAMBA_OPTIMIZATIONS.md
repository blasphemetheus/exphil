# Mamba Optimizations

Comprehensive catalog of optimization opportunities for Mamba SSM in ExPhil, from quick wins to custom CUDA kernels.

## Table of Contents

1. [Current Status](#current-status)
2. [Optimization Tiers](#optimization-tiers)
3. [Tier 1: Elixir/XLA Level](#tier-1-elixirxla-level)
4. [Tier 2: Custom XLA Operations](#tier-2-custom-xla-operations)
5. [Tier 3: Custom CUDA/Triton Kernels](#tier-3-custom-cudatriton-kernels)
6. [Tier 4: Hardware-Specific](#tier-4-hardware-specific)
7. [Memory Optimizations](#memory-optimizations)
8. [Quantization](#quantization)
9. [Learning Resources](#learning-resources)

---

## Current Status

### Implemented Variants

| Variant | File | Algorithm | Training | Inference |
|---------|------|-----------|----------|-----------|
| **GatedSSM** | `gated_ssm.ex` | Simplified gating (no recurrence) | ~35ms | ~35ms |
| **Mamba** | `mamba.ex` | Blelloch parallel scan | ~59ms | ~59ms |
| **MambaCumsum** | `mamba_cumsum.ex` | Log-space cumsum trick | TBD | TBD |

### Key Insight from Optimization Attempts

**Elixir-level loops with concrete values compile better than defn loops with tensor values.**

```elixir
# GOOD: Enum.reduce with concrete stride (compiles to efficient XLA)
Enum.reduce(0..(log_len - 1), {a, b}, fn level, {a_curr, b_curr} ->
  stride = round(:math.pow(2, level))  # Concrete integer!
  # XLA sees: slice with stride=1, then stride=2, then stride=4...
  ...
end)

# BAD: defn while with tensor stride (1000x slower!)
while {a, b, level}, level < log_len do
  stride = Nx.pow(2, level)  # Tensor! XLA can't optimize
  # XLA sees: gather with unknown indices
  ...
end
```

---

## Optimization Tiers

| Tier | Approach | Effort | Potential Speedup | Skills Needed |
|------|----------|--------|-------------------|---------------|
| 1 | Elixir/XLA tweaks | Low | 1.5-3x | Elixir, Nx |
| 2 | Custom XLA ops | Medium | 2-5x | C++, XLA |
| 3 | CUDA/Triton kernels | High | 5-20x | CUDA, GPU arch |
| 4 | Hardware-specific | Very High | 10-50x | Tensor cores, assembly |

---

## Tier 1: Elixir/XLA Level

These optimizations stay within Elixir and leverage XLA's existing primitives.

### 1.1 Log-Space Cumsum (Implemented: MambaCumsum)

**Status:** Implemented in `mamba_cumsum.ex`

**Idea:** Reformulate the SSM scan using cumulative sums instead of parallel prefix scan.

The SSM recurrence `h[t] = A[t] * h[t-1] + Bx[t]` has closed form:
```
h[t] = P[t] * cumsum(Bx / P)[t]
where P[k] = prod_{j=0}^{k-1} A[j] = exp(cumsum(log(A)))
```

**Why it might be faster:**
- XLA's `cumulative_sum` is highly optimized (fused kernel)
- Two cumsums replace O(log L) scan levels
- Better memory access patterns

**Trade-offs:**
- Numerical precision issues for very long sequences (P → 0)
- May not capture all Mamba dynamics perfectly

### 1.2 Hillis-Steele vs Blelloch Scan

**Status:** Not implemented

**Current:** Blelloch scan - O(L) work, O(log L) depth

**Alternative:** Hillis-Steele scan - O(L log L) work, O(log L) depth

```
Blelloch (work-efficient):
Level 0: [1] [2] [3] [4] [5] [6] [7] [8]
Level 1: [1] [1+2] [3] [3+4] [5] [5+6] [7] [7+8]  (stride 1, half elements)
Level 2: [1] [1+2] [3] [1-4] [5] [5+6] [7] [5-8]  (stride 2, quarter elements)

Hillis-Steele (step-efficient):
Level 0: [1] [2] [3] [4] [5] [6] [7] [8]
Level 1: [1] [1+2] [2+3] [3+4] [4+5] [5+6] [6+7] [7+8]  (stride 1, ALL elements)
Level 2: [1] [1+2] [1-3] [1-4] [2-5] [3-6] [4-7] [5-8]  (stride 2, ALL elements)
```

**Why Hillis-Steele might be faster on GPU:**
- More parallelism (all elements active every level)
- Better GPU utilization despite more total work
- Simpler memory access pattern

**Implementation:**
```elixir
defp hillis_steele_scan(a, b) do
  seq_len = Nx.axis_size(a, 1)
  log_len = ceil(:math.log2(seq_len))

  Enum.reduce(0..(log_len - 1), {a, b}, fn level, {a_curr, b_curr} ->
    stride = round(:math.pow(2, level))

    # Shift ALL elements (not just alternating like Blelloch)
    a_shifted = Nx.pad(
      Nx.slice_along_axis(a_curr, 0, seq_len - stride, axis: 1),
      1.0,
      [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
    )
    b_shifted = Nx.pad(
      Nx.slice_along_axis(b_curr, 0, seq_len - stride, axis: 1),
      0.0,
      [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
    )

    # Combine ALL pairs
    a_new = Nx.multiply(a_curr, a_shifted)
    b_new = Nx.add(Nx.multiply(a_curr, b_shifted), b_curr)

    {a_new, b_new}
  end)
end
```

### 1.3 Chunked Processing with Inter-Chunk Scan

**Status:** Not implemented

**Idea:** Process sequence in chunks, run fast parallel ops within chunks, short sequential scan between chunks.

```
Sequence: [frame_1, frame_2, ..., frame_180]
Chunk size: 60

Chunk 1: [1-60]   → parallel scan → state_60
Chunk 2: [61-120] → parallel scan (init from state_60) → state_120
Chunk 3: [121-180] → parallel scan (init from state_120) → state_180

Inter-chunk scan: 3 elements instead of 180!
```

**Benefits:**
- Memory efficiency for long sequences
- Can tune chunk size for GPU occupancy
- Foundation for SSD algorithm (Tier 2)

### 1.4 Mixed Precision (BF16/FP16)

**Status:** Partially implemented (training config supports it)

**Idea:** Use lower precision for most ops, FP32 for numerically sensitive parts.

```elixir
# Force BF16 for bulk computation
x = Nx.as_type(x, :bf16)
a = Nx.as_type(a, :bf16)

# Keep FP32 for cumsum/scan accumulation (numerical stability)
h = Nx.as_type(h, :f32)
```

**Potential speedup:** 2x on tensor cores (see Tier 4)

### 1.5 JIT Compilation Hints

**Status:** Not explored

**Idea:** Use XLA's `jit_compile` options and operation fusion hints.

```elixir
# Force specific XLA optimizations
Nx.Defn.jit(fn x -> ... end,
  compiler: EXLA,
  client: :cuda,
  run_options: [
    # XLA-specific options
    keep_on_device: true,
    lazy_transfers: true
  ]
)
```

---

## Tier 2: Custom XLA Operations

These require writing C++ code that integrates with XLA.

### 2.1 State Space Duality (SSD) - Mamba-2 Algorithm

**Status:** Not implemented

**Idea:** Decompose SSM into matrix multiplications that leverage tensor cores.

The SSD algorithm from Mamba-2:
1. **Intra-chunk matmul:** Dense matmul within chunks (tensor cores!)
2. **Chunk state computation:** Parallel per-chunk final states
3. **Inter-chunk recurrence:** Tiny sequential scan (chunk_count elements)
4. **State-to-output:** Parallel output computation

```
For L=180, chunk_size=16:
- 11 chunks
- Intra-chunk: 11 parallel matmuls of size 16x16
- Inter-chunk: scan over 11 elements (trivial)
```

**Why it's fast:**
- Converts scan → matmul (tensor cores are 10-20x faster)
- O(L) work like Blelloch but with hardware acceleration
- Used by production Mamba-2 implementations

**Implementation approach:**
1. Write as Elixir/Nx first (prototype)
2. Profile to find bottlenecks
3. Move hot paths to custom XLA op

**Resources:**
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)
- [SSD Algorithm Deep Dive](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/)

### 2.2 Custom XLA Operation via EXLA

**Status:** Not implemented

**Idea:** Write a C++ function that XLA can call, register it as a custom op.

**Steps:**
1. Write C++ implementation
2. Create XLA CustomCall target
3. Register with EXLA
4. Call from Elixir via `EXLA.Defn.custom_call/4`

**Example structure:**
```cpp
// custom_scan.cc
#include "xla/service/custom_call_target_registry.h"

void SelectiveScan(void* out, void** ins) {
  // ins[0] = x, ins[1] = A, ins[2] = B, ins[3] = C, ins[4] = dt
  float* x = static_cast<float*>(ins[0]);
  // ... implement scan ...
}

XLA_REGISTER_CUSTOM_CALL_TARGET(SelectiveScan, "Host");
```

```elixir
# Elixir side
defn selective_scan(x, a, b, c, dt) do
  EXLA.Defn.custom_call(
    "SelectiveScan",
    [x, a, b, c, dt],
    result_shape: Nx.shape(x),
    result_type: Nx.type(x)
  )
end
```

**Resources:**
- [XLA Custom Calls](https://www.tensorflow.org/xla/custom_call)
- [EXLA Source](https://github.com/elixir-nx/nx/tree/main/exla)

---

## Tier 3: Custom CUDA/Triton Kernels

Maximum performance but requires GPU programming expertise.

### 3.1 Fused Selective Scan Kernel (CUDA)

**Status:** Not implemented

**Idea:** Write a single CUDA kernel that does the entire selective scan without memory round-trips.

The official Mamba implementation uses this approach:
- [mamba/csrc/selective_scan/](https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan)

**Key optimizations in the reference:**
1. **Fused discretization:** Compute A_bar, B_bar in-kernel
2. **Shared memory scan:** Use fast shared memory for prefix sum
3. **Warp-level primitives:** `__shfl_xor_sync` for within-warp communication
4. **Register blocking:** Keep intermediate values in registers

**Kernel structure:**
```cuda
__global__ void selective_scan_fwd_kernel(
    const float* __restrict__ x,      // [B, L, D]
    const float* __restrict__ A,      // [D, N]
    const float* __restrict__ B,      // [B, L, N]
    const float* __restrict__ C,      // [B, L, N]
    const float* __restrict__ dt,     // [B, L, D]
    float* __restrict__ out,          // [B, L, D]
    int batch, int seqlen, int dim, int state_size
) {
    // Each block handles one (batch, dim) pair
    int b = blockIdx.x;
    int d = blockIdx.y;

    // Shared memory for parallel scan within block
    __shared__ float s_a[MAX_SEQLEN];
    __shared__ float s_b[MAX_SEQLEN];

    // Load and discretize
    for (int l = threadIdx.x; l < seqlen; l += blockDim.x) {
        float dt_val = dt[b * seqlen * dim + l * dim + d];
        s_a[l] = expf(dt_val * A[d * state_size + ...]);
        s_b[l] = dt_val * B[...] * x[...];
    }
    __syncthreads();

    // Parallel scan in shared memory
    // ... Blelloch or Hillis-Steele implementation ...

    // Write output
    for (int l = threadIdx.x; l < seqlen; l += blockDim.x) {
        out[...] = C[...] * s_h[l];
    }
}
```

**Integration with Elixir:**
1. Compile CUDA kernel to PTX/CUBIN
2. Wrap with custom XLA op (see Tier 2)
3. Or use NIF to call CUDA directly

### 3.2 Triton Kernel

**Status:** Not implemented

**Idea:** Use Triton (Python DSL) to write GPU kernels with less boilerplate than CUDA.

**Advantages over CUDA:**
- Higher-level abstractions
- Automatic memory coalescing
- Easier to experiment with

**Example structure:**
```python
import triton
import triton.language as tl

@triton.jit
def selective_scan_kernel(
    x_ptr, a_ptr, b_ptr, c_ptr, dt_ptr, out_ptr,
    batch, seqlen, dim, state_size,
    BLOCK_SIZE: tl.constexpr
):
    # Get position
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Load inputs
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid_batch * seqlen * dim + offs * dim + pid_dim)

    # Compute scan (Triton handles parallelization)
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for n in range(state_size):
        a_bar = tl.exp(dt * tl.load(a_ptr + pid_dim * state_size + n))
        # ... associative scan using tl.associative_scan ...

    # Store output
    tl.store(out_ptr + ..., y)
```

**Integration with Elixir:**
1. Write Triton kernel in Python
2. Compile to CUBIN
3. Load via custom XLA op or Python interop

**Resources:**
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton Scan Example](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html)

### 3.3 FlashAttention-Style Fused Kernel

**Status:** Not implemented

**Idea:** Apply FlashAttention's memory-efficient techniques to SSM scan.

FlashAttention's key innovations:
1. **Tiling:** Process in blocks that fit in SRAM
2. **Recomputation:** Recompute forward pass in backward (saves memory)
3. **Online softmax:** Never materialize full attention matrix

Applied to Mamba:
1. **Tiled scan:** Process L in tiles, carry state between tiles
2. **Fused backward:** Compute gradients in same kernel as forward
3. **Memory efficiency:** O(1) memory for hidden states

**Resources:**
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention) - Similar ideas for linear attention

---

## Tier 4: Hardware-Specific

Optimizations that target specific GPU features.

### 4.1 Tensor Core Utilization

**Status:** Partial (depends on XLA auto-optimization)

**Idea:** Structure matrix multiplications to use tensor cores (16x16 or 8x8 tiles).

**Requirements:**
- Dimensions must be multiples of 8 (FP16) or 16 (TF32)
- Data must be in specific layouts
- Use `mma` (matrix-multiply-accumulate) instructions

**For Mamba:**
- SSD algorithm naturally produces matmuls suitable for tensor cores
- Can pad state_size/hidden_size to multiples of 16

```elixir
# Ensure tensor core compatibility
hidden_size = 256  # Multiple of 16 ✓
state_size = 16    # Multiple of 16 ✓
```

### 4.2 Memory Coalescing

**Status:** XLA handles this, but can be improved

**Idea:** Ensure threads access contiguous memory locations.

**Bad (strided access):**
```
Thread 0 reads: data[0], data[100], data[200]
Thread 1 reads: data[1], data[101], data[201]
```

**Good (coalesced access):**
```
Thread 0 reads: data[0], data[1], data[2]
Thread 1 reads: data[32], data[33], data[34]
```

**For Mamba:**
- Layout tensors as [batch, seq, hidden] not [batch, hidden, seq]
- Scan along contiguous dimension

### 4.3 Warp-Level Primitives

**Status:** Not implemented (requires CUDA)

**Idea:** Use warp shuffle instructions for fast intra-warp communication.

```cuda
// Warp-level parallel scan (32 threads)
float val = input[threadIdx.x];

// Kogge-Stone parallel scan within warp
for (int offset = 1; offset < 32; offset *= 2) {
    float n = __shfl_up_sync(0xffffffff, val, offset);
    if (threadIdx.x >= offset) val += n;
}
// val now contains prefix sum for this warp
```

**Speedup:** Avoids shared memory for intra-warp operations

---

## Memory Optimizations

### Gradient Checkpointing

**Status:** Not implemented for Mamba specifically

**Idea:** Don't store all intermediate activations; recompute in backward pass.

```elixir
# Without checkpointing: store all h[0], h[1], ..., h[L-1]
# Memory: O(L * state_size)

# With checkpointing: store only h[0], h[L/2], h[L-1]
# Memory: O(sqrt(L) * state_size)
# Recompute h[1..L/2-1] and h[L/2+1..L-2] during backward
```

**Implementation in Axon:**
```elixir
# Use Axon's built-in checkpointing
Axon.layer(
  &my_scan/2,
  [input],
  name: "scan",
  op_name: :checkpoint  # Recompute during backward
)
```

### Activation Memory vs Compute Trade-off

| Approach | Memory | Compute | When to Use |
|----------|--------|---------|-------------|
| Store all | O(L) | 1x | Small L, plenty of VRAM |
| Checkpoint every √L | O(√L) | 2x | Medium L |
| Checkpoint every k | O(L/k) | 1 + L/k | Tunable |
| Full recompute | O(1) | 2x | Maximum memory savings |

---

## Quantization

### INT8 Inference

**Status:** Documented in INFERENCE.md

**Idea:** Quantize weights and activations to 8-bit integers.

```bash
# Export to ONNX with INT8 quantization
python priv/python/export_onnx.py \
  --checkpoint checkpoints/model.axon \
  --output model_int8.onnx \
  --quantize int8
```

**Speedup:** 2-4x on GPUs with INT8 tensor cores

### FP8 (Hopper GPUs)

**Status:** Not implemented

**Idea:** Use FP8 format on H100/H200 GPUs for 2x over FP16.

**Requirements:**
- CUDA 12.0+
- Hopper architecture GPU
- Careful handling of dynamic range

---

## Learning Resources

### Understanding SSM/Mamba

| Resource | Type | Level | Link |
|----------|------|-------|------|
| Mamba Paper | Paper | Advanced | [arXiv](https://arxiv.org/abs/2312.00752) |
| Visual Guide to Mamba | Blog | Beginner | [Link](https://www.maartengrootendorst.com/blog/mamba/) |
| Mamba-2 Paper | Paper | Advanced | [arXiv](https://arxiv.org/abs/2405.21060) |
| SSD Algorithm Deep Dive | Blog | Intermediate | [Link](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/) |
| Annotated S4 | Blog | Intermediate | [Link](https://srush.github.io/annotated-s4/) |

### GPU Programming

| Resource | Type | Level | Link |
|----------|------|-------|------|
| CUDA C Programming Guide | Docs | Beginner | [NVIDIA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) |
| GPU Gems 3 - Parallel Scan | Book Chapter | Intermediate | [Link](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) |
| Triton Tutorials | Tutorial | Beginner | [Link](https://triton-lang.org/main/getting-started/tutorials/) |
| CUTLASS | Library | Advanced | [GitHub](https://github.com/NVIDIA/cutlass) |

### XLA/EXLA

| Resource | Type | Level | Link |
|----------|------|-------|------|
| XLA Overview | Docs | Beginner | [TensorFlow](https://www.tensorflow.org/xla) |
| XLA Custom Calls | Docs | Advanced | [TensorFlow](https://www.tensorflow.org/xla/custom_call) |
| EXLA Source | Code | Advanced | [GitHub](https://github.com/elixir-nx/nx/tree/main/exla) |
| Nx Guides | Docs | Beginner | [HexDocs](https://hexdocs.pm/nx/Nx.html) |

### Reference Implementations

| Project | Language | What to Learn | Link |
|---------|----------|---------------|------|
| Official Mamba | Python/CUDA | Fused kernels | [GitHub](https://github.com/state-spaces/mamba) |
| Mamba.py | Python | Pure PyTorch impl | [GitHub](https://github.com/alxndrTL/mamba.py) |
| Flash Linear Attention | Python/Triton | FlashAttention for linear | [GitHub](https://github.com/sustcsonglin/flash-linear-attention) |
| Causal Conv1d CUDA | CUDA | Fused causal conv | [GitHub](https://github.com/Dao-AILab/causal-conv1d) |

---

## Implementation Roadmap

| Phase | Optimization | Effort | Expected Speedup | Status |
|-------|--------------|--------|------------------|--------|
| 0 | Baseline Blelloch scan | Done | 1x | ✅ |
| 1a | MambaCumsum variant | Low | 1.2-2x | ✅ |
| 1b | Hillis-Steele scan | Low | 1.1-1.5x | Planned |
| 1c | Mixed precision (BF16) | Low | 1.5-2x | Partial |
| 2a | Chunked + inter-chunk scan | Medium | 1.5-2x | Planned |
| 2b | SSD algorithm (Elixir) | Medium | 2-3x | Planned |
| 2c | Custom XLA op (C++) | High | 2-5x | Future |
| 3a | Triton kernel | High | 5-10x | Future |
| 3b | CUDA fused kernel | Very High | 10-20x | Future |
| 4 | Tensor core optimization | Very High | 20-50x | Future |

---

## Benchmarking Commands

```bash
# Inference benchmark (all 3 variants)
mix run scripts/benchmark_mamba_vs_gated.exs

# Training benchmark (tests L=60, L=120, L=180)
mix run scripts/benchmark_mamba_training.exs

# Full architecture comparison
mix run scripts/benchmark_architectures.exs \
  --replays /workspace/replays/mewtwo \
  --max-files 20 \
  --epochs 3

# Profile specific operations
mix run scripts/profile_mamba.exs --operation scan --seq-len 180
```

---

## Next Steps

1. **Benchmark current variants** - Run both scripts on GPU, establish baselines
2. **Implement Hillis-Steele** - Quick test if more parallelism helps
3. **Prototype SSD in Elixir** - Validate algorithm before optimizing
4. **Learn Triton** - Write a simple scan kernel as learning exercise
5. **Study official Mamba CUDA** - Understand their optimizations
