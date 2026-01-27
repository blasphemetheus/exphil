# Training Kernel Optimizations

This document outlines GPU kernel optimizations to speed up **training** (not just inference) across all ExPhil backbones.

## Executive Summary

| Optimization | Effort | Speedup | Backbones | Status |
|--------------|--------|---------|-----------|--------|
| **BF16 mixed precision** | Low | ~1.5-2x | All | ⬜ Planned |
| **Mamba backward kernel** | Medium | ~5x | Mamba | 🚧 In Progress |
| **XLA Custom Call** | High | ~10x | Mamba | ⬜ Future |
| **Fused LayerNorm+Act** | Medium | ~1.2x | All | ⬜ Future |
| **Fused optimizer** | Low | ~1.1x | All | ⬜ Future |

---

## Current Training Performance (RTX 4090)

| Backbone | Forward | Backward | Total/Batch | Bottleneck |
|----------|---------|----------|-------------|------------|
| MLP | ~5ms | ~10ms | ~15ms | ✅ Fast |
| LSTM | ~5ms | ~15ms | ~20ms | ✅ cuDNN |
| Attention | ~0.1ms | ~0.3ms | ~0.4ms | ✅ Flash Attn |
| GRU | ~5ms | ~15ms | ~20ms | ✅ cuDNN |
| **Mamba (Nx/XLA)** | ~55ms | ~110ms | **~165ms** | ❌ Slow scan |
| Mamba (NIF forward) | ~11ms | N/A | N/A | ❌ No backward |

**Key insight:** Mamba is the only backbone that needs optimization. Others already use optimized kernels (cuDNN for RNNs, Flash Attention for transformers).

---

## 1. BF16 Mixed Precision Training

### What It Does
Uses Brain Float 16 (BF16) for forward/backward passes while keeping FP32 master weights. RTX 4090's tensor cores are ~2x faster with BF16.

### Implementation Plan

```elixir
# In training config
defstruct [
  # ... existing fields
  mixed_precision: :fp32,  # :fp32, :bf16, :fp16
  loss_scale: 1.0,         # For FP16 (BF16 doesn't need scaling)
]

# In training loop
defn train_step_bf16(params, batch, opts) do
  # Cast inputs to BF16
  states = Nx.as_type(batch.states, :bf16)

  # Forward + backward in BF16
  {loss, grads} = value_and_grad(params, fn p ->
    # Compute in BF16
    logits = forward(p, states)
    compute_loss(logits, batch.actions)
  end)

  # Accumulate gradients in FP32 for stability
  grads_fp32 = tree_map(grads, &Nx.as_type(&1, :f32))

  # Update FP32 master weights
  updated_params = apply_gradients(params, grads_fp32, opts)

  {updated_params, %{loss: loss}}
end
```

### Considerations
- **BF16 vs FP16:** BF16 has same exponent range as FP32, so no loss scaling needed
- **Gradient accumulation:** Accumulate in FP32 to avoid precision loss
- **Layer norm:** May need FP32 for stability
- **Softmax:** Should stay in FP32 for numerical stability

### CLI Flag
```bash
mix run scripts/train_from_replays.exs --mixed-precision bf16
```

### Expected Speedup
- Matrix multiplications: ~2x
- Overall training: ~1.5-1.8x (limited by non-matmul ops)

---

## 2. Mamba Backward Kernel (CUDA)

### The Problem
The forward NIF breaks autodiff because `Nx.to_binary()` severs the computation graph. We need a custom backward kernel that computes gradients directly in CUDA.

### Mathematical Background

**Forward pass:**
```
Discretize:
  A_bar[t] = exp(dt[t] * A)
  B_bar[t] = dt[t] * B[t]

Recurrence:
  h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]

Output:
  y[t] = C[t] · h[t]  (dot product over state dim)
```

**Backward pass (given dy):**
```
For the output: y[t] = C[t] · h[t]
  dC[t] = dy[t] * h[t]           (outer product)
  dh[t] += dy[t] * C[t]          (gradient flows to hidden state)

For the recurrence: h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
  dh[t-1] += A_bar[t] * dh[t]    (gradient flows backward through time)
  dA_bar[t] = dh[t] * h[t-1]
  dB_bar[t] = dh[t] * x[t]
  dx[t] = B_bar[t] * dh[t]

For discretization:
  dA += sum_t(dA_bar[t] * dt[t] * A_bar[t])
  d_dt[t] = dA_bar[t] * A * A_bar[t] + dB_bar[t] * B[t]
  dB[t] = dB_bar[t] * dt[t]
```

**Key insight:** The backward pass through the recurrence is a **reverse scan**:
```
dh[T-1] = dy[T-1] * C[T-1]
dh[T-2] = A_bar[T-1] * dh[T-1] + dy[T-2] * C[T-2]
...
```

This can also be parallelized using associative scan!

### Implementation Plan

**Step 1: Add backward kernel to Rust NIF**

```rust
// native/selective_scan_nif/src/kernel.rs

const SELECTIVE_SCAN_BACKWARD_KERNEL: &str = r#"
extern "C" __global__ void selective_scan_backward_kernel(
    const float* dy,      // [batch, seq_len, hidden]
    const float* x,       // [batch, seq_len, hidden]
    const float* h,       // [batch, seq_len, hidden, state] (saved from forward)
    const float* dt,      // [batch, seq_len, hidden]
    const float* A,       // [hidden, state]
    const float* B,       // [batch, seq_len, state]
    const float* C,       // [batch, seq_len, state]
    float* dx,            // [batch, seq_len, hidden]
    float* d_dt,          // [batch, seq_len, hidden]
    float* dA,            // [hidden, state]
    float* dB,            // [batch, seq_len, state]
    float* dC,            // [batch, seq_len, state]
    int batch, int seq_len, int hidden, int state,
    float dt_min, float dt_max
) {
    // Each block handles one (batch, hidden) pair
    int b = blockIdx.x;
    int h_idx = blockIdx.y;

    // Reverse scan for dh
    float dh[STATE_SIZE];  // Accumulator for hidden state gradient
    for (int n = 0; n < state; n++) dh[n] = 0.0f;

    // Scan backward through time
    for (int t = seq_len - 1; t >= 0; t--) {
        // Add gradient from output
        for (int n = 0; n < state; n++) {
            float c_tn = C[...];
            dh[n] += dy[...] * c_tn;
            dC[...] = dy[...] * h[...];
        }

        // Compute input gradients
        float dt_t = dt[...];
        for (int n = 0; n < state; n++) {
            float a_bar = expf(dt_t * A[...]);
            dx[...] += dh[n] * dt_t * B[...];
            // ... more gradient computations
        }

        // Propagate dh backward
        for (int n = 0; n < state; n++) {
            float a_bar = expf(dt_t * A[...]);
            dh[n] = a_bar * dh[n];  // dh[t-1] contribution
        }
    }
}
"#;
```

**Step 2: Add NIF function for backward**

```rust
// In lib.rs
#[rustler::nif]
fn selective_scan_backward(
    dy: Binary,
    x: Binary,
    h: Binary,  // Saved hidden states from forward
    dt: Binary,
    a: Binary,
    b: Binary,
    c: Binary,
    shape: (usize, usize, usize, usize),  // batch, seq_len, hidden, state
) -> Result<(Binary, Binary, Binary, Binary, Binary), Error> {
    // Returns (dx, d_dt, dA, dB, dC)
}
```

**Step 3: Wrap with custom_grad in Elixir**

```elixir
# In lib/exphil/native/selective_scan.ex
defn scan_with_grad(x, dt, a, b, c) do
  custom_grad(
    # Forward: run NIF, also save hidden states
    {y, h_saved} = scan_forward_saving_hidden(x, dt, a, b, c),

    # Backward: run backward NIF
    fn dy ->
      {dx, d_dt, dA, dB, dC} = scan_backward(dy, x, h_saved, dt, a, b, c)
      {dx, d_dt, dA, dB, dC}
    end
  )
end
```

### Memory Consideration
The backward pass needs the hidden states `h[t]` from the forward pass. Options:
1. **Save all h[t]** - Uses O(batch * seq_len * hidden * state) memory
2. **Recompute h[t]** - Run forward scan again during backward (trades compute for memory)
3. **Checkpointing** - Save every Nth hidden state, recompute others

### Expected Speedup
- Forward: ~11ms (already have)
- Backward: ~15-20ms (reverse scan, similar complexity)
- Total: ~25-30ms vs ~165ms = **~5-6x speedup**

---

## 3. XLA Custom Call (Future)

### What It Does
Registers our CUDA kernel directly with XLA, allowing:
- Zero GPU↔CPU data transfer
- XLA handles gradient computation automatically
- Can fuse with adjacent operations

### Why It's Better Than NIF
```
NIF approach:
  EXLA tensor → CPU copy → NIF → CUDA → CPU copy → EXLA tensor

XLA Custom Call:
  XLA tensor (GPU) → Custom kernel (GPU) → XLA tensor (GPU)
```

### Implementation Sketch
```cpp
// Register with XLA
XLA_REGISTER_CUSTOM_CALL_TARGET(SelectiveScanForward, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET(SelectiveScanBackward, "CUDA");

// In Elixir
defn selective_scan_xla(x, dt, a, b, c) do
  EXLA.Defn.custom_call(
    "SelectiveScanForward",
    [x, dt, a, b, c],
    result_shape: Nx.shape(x)
  )
end
```

### Blockers
- Requires changes to EXLA or XLA FFI support
- More complex build setup (XLA headers)
- Best tackled after BF16 and backward kernel

---

## 4. Fused LayerNorm + Activation

### What It Does
Combines LayerNorm and activation (SiLU, GELU) into a single kernel to reduce memory bandwidth.

### Current (Unfused)
```
x → [LayerNorm kernel] → normalized → [SiLU kernel] → activated
     ↑ read x, write normalized      ↑ read normalized, write activated
     (memory bandwidth bottleneck)
```

### Fused
```
x → [Fused LayerNorm+SiLU kernel] → activated
     ↑ read x once, write activated once
```

### Implementation
Could use Triton or custom CUDA:

```python
# Triton fused kernel
@triton.jit
def fused_layernorm_silu(x_ptr, out_ptr, ...):
    # Load x
    x = tl.load(x_ptr + offsets)

    # LayerNorm
    mean = tl.sum(x) / n
    var = tl.sum((x - mean) ** 2) / n
    normalized = (x - mean) / tl.sqrt(var + eps)

    # SiLU
    activated = normalized * tl.sigmoid(normalized)

    # Store once
    tl.store(out_ptr + offsets, activated)
```

### Expected Speedup
~10-20% reduction in memory-bound operations.

---

## 5. Fused Optimizer Step

### What It Does
Combines gradient application with weight update in a single kernel.

### Current
```
for each parameter:
  grad = gradients[param]           # read
  momentum = momentums[param]       # read
  new_momentum = beta * momentum + grad  # compute
  param = param - lr * new_momentum     # compute
  momentums[param] = new_momentum   # write
  parameters[param] = param         # write
```

### Fused
```
fused_adam_kernel(params, grads, momentums, velocities, lr, beta1, beta2)
  # Single kernel: read params+grads+state, write params+state
```

### Implementation
EXLA may already do some fusion. Custom kernel would help for large models.

---

## Implementation Roadmap

### Phase 1: Mamba Backward Kernel (Current)
1. ✅ Document the math
2. ✅ Implement CUDA backward kernel (`kernel.rs`)
3. ✅ Add NIF bindings (`lib.rs`)
4. ✅ Add Elixir wrapper (`selective_scan.ex`)
5. ⬜ Wrap with `Nx.Defn.custom_grad` for automatic differentiation
6. ⬜ Benchmark training speed

### Phase 2: BF16 Mixed Precision
1. ⬜ Add `--mixed-precision` flag
2. ⬜ Implement BF16 casting in training loop
3. ⬜ Handle numerical stability (softmax, layernorm)
4. ⬜ Benchmark speedup across backbones

### Phase 3: Fused Operations
1. ⬜ Profile to identify biggest bottlenecks
2. ⬜ Implement fused LayerNorm+Activation
3. ⬜ Implement fused optimizer (if beneficial)

### Phase 4: XLA Custom Call
1. ⬜ Wait for EXLA FFI support or contribute it
2. ⬜ Port kernels to XLA CustomCall interface
3. ⬜ Remove GPU↔CPU transfers entirely

---

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Section 3.3 discusses efficient scan
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Fused attention kernels
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - Original paper
- [BF16 Training](https://cloud.google.com/tpu/docs/bfloat16) - Google's BF16 guide
