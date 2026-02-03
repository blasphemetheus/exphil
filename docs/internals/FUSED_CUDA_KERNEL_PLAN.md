# Fused CUDA Kernel Plan for Mamba SSM

This document outlines the plan for implementing a fused CUDA kernel for the Mamba selective scan operation, enabling true training speedup (not just inference).

## Current State

| Component | Training | Inference | Notes |
|-----------|----------|-----------|-------|
| Pure Nx Mamba | ~200ms/batch | ~55ms | Gradients work via autodiff |
| MambaNIF (Rust/CUDA) | ❌ Breaks gradients | ~11ms (5x faster) | Forward-only |

**The gap:** NIF breaks gradients because `Nx.to_binary()` severs the computation graph.

## Why Fused Kernels Matter

### Training Time Breakdown
```
Forward pass:    ~40-60ms  (scan dominates)
Backward pass:   ~80-120ms (2x forward, typical for autodiff)
Optimizer step:  ~5-10ms
─────────────────────────────
Total:           ~150-200ms/batch
```

### Speedup Potential

| Optimization | Forward | Backward | Total | Training Speedup |
|--------------|---------|----------|-------|------------------|
| Current XLA | 60ms | 120ms | 200ms | 1x |
| 5x faster forward only | 12ms | 120ms | 150ms | 1.3x |
| 5x faster forward+backward | 12ms | 24ms | 50ms | 4x |
| 10x fused + custom grad | 6ms | 12ms | 30ms | 6.7x |

**Key insight:** To get real training speedup, we need fast backward pass too.

## Implementation Options

### Option 1: XLA Custom Call (Recommended)

Keep tensors on GPU throughout. No `Nx.to_binary()` needed.

**Pros:**
- Integrates with Axon/EXLA computation graph
- Gradients can flow via `Nx.Defn.custom_grad`
- No CPU-GPU transfer overhead

**Cons:**
- Requires understanding XLA internals
- C++ development

**Implementation path:**
```
1. Write C++ selective scan kernel (forward + backward)
2. Register as XLA CustomCall target
3. Expose via EXLA.Defn.custom_call/4
4. Wrap with Nx.Defn.custom_grad for autodiff
```

**Files to create:**
- `native/xla_selective_scan/selective_scan.cc` - CUDA kernel
- `native/xla_selective_scan/BUILD` - Bazel build
- `lib/exphil/native/xla_selective_scan.ex` - Elixir bindings

### Option 2: Triton Kernel + PyTorch Bridge

Write kernel in Triton (Python DSL), call via PyTorch port.

**Pros:**
- Triton is easier than raw CUDA
- Can prototype quickly
- Triton handles memory coalescing automatically

**Cons:**
- Requires Python runtime
- Added latency from port communication
- Two ML frameworks (Nx + PyTorch) is complexity

**Implementation path:**
```
1. Write Triton kernel (forward + backward)
2. Export as TorchScript or compile to CUBIN
3. Call via existing PyTorch port
4. Manual gradient accumulation (not integrated with Axon)
```

**Files:**
- `priv/triton/selective_scan.py` - Triton kernel
- `lib/exphil/bridge/triton_scan.ex` - Elixir interface

### Option 3: Pure CUDA NIF with Manual Gradients

Extend existing NIF with backward kernel.

**Pros:**
- Builds on existing infrastructure
- Maximum control

**Cons:**
- Still breaks autodiff (must manually wire gradients)
- Doesn't integrate with `Axon.Loop`
- Maintenance burden

**Implementation path:**
```
1. Add backward kernel to Rust NIF
2. Create custom training loop that:
   a. Calls NIF forward, saves activations
   b. Computes loss (in Nx)
   c. Calls NIF backward with dL/dout
   d. Manually applies gradients
```

## Recommended Approach: Option 1 (XLA Custom Call)

### Phase 1: CUDA Kernel (2-3 days)

```cpp
// native/xla_selective_scan/selective_scan_kernel.cu

// Forward kernel: parallel scan with fused discretization
__global__ void selective_scan_fwd_kernel(
    const float* x,      // [B, L, D]
    const float* dt,     // [B, L, D]
    const float* A,      // [D, N]
    const float* B,      // [B, L, N]
    const float* C,      // [B, L, N]
    float* out,          // [B, L, D]
    float* h_all,        // [B, L, D, N] - saved for backward
    int batch, int seqlen, int dim, int state_size
) {
    // Blelloch parallel scan in shared memory
    // Key optimizations:
    // - Fused exp(dt * A) discretization
    // - Shared memory for scan
    // - Warp shuffle for intra-warp
}

// Backward kernel: compute gradients
__global__ void selective_scan_bwd_kernel(
    const float* dout,   // [B, L, D] - gradient from loss
    const float* h_all,  // saved hidden states
    const float* x, const float* dt, const float* A, const float* B, const float* C,
    float* dx, float* ddt, float* dA, float* dB, float* dC,
    int batch, int seqlen, int dim, int state_size
) {
    // Reverse scan to propagate gradients
}
```

### Phase 2: XLA Integration (1-2 days)

```cpp
// native/xla_selective_scan/selective_scan_op.cc
#include "xla/service/custom_call_target_registry.h"

void SelectiveScanFwd(void* out, void** ins, const char* opaque, size_t opaque_len) {
    // Unpack inputs, launch kernel
}

void SelectiveScanBwd(void* out, void** ins, const char* opaque, size_t opaque_len) {
    // Unpack inputs, launch backward kernel
}

XLA_REGISTER_CUSTOM_CALL_TARGET(SelectiveScanFwd, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET(SelectiveScanBwd, "CUDA");
```

### Phase 3: Elixir Bindings (1 day)

```elixir
# lib/exphil/native/xla_selective_scan.ex
defmodule ExPhil.Native.XLASelectiveScan do
  import Nx.Defn

  @doc "Forward pass with saved states for backward"
  defn forward(x, dt, a, b, c) do
    # Custom call returns {output, hidden_states}
    EXLA.Defn.custom_call("SelectiveScanFwd", [x, dt, a, b, c],
      result_shape: {Nx.shape(x), {Nx.axis_size(x, 0), Nx.axis_size(x, 1), ...}},
      result_type: {:f32, :f32}
    )
  end

  @doc "Backward pass"
  defn backward(dout, h_all, x, dt, a, b, c) do
    EXLA.Defn.custom_call("SelectiveScanBwd", [dout, h_all, x, dt, a, b, c],
      result_shape: ...,
      result_type: ...
    )
  end

  @doc "Full selective scan with gradient support"
  defn scan(x, dt, a, b, c) do
    custom_grad(
      fn x, dt, a, b, c ->
        {out, h_all} = forward(x, dt, a, b, c)
        # Return output and save h_all for backward
        {out, {h_all, x, dt, a, b, c}}
      end,
      fn {h_all, x, dt, a, b, c}, dout ->
        {dx, ddt, da, db, dc} = backward(dout, h_all, x, dt, a, b, c)
        {dx, ddt, da, db, dc}
      end
    ).(x, dt, a, b, c)
  end
end
```

## Reference Implementations

Study these before implementing:

| Resource | What to Learn |
|----------|---------------|
| [mamba/csrc/selective_scan](https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan) | Official CUDA kernels |
| [mamba.py](https://github.com/alxndrTL/mamba.py) | Pure PyTorch reference |
| [XLA Custom Calls](https://www.tensorflow.org/xla/custom_call) | XLA integration |
| [EXLA source](https://github.com/elixir-nx/nx/tree/main/exla) | How to add custom ops |

## Build System

```bash
# native/xla_selective_scan/BUILD (Bazel)
cc_library(
    name = "selective_scan_kernel",
    srcs = ["selective_scan_kernel.cu"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

cc_library(
    name = "selective_scan_op",
    srcs = ["selective_scan_op.cc"],
    deps = [
        ":selective_scan_kernel",
        "@org_tensorflow//tensorflow/compiler/xla:custom_call",
    ],
)
```

## Testing Strategy

1. **Numerical correctness:** Compare outputs to pure Nx implementation
2. **Gradient check:** Finite differences vs analytical gradients
3. **Performance:** Benchmark against pure Nx and NIF
4. **Memory:** Profile GPU memory usage

```elixir
# test/exphil/native/xla_selective_scan_test.exs
defmodule ExPhil.Native.XLASelectiveScanTest do
  use ExUnit.Case

  test "forward matches pure Nx" do
    # Generate random inputs
    # Compare XLA kernel output to ExPhil.Networks.Mamba output
  end

  test "gradients are correct" do
    # Use Nx.Defn.grad to compute gradients
    # Compare to finite differences
  end

  @tag :benchmark
  test "performance improvement" do
    # Time pure Nx vs XLA kernel
    # Assert >= 3x speedup
  end
end
```

## Timeline

| Phase | Task | Effort | Status |
|-------|------|--------|--------|
| 0 | Study official Mamba CUDA | 1 day | TODO |
| 1 | CUDA kernel (fwd + bwd) | 2-3 days | TODO |
| 2 | XLA CustomCall integration | 1-2 days | TODO |
| 3 | Elixir bindings + custom_grad | 1 day | TODO |
| 4 | Testing + benchmarking | 1 day | TODO |
| 5 | Integration with Mamba.build | 0.5 day | TODO |

**Total: ~7-8 days of focused work**

## Decision Point

Before starting, consider:

1. **Is this worth it?** 4-6x training speedup is significant, but:
   - Model size is small (converges in hours anyway)
   - Embedding cache already removes the other bottleneck
   - MambaNIF already solves inference

2. **Alternatives:**
   - Use attention backbone (no custom kernel needed)
   - Accept current Mamba speed, focus on data quality
   - Wait for XLA/JAX Mamba support upstream

3. **If proceeding:** Start with Option 1 (XLA Custom Call) for proper gradient integration.
