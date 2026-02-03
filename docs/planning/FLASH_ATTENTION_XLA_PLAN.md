# FlashAttention XLA Custom Call Implementation Plan

*Zero-copy GPU integration with full Axon autodiff support*

---

## Executive Summary

This document outlines the implementation plan for integrating FlashAttention as an XLA custom call in EXLA. This is the "hard optimization" that provides:

- **Zero GPU-CPU copies** - tensors stay on device
- **Full Axon compatibility** - works with `Axon.Loop`, checkpointing, etc.
- **3x training speedup** over current NIF approach (which has copy overhead)

**Estimated effort:** 4-6 weeks
**Complexity:** High (requires C++/CUDA, XLA internals, EXLA modifications)

---

## Current State vs Target State

### Current: NIF Approach (Implemented)

```
EXLA tensor (GPU)
       ↓
Nx.to_binary() ──────────────────┐
       ↓                          │ 2 GPU-CPU
CPU binary ────► NIF ────► CUDA   │ round trips
       ↓                          │
Nx.from_binary() ←───────────────┘
       ↓
EXLA tensor (GPU)
```

**Limitations:**
- 2 GPU-CPU memory copies per call
- Cannot participate in XLA graph optimization
- Manual gradient wiring required

### Target: XLA Custom Call

```
EXLA tensor (GPU)
       ↓
stablehlo.custom_call ──► CUDA kernel (same GPU memory)
       ↓
EXLA tensor (GPU)
```

**Benefits:**
- Zero copies - tensors never leave GPU
- XLA can fuse operations around the custom call
- Works with `Nx.Defn.custom_grad` for autodiff

---

## Implementation Phases

### Phase 1: Research & Prototype (1 week)

**Goal:** Validate that GPU custom calls are possible in EXLA

#### 1.1 Study XLA FFI for GPU

Read XLA FFI documentation and examples:
- https://github.com/openxla/xla/blob/main/xla/ffi/api/ffi.h
- https://github.com/openxla/xla/tree/main/xla/service/gpu/fusions

Key questions to answer:
- [ ] How does XLA FFI register GPU handlers vs CPU handlers?
- [ ] What's the device memory access pattern?
- [ ] How are CUDA streams handled?
- [ ] Can we use cuDNN or do we need raw CUDA?

#### 1.2 Analyze EXLA Build System

Understand how EXLA compiles native code:

```
deps/exla/
├── Makefile              # Main build orchestration
├── c_src/
│   └── exla/
│       ├── exla.cc       # Main EXLA NIF
│       ├── mlir/         # MLIR/StableHLO bindings
│       └── custom_calls/ # CPU custom ops (QR, LU, Eigh)
└── lib/
    └── exla/
        └── mlir/
            └── value.ex  # Elixir bindings for custom_call
```

Key questions:
- [ ] Where does EXLA get XLA from? (prebuilt? compiled?)
- [ ] How are CPU custom calls linked?
- [ ] Is there nvcc/CUDA in the build chain already?

#### 1.3 Create Minimal GPU Custom Call

Write a trivial GPU custom call (e.g., element-wise add) to validate the approach:

```cpp
// c_src/exla/custom_calls/test_gpu_add.cu
#include "xla/ffi/api/ffi.h"

static ffi::Error gpu_add_impl(
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> b,
    ffi::Result<ffi::Buffer<ffi::F32>> out
) {
    // Launch simple CUDA kernel
    // a, b, out are already on GPU!
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(gpu_add, gpu_add_impl, ...);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "gpu_add", "CUDA", gpu_add);
//                                                        ^^^^
//                                                   GPU device!
```

**Deliverable:** Working GPU custom call that adds two tensors

---

### Phase 2: EXLA Modifications (2 weeks)

**Goal:** Add GPU custom call infrastructure to EXLA

#### 2.1 Modify EXLA Makefile for CUDA

Add CUDA compilation support:

```makefile
# deps/exla/Makefile (modified)

# Detect CUDA
CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_80  # Ampere

# CUDA sources
CUDA_SRCS := $(wildcard c_src/exla/custom_calls/*.cu)
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)

# CUDA compilation flags
CUDA_FLAGS := -O3 -arch=$(CUDA_ARCH) -Xcompiler -fPIC

# Compile .cu files
%.o: %.cu
	$(NVCC) -c -o $@ $< $(CUDA_FLAGS) -I$(XLA_INCLUDE)

# Link CUDA objects into libexla.so
$(PRIV_DIR)/libexla.so: $(OBJS) $(CUDA_OBJS)
	...
```

#### 2.2 Create GPU Custom Call Registration

```cpp
// c_src/exla/custom_calls/gpu_registry.cc

#include "xla/ffi/api/ffi.h"

// Forward declarations
extern ffi::Handler flash_attn_fwd;
extern ffi::Handler flash_attn_bwd;

// Register all GPU custom calls
void RegisterGpuCustomCalls() {
    XLA_FFI_REGISTER_HANDLER(
        ffi::GetXlaFfiApi(),
        "flash_attn_fwd",
        "CUDA",
        flash_attn_fwd
    );
    XLA_FFI_REGISTER_HANDLER(
        ffi::GetXlaFfiApi(),
        "flash_attn_bwd",
        "CUDA",
        flash_attn_bwd
    );
}
```

#### 2.3 Expose Custom Call in Elixir

Modify `deps/exla/lib/exla/mlir/value.ex`:

```elixir
defmodule EXLA.MLIR.Value do
  # ... existing code ...

  @doc """
  Call a GPU custom operation.

  ## Example

      {output, logsumexp} = gpu_custom_call(
        "flash_attn_fwd",
        [q, k, v],
        [output_type, lse_type],
        causal: true
      )
  """
  def gpu_custom_call(name, operands, result_types, opts \\ []) do
    # Build stablehlo.custom_call with backend_config
    attributes = [
      call_target_name: attr_string(name),
      api_version: attr_i32(4),  # XLA_FFI_API_VERSION
      backend_config: encode_backend_config(opts)
    ]

    op(func, "stablehlo.custom_call", operands, result_types,
       attributes: attributes)
  end

  defp encode_backend_config(opts) do
    # Encode options as backend_config for the kernel
    opts
    |> Keyword.take([:causal, :num_heads, :head_dim])
    |> :erlang.term_to_binary()
    |> Base.encode64()
  end
end
```

**Deliverable:** EXLA fork with GPU custom call infrastructure

---

### Phase 3: FlashAttention Kernels (2 weeks)

**Goal:** Implement FlashAttention forward/backward as GPU custom calls

#### 3.1 FlashAttention Forward Kernel

```cpp
// c_src/exla/custom_calls/flash_attention_fwd.cu

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

// FlashAttention-2 forward kernel
// Reference: https://github.com/Dao-AILab/flash-attention
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,  // [batch, seq, heads, dim]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,        // Output
    float* __restrict__ L,        // Logsumexp [batch, heads, seq]
    int batch, int seq_len, int num_heads, int head_dim,
    float scale, bool causal
) {
    // Tiled FlashAttention-2 implementation
    // See Algorithm 1 in the paper

    // Each thread block processes one (batch, head) pair
    int b = blockIdx.x;
    int h = blockIdx.y;

    // Shared memory for K, V tiles
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + TILE_SIZE * head_dim;

    // ... tiled attention implementation ...
}

static ffi::Error flash_attn_fwd_impl(
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> k,
    ffi::Buffer<ffi::F32> v,
    ffi::Result<ffi::Buffer<ffi::F32>> output,
    ffi::Result<ffi::Buffer<ffi::F32>> logsumexp,
    ffi::Dictionary backend_config
) {
    // Extract dimensions from buffer shapes
    auto q_dims = q.dimensions();
    int batch = q_dims[0];
    int seq_len = q_dims[1];
    int num_heads = q_dims[2];
    int head_dim = q_dims[3];

    // Parse backend_config for causal flag
    bool causal = /* decode from backend_config */;
    float scale = 1.0f / sqrtf(head_dim);

    // Get device pointers (already on GPU!)
    const float* q_ptr = q.data();
    const float* k_ptr = k.data();
    const float* v_ptr = v.data();
    float* o_ptr = output->data();
    float* l_ptr = logsumexp->data();

    // Launch kernel
    dim3 grid(batch, num_heads);
    dim3 block(256);
    size_t smem_size = 2 * TILE_SIZE * head_dim * sizeof(float);

    flash_attn_fwd_kernel<<<grid, block, smem_size>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        batch, seq_len, num_heads, head_dim,
        scale, causal
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    flash_attn_fwd,
    flash_attn_fwd_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // Q
        .Arg<ffi::Buffer<ffi::F32>>()  // K
        .Arg<ffi::Buffer<ffi::F32>>()  // V
        .Ret<ffi::Buffer<ffi::F32>>()  // Output
        .Ret<ffi::Buffer<ffi::F32>>()  // Logsumexp
        .Attr<ffi::Dictionary>("backend_config")
);
```

#### 3.2 FlashAttention Backward Kernel

```cpp
// c_src/exla/custom_calls/flash_attention_bwd.cu

// Similar structure to forward, but implements Algorithm 4
// from the FlashAttention-2 paper

__global__ void flash_attn_bwd_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ L,  // Logsumexp from forward
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int batch, int seq_len, int num_heads, int head_dim,
    float scale, bool causal
) {
    // Tiled backward implementation
    // Recomputes attention scores using saved logsumexp
}
```

#### 3.3 Elixir Wrapper with custom_grad

```elixir
defmodule ExPhil.Networks.XLAFlashAttention do
  @moduledoc """
  FlashAttention via XLA custom call - works with Axon autodiff!

  This replaces the NIF-based FlashAttention when running on GPU.
  Tensors never leave the GPU, providing ~3x speedup over NIF approach.
  """

  import Nx.Defn

  @doc """
  Flash attention with automatic gradient support.

  ## Example

      defn attention_block(q, k, v) do
        XLAFlashAttention.attention(q, k, v, causal: true)
      end

      # Gradients work automatically!
      grad_fn = Nx.Defn.grad(&attention_block/3)
  """
  defn attention(q, k, v, opts \\ []) do
    causal = opts[:causal] || true

    # Get dimensions for output types
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    # Forward pass via XLA custom call
    {output, logsumexp} = EXLA.gpu_custom_call(
      "flash_attn_fwd",
      [q, k, v],
      [
        Nx.template({batch, seq_len, num_heads, head_dim}, :f32),
        Nx.template({batch, num_heads, seq_len}, :f32)
      ],
      causal: causal
    )

    # Define custom gradient
    Nx.Defn.Kernel.custom_grad(output, [q, k, v], fn d_out ->
      {dq, dk, dv} = EXLA.gpu_custom_call(
        "flash_attn_bwd",
        [d_out, q, k, v, output, logsumexp],
        [
          Nx.template({batch, seq_len, num_heads, head_dim}, :f32),
          Nx.template({batch, seq_len, num_heads, head_dim}, :f32),
          Nx.template({batch, seq_len, num_heads, head_dim}, :f32)
        ],
        causal: causal
      )
      [dq, dk, dv]
    end)
  end
end
```

**Deliverable:** Working FlashAttention via XLA custom call with autodiff

---

### Phase 4: Testing & Optimization (1 week)

#### 4.1 Correctness Tests

```elixir
defmodule ExPhil.Networks.XLAFlashAttentionTest do
  use ExUnit.Case

  describe "forward correctness" do
    test "matches Nx reference implementation" do
      {q, k, v} = random_qkv(batch: 2, seq: 64, heads: 4, dim: 64)

      xla_out = XLAFlashAttention.attention(q, k, v)
      nx_out = Attention.scaled_dot_product_attention(q, k, v)

      assert_all_close(xla_out, nx_out, atol: 1.0e-5)
    end
  end

  describe "backward correctness" do
    test "gradients match Nx autodiff" do
      # Compare XLA custom_grad with Nx.Defn.grad
    end
  end

  describe "performance" do
    test "faster than NIF on GPU" do
      # Benchmark XLA vs NIF
    end
  end
end
```

#### 4.2 Performance Optimization

- Tune CUDA kernel block/grid dimensions
- Optimize shared memory usage
- Test different tile sizes
- Profile with NVIDIA Nsight

#### 4.3 Integration with Axon

Test with full training pipeline:

```elixir
# Should work with Axon.Loop now!
model = build_attention_model()

model
|> Axon.Loop.trainer(:categorical_cross_entropy, Polaris.Optimizers.adam())
|> Axon.Loop.run(dataset, epochs: 10)
```

---

## File Changes Summary

### New Files

| File | Description |
|------|-------------|
| `deps/exla/c_src/exla/custom_calls/flash_attention_fwd.cu` | Forward CUDA kernel |
| `deps/exla/c_src/exla/custom_calls/flash_attention_bwd.cu` | Backward CUDA kernel |
| `deps/exla/c_src/exla/custom_calls/flash_attention.h` | Shared declarations |
| `deps/exla/c_src/exla/custom_calls/gpu_registry.cc` | GPU handler registration |
| `lib/exphil/networks/xla_flash_attention.ex` | Elixir wrapper with defn |

### Modified Files

| File | Changes |
|------|---------|
| `deps/exla/Makefile` | Add CUDA compilation, link CUDA objects |
| `deps/exla/lib/exla/mlir/value.ex` | Add `gpu_custom_call/4` function |
| `deps/exla/c_src/exla/exla.cc` | Call GPU registration on init |

---

## Risks & Mitigations

### Risk 1: XLA FFI GPU Support Incomplete

**Risk:** XLA FFI may not fully support GPU custom calls the way we expect.

**Mitigation:**
- Phase 1 prototype validates this early
- Fallback: Use XLA's older `CustomCallThunk` API

### Risk 2: EXLA Build Complexity

**Risk:** Adding CUDA to EXLA's build may break existing builds.

**Mitigation:**
- Make CUDA optional (`MIX_ENV=cuda mix compile`)
- Test on multiple platforms

### Risk 3: Upstream Acceptance

**Risk:** EXLA maintainers may not accept GPU custom call PRs.

**Mitigation:**
- Engage with community on Elixir Forum first (draft exists)
- Make it fully optional/pluggable
- Maintain as a fork if needed

### Risk 4: CUDA Kernel Performance

**Risk:** Our CUDA kernels may be slower than cuDNN.

**Mitigation:**
- Start with reference implementation, optimize later
- Consider wrapping cuDNN's flash attention if available
- Benchmark against PyTorch's FlashAttention

---

## Decision Points

### Decision 1: Fork vs Upstream

**Options:**
1. **Fork EXLA** - Full control, faster iteration
2. **Upstream PR** - Community benefit, maintenance burden shared

**Recommendation:** Start with fork, upstream after validation

### Decision 2: CUDA vs cuDNN

**Options:**
1. **Raw CUDA** - Full control, more work
2. **cuDNN wrapper** - Less work, depends on cuDNN version

**Recommendation:** Start with raw CUDA (cuDNN flash attention has known issues)

### Decision 3: Scope

**Options:**
1. **FlashAttention only** - Focused, achievable
2. **General GPU custom call infra** - Broader benefit, more work

**Recommendation:** Focus on FlashAttention, but design for extensibility

---

## Success Criteria

1. **Functional:** FlashAttention works via XLA custom call
2. **Correct:** Gradients match Nx reference within 1e-5
3. **Fast:** 3x+ speedup over NIF approach on Ampere GPU
4. **Integrated:** Works with `Axon.Loop` out of the box
5. **Maintainable:** Clean separation, documented API

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Research | GPU custom call prototype working |
| 2-3 | EXLA Mods | EXLA fork with CUDA build support |
| 4-5 | Kernels | FlashAttention fwd/bwd kernels |
| 6 | Integration | Full Axon integration, tests passing |

---

## References

- [XLA FFI Documentation](https://github.com/openxla/xla/blob/main/xla/ffi/api/ffi.h)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [FlashAttention Reference Implementation](https://github.com/Dao-AILab/flash-attention)
- [EXLA Source](https://github.com/elixir-nx/nx/tree/main/exla)
- [StableHLO custom_call](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call)

---

*Last updated: 2026-01-28*
