# EXLA GPU Custom Calls Implementation

*Progress on XLA custom call integration for FlashAttention*

---

## Overview

This document tracks our work on adding GPU custom call support to EXLA,
enabling native CUDA kernel integration for FlashAttention and other
GPU-accelerated operations.

The EXLA fork is located at: `/home/dori/git/melee/nx/exla`

---

## Completed Work

### 1. EXLA Build System Analysis

**Key Findings:**
- XLA is obtained as prebuilt archive via `xla` package
- Extracted to `cache/xla_extension` with headers and libs
- Makefile already detects `nvcc` and enables CUDA compilation
- Custom calls in `c_src/exla/custom_calls/*.cc` are auto-compiled

### 2. Makefile Modification

**File:** `Makefile`

Added support for `.cu` files in custom_calls:

```makefile
# CUDA sources in custom_calls (compiled with nvcc when available)
CUDA_SOURCES = $(wildcard $(EXLA_DIR)/custom_calls/*.cu)
CUDA_OBJECTS = $(patsubst $(EXLA_DIR)/%.cu,$(EXLA_CACHE_OBJ_DIR)/%.o,$(CUDA_SOURCES))

# Compile .cu files in custom_calls with nvcc
$(EXLA_CACHE_OBJ_DIR)/custom_calls/%.o: $(EXLA_DIR)/custom_calls/%.cu
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)/custom_calls
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
```

### 3. GPU Custom Call Prototype

**File:** `c_src/exla/custom_calls/gpu_add.cu`

Simple CUDA kernel to validate the approach:

```cpp
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

// CUDA kernel
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n);

// FFI handler
static ffi::Error gpu_add_impl(
    cudaStream_t stream,  // From PlatformStream binding
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> b,
    ffi::Result<ffi::Buffer<ffi::F32>> out
);

// Registration
XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(),
    "exla_gpu_add_f32",
    "CUDA",  // Platform - routes to GPU device
    gpu_add
);
#endif
```

### 4. Elixir Value Binding

**File:** `lib/exla/mlir/value.ex`

Added `gpu_add/3` function:

```elixir
def gpu_add(%Value{function: func} = a, %Value{function: func} = b, out_typespec) do
  operands = [a, b]
  result_types = typespecs_to_mlir_types([out_typespec])

  attributes = [
    call_target_name: attr_string("exla_gpu_add_f32"),
    api_version: attr_i32(4)
  ]

  op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
end
```

### 5. Documentation Module

**File:** `lib/exla/gpu_custom_call.ex`

Module documenting the GPU custom call architecture and next steps.

---

## Architecture

GPU custom calls in EXLA work through three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Nx.Defn Code                                 │
│  defn attention(q, k, v) do ... end                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXLA.Defn Compiler                              │
│  cached_recur_operator matches on platform: :cuda               │
│  Routes to Value.flash_attention(...)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXLA.MLIR.Value                                 │
│  Builds stablehlo.custom_call with:                             │
│    - call_target_name: "exla_flash_attn_fwd"                    │
│    - api_version: 4 (typed FFI)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  XLA Runtime                                     │
│  Routes to handler registered with platform "CUDA"              │
│  Provides cudaStream_t for kernel execution                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CUDA Kernel                                     │
│  flash_attention_fwd.cu                                         │
│  Tensors already on GPU (zero-copy!)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## XLA FFI Key Patterns

### GPU Handler Registration

```cpp
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    handler_name,
    impl_function,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // Get CUDA stream
        .Arg<ffi::Buffer<ffi::F32>>()               // Input buffers
        .Ret<ffi::Buffer<ffi::F32>>()               // Output buffers
);

XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(),
    "handler_name",
    "CUDA",  // Platform: "Host" for CPU, "CUDA" for GPU
    handler_name
);
```

### Handler Implementation

```cpp
static ffi::Error impl_function(
    cudaStream_t stream,          // CUDA stream for kernel launch
    ffi::Buffer<ffi::F32> input,  // Device pointer (already on GPU!)
    ffi::Result<ffi::Buffer<ffi::F32>> output
) {
    const float* in_ptr = input.typed_data();   // GPU memory
    float* out_ptr = output->typed_data();      // GPU memory

    // Launch kernel on provided stream
    my_kernel<<<grid, block, 0, stream>>>(in_ptr, out_ptr, n);

    return ffi::Error::Success();
}
```

---

## Next Steps

### Phase 1: Validate Prototype

1. Test `gpu_add.cu` on RunPod with CUDA
2. Verify XLA routes to CUDA handler correctly
3. Benchmark to confirm zero-copy performance

### Phase 2: FlashAttention Forward

1. Create `flash_attention_fwd.cu`:
   - Tiled attention algorithm
   - Save logsumexp for backward pass
   - Register as `exla_flash_attn_fwd`

2. Add `Value.flash_attention_forward/4`:
   - Returns `{output, logsumexp}`
   - Uses custom_call with appropriate typespecs

### Phase 3: FlashAttention Backward

1. Create `flash_attention_bwd.cu`:
   - Recomputes attention using logsumexp
   - Returns `{dq, dk, dv}`

2. Add `Value.flash_attention_backward/7`

### Phase 4: Integration

1. Add pattern matching in `EXLA.Defn` for attention operations
2. Implement `Nx.Defn.Kernel.custom_grad` wrapper
3. Test with Axon training loop

---

## Testing Commands

```bash
# On machine with CUDA
cd ~/git/melee/nx/exla

# Ensure CUDA is detected
which nvcc

# Force rebuild with CUDA
EXLA_FORCE_REBUILD=true mix compile

# Test GPU custom call
XLA_TARGET=cuda iex -S mix
iex> EXLA.GPUCustomCall.status()
```

---

## References

- [XLA Custom Calls](https://openxla.org/xla/custom_call)
- [JAX FFI Documentation](https://docs.jax.dev/en/latest/ffi.html)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [EXLA Source](https://github.com/elixir-nx/nx/tree/main/exla)

---

*Last updated: 2026-01-28*
