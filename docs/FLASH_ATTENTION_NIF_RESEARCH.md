# FlashAttention NIF Integration Research

This document summarizes research into integrating FlashAttention-2 via a Rust NIF for ExPhil.

## Executive Summary

**Recommendation:** Start with **Option B (XLA FFI)** if EXLA adds support, otherwise **Option C (Simplified NIF)** for forward-only inference. Full NIF with backward pass (Option A) is high effort and likely unnecessary given our use case.

| Approach | Effort | Training | Inference | Recommended |
|----------|--------|----------|-----------|-------------|
| A. Full NIF (forward + backward) | 4-6 weeks | Yes | Yes | No |
| B. XLA FFI / cuDNN | 1 week* | Yes | Yes | **Yes** (if available) |
| C. Simplified NIF (forward only) | 1-2 weeks | No | Yes | **Yes** (fallback) |
| D. Python Bridge | Done | Limited | Yes | Prototype only |

*Depends on EXLA team adding support

---

## Option A: Full Rustler NIF with FlashAttention-2

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Elixir/Nx                               │
│  query = Nx.tensor(...)                                      │
│  output = ExPhil.Native.FlashAttention.forward(q, k, v)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ NIF call
┌──────────────────────────────▼──────────────────────────────┐
│                    Rustler NIF (Rust)                        │
│  - Tensor validation                                         │
│  - Memory layout conversion                                  │
│  - CUDA stream management                                    │
└──────────────────────────────┬──────────────────────────────┘
                               │ FFI
┌──────────────────────────────▼──────────────────────────────┐
│                 FlashAttention-2 CUDA Kernels                │
│  - flash_api.cu                                              │
│  - fmha_fwd_hdim{32,64,128}.cu                              │
│  - fmha_bwd_hdim{32,64,128}.cu (for training)               │
└─────────────────────────────────────────────────────────────┘
```

### Dependencies

Based on [candle-flash-attn](https://github.com/huggingface/candle-flash-attn-v1):

1. **CUDA Toolkit 12.0+** - nvcc compiler, runtime
2. **CUTLASS** - NVIDIA's header-only CUDA templates
3. **Rustler** - Safe Rust NIF bindings for Erlang
4. **cuBLAS/cuDNN** - Optional, for memory allocation

### Build Process

From [candle-flash-attn build.rs](https://github.com/huggingface/candle-flash-attn-v1/blob/main/build.rs):

```rust
// Simplified build process
fn main() {
    // 1. Detect GPU compute capability
    let compute_cap = detect_compute_cap(); // e.g., "sm_86" for RTX 3090

    // 2. Compile CUDA kernels to object files
    for kernel in ["flash_api.cu", "fmha_fwd_hdim64.cu", ...] {
        nvcc::compile(kernel, &[
            "-O3",
            "-std=c++17",
            &format!("-arch={}", compute_cap),
            "-Icutlass/include",
        ]);
    }

    // 3. Create static library
    nvcc::create_lib("libflashattention.a", &object_files);

    // 4. Link with Rust
    println!("cargo:rustc-link-lib=static=flashattention");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
```

### Tensor Memory Transfer

Nx provides `to_pointer` and `from_pointer` for zero-copy data transfer:

```elixir
# Elixir side
defmodule ExPhil.Native.FlashAttention do
  use Rustler, otp_app: :exphil, crate: "exphil_flash_attention"

  def forward(query, key, value, opts \\ []) do
    # Get raw pointers (zero-copy if already on GPU)
    {q_ptr, q_shape} = Nx.to_pointer(query)
    {k_ptr, k_shape} = Nx.to_pointer(key)
    {v_ptr, v_shape} = Nx.to_pointer(value)

    # Call NIF
    result_ptr = nif_forward(q_ptr, k_ptr, v_ptr, q_shape, opts)

    # Wrap result back in Nx tensor
    Nx.from_pointer(result_ptr, query.type, q_shape)
  end

  defp nif_forward(_q, _k, _v, _shape, _opts), do: :erlang.nif_error(:not_loaded)
end
```

```rust
// Rust NIF side
use rustler::{Env, Term, NifResult};

#[rustler::nif]
fn nif_forward(
    env: Env,
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    shape: Vec<i64>,
    opts: Term,
) -> NifResult<u64> {
    // Cast pointers to CUDA device pointers
    let q = q_ptr as *const f16;
    let k = k_ptr as *const f16;
    let v = v_ptr as *const f16;

    // Allocate output
    let output = cuda_malloc(shape.iter().product() * 2); // f16

    // Call FlashAttention kernel
    unsafe {
        flash_attention_forward(
            q, k, v, output,
            shape[0], shape[1], shape[2], // batch, seq, dim
            opts.causal,
        );
    }

    Ok(output as u64)
}
```

### Challenges

1. **Backward Pass Complexity**: FlashAttention backward is ~3x the code of forward
2. **Gradient Integration**: Must integrate with Axon's autodiff
3. **Memory Management**: CUDA memory lifecycle across NIF boundary
4. **Build Complexity**: CUDA compilation is slow (~10 min), needs caching
5. **Platform Lock-in**: NVIDIA-only, requires specific GPU generations

### Effort Estimate: 4-6 weeks

---

## Option B: XLA FFI / cuDNN Flash Attention

### Background

JAX provides built-in flash attention via cuDNN:

```python
# JAX API
jax.nn.dot_product_attention(q, k, v, implementation="cudnn")
```

This is exposed through XLA's [Foreign Function Interface (FFI)](https://openxla.org/xla/custom_call).

### XLA FFI Architecture

```cpp
// C++ FFI handler registration
#include "xla/ffi/ffi.h"

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlashAttentionHandler,
    FlashAttentionImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::Buffer<F16>>("query")
        .Arg<xla::ffi::Buffer<F16>>("key")
        .Arg<xla::ffi::Buffer<F16>>("value")
        .Ret<xla::ffi::Buffer<F16>>("output")
        .Attr<bool>("causal")
);
```

### EXLA Integration Path

If EXLA adds cuDNN flash attention support:

```elixir
# Hypothetical future API
defn flash_attention(query, key, value, opts \\ []) do
  EXLA.Backend.fused_attention(query, key, value,
    scale: Nx.rsqrt(Nx.axis_size(query, -1)),
    is_causal: opts[:causal] || true
  )
end
```

### Current Status (Jan 2026)

- **JAX**: Has `jax.nn.dot_product_attention` with `implementation="cudnn"`
- **XLA**: Supports cuDNN FMHA via `--xla_gpu_enable_cudnn_fmha=true`
- **EXLA**: No direct exposure yet (see Nx Ecosystem Status below)

### JAX cuDNN Flash Attention Known Issues (2025)

JAX's cuDNN implementation has several reported problems:

| Issue | Description | Workaround |
|-------|-------------|------------|
| [#25986](https://github.com/jax-ml/jax/issues/25986) | Tensor stride errors during JIT | Use `implementation="xla"` |
| [#29605](https://github.com/jax-ml/jax/issues/29605) | Incompatible with `jax.vmap` | Avoid vmap with cudnn |
| [#30593](https://github.com/jax-ml/jax/issues/30593) | Bias gradient only works batch=1 | Use batch=1 or xla |
| [#32430](https://github.com/jax-ml/jax/issues/32430) | Fails on CUDA 13 | Use CUDA 12 |
| [#27599](https://github.com/jax-ml/jax/issues/27599) | Multi-GPU graph capture fails | Use single GPU |

These issues suggest cuDNN FMHA is still maturing. The Nx team may be wise to wait for stability.

### Nx Ecosystem Status

**Issue [#1461](https://github.com/elixir-nx/nx/issues/1461) (closed March 2024)** explored "Special node acceleration via metadata" for custom operations like flash attention.

Key quotes from the discussion:

> "Axon marks each layer with `:op` as `:metadata`, which means that a compiler could replace operations with a better one depending on what it is." - @seanmor5

> "I struggle to see how it would work in practice. If EXLA has to match on the metadata, then it requires EXLA itself to know about the operation... we would need to make EXLA itself extensible. Custom calls are one mechanism to achieve this." - @josevalim

> "Let's experiment a bit... so we can start exploring this domain" - @polvalente

The issue was closed as experimental/exploratory rather than actionable.

### Advantages

- Automatic differentiation works
- Maintained by Google/NVIDIA
- No custom CUDA compilation
- Works with existing EXLA tensors

### Effort Estimate: 1 week (once EXLA supports it)

**Action Items**:
1. Monitor [elixir-nx/nx](https://github.com/elixir-nx/nx) for cuDNN attention support
2. Consider opening feature request for real-time inference use case
3. Explore EXLA's `c_src/exla/custom_calls/` for contribution path

### EXLA Custom Calls Analysis (Jan 2026)

Explored `exla/c_src/exla/custom_calls/` to understand the contribution pattern:

**CPU Custom Calls (QR, LU, Eigh):**
```cpp
// Pattern: Type-specific wrappers around templated implementation
XLA_FFI_DEFINE_HANDLER_SYMBOL(qr_cpu_custom_call_f32,
                              qr_cpu_custom_call_f32_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_f32", "Host",
                         qr_cpu_custom_call_f32);
```

**Runtime Callback Bridge:**
- Allows calling Elixir functions from XLA computations
- Used for operations that need dynamic behavior

**GPU Custom Calls:**
- **Not currently supported** - no `.cu` files in custom_calls
- GPU operations go through XLA's standard compilation path
- EXLA's `exla_cuda.cc` only handles IPC memory handles

**Gap Analysis:**

| Feature | PyTorch | EXLA |
|---------|---------|------|
| CPU custom ops | `torch.utils.cpp_extension` | XLA FFI custom calls |
| GPU custom ops | `torch.utils.cpp_extension` (CUDA) | **Not available** |
| cuDNN integration | Built-in | XLA backend only |
| User-provided kernels | Easy via extension API | Not supported |

**Contribution Path:**
Adding GPU custom calls to EXLA would require:
1. Modifying `Makefile` to compile `.cu` files
2. Adding GPU device handler registration (`"CUDA"` instead of `"Host"`)
3. Managing CUDA stream synchronization with XLA
4. Handling device memory allocation/deallocation

This is non-trivial but feasible - similar to how Candle does it.

---

## Option C: Simplified NIF (Forward Only)

### Rationale

For **inference-only** use cases (playing against Dolphin), we don't need the backward pass. This dramatically simplifies implementation.

### Reference: flash-attention-minimal

The [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) project shows FlashAttention forward in ~100 lines of CUDA:

```cuda
// Simplified forward pass structure
__global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O,
    int N, int d,
    int Tc, int Tr, int Bc, int Br,
    float softmax_scale
) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Shared memory for tiles
    extern __shared__ float sram[];
    float* Qi = sram;
    float* Kj = &sram[Br * d];
    float* Vj = &sram[Br * d + Bc * d];
    float* S = &sram[Br * d + 2 * Bc * d];

    // Load Q tile to shared memory
    // ... tiled computation with online softmax ...

    // Write output
    for (int i = 0; i < d; i++) {
        O[qkv_offset + (row * d) + i] = ...;
    }
}
```

### Simplified NIF Structure

```
lib/exphil_flash_attention/
├── native/
│   ├── Cargo.toml
│   ├── src/
│   │   └── lib.rs          # Rustler NIF (~100 lines)
│   └── cuda/
│       └── flash_fwd.cu    # Forward kernel (~200 lines)
└── lib/
    └── flash_attention.ex  # Elixir API
```

### API Design

```elixir
defmodule ExPhil.Native.FlashAttention do
  @moduledoc """
  Native FlashAttention-2 forward pass for inference.

  Note: This is forward-only (no gradients). For training,
  use Pure Nx memory_efficient_attention or the Python bridge.
  """

  use Rustler, otp_app: :exphil, crate: "exphil_flash_attention"

  @doc """
  Run FlashAttention forward pass on GPU.

  ## Requirements
  - CUDA 12.0+
  - Ampere+ GPU (RTX 30xx, 40xx, A100, H100)
  - Tensors must be f16 or bf16

  ## Fallback
  Returns {:error, :not_available} if requirements not met.
  Use memory_efficient_attention as fallback.
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, atom()}
  def forward(_query, _key, _value, _opts \\ []) do
    :erlang.nif_error(:not_loaded)
  end

  @doc "Check if NIF is loaded and GPU is compatible."
  @spec available?() :: boolean()
  def available?, do: false  # NIF override
end
```

### Effort Estimate: 1-2 weeks

---

## Option D: Python Bridge (Completed)

Already implemented in this codebase. See:
- `priv/python/flash_attention_server.py`
- `lib/exphil/bridge/flash_attention_port.ex`

### Limitations

- ~1ms serialization overhead per call
- Python GIL contention for concurrent calls
- Requires Python environment deployment
- Not suitable for real-time inference (60 FPS)

### Use Cases

- Prototyping and experimentation
- Batch processing (where overhead is amortized)
- Comparing implementations

---

## Existing Rust Flash Attention Crates

### 1. candle-flash-attn

**Repository**: [huggingface/candle](https://github.com/huggingface/candle)

Full FlashAttention-2 integration for the Candle ML framework:

```rust
use candle_flash_attn::flash_attn;

let output = flash_attn(&q, &k, &v, softmax_scale, causal)?;
```

**Pros**:
- Production-tested by HuggingFace
- Supports forward and backward
- Active maintenance

**Cons**:
- Tightly coupled to Candle's tensor types
- Would need adapter layer for Nx tensors

### 2. candle-flash-attn-v3

**Repository**: [michaelfeil/candle-flash-attn-v3](https://github.com/michaelfeil/candle-flash-attn-v3)

FlashAttention-3 for Hopper GPUs (H100).

### 3. burn_attention

**Crate**: [burn_attention](https://docs.rs/burn_attention/latest/burn_attention/)

Flash Attention for the Burn deep learning framework. Supports multiple backends (CubeCL, CUDA, WGPU).

---

## Tensor Memory Layout

### Nx/EXLA Layout

Nx uses row-major (C-contiguous) layout by default:

```elixir
# Shape: {batch, seq_len, num_heads, head_dim}
# Memory: batch varies slowest, head_dim varies fastest
tensor = Nx.iota({2, 4, 8, 64})
```

### FlashAttention Layout

FlashAttention-2 expects: `[batch, seq_len, num_heads, head_dim]`

This matches Nx's default layout!

### Data Type Requirements

| FlashAttention Version | Supported Types |
|------------------------|-----------------|
| FA2 (Ampere) | fp16, bf16 |
| FA3 (Hopper) | fp16, bf16, fp8 |

Nx tensors default to f32, so conversion is needed:

```elixir
query_f16 = Nx.as_type(query, :f16)
```

---

## Key Questions Answered

### 1. Can we avoid the backward pass for inference?

**Yes.** For playing against Dolphin, we only need forward inference. The backward pass is only needed for training.

### 2. What's the tensor transfer overhead between EXLA and NIF?

With `Nx.to_pointer`/`from_pointer`, transfer can be **zero-copy** if:
- Tensor is already on GPU (EXLA.Backend)
- Memory layout matches (row-major)
- Data type matches (f16/bf16)

If conversion is needed, overhead is O(n) memory copy.

### 3. Are there simpler alternatives to full NIF?

**Yes:**
- **XLA FFI**: If EXLA adds cuDNN FMHA support
- **Forward-only NIF**: ~200 lines of CUDA vs ~2000 for full
- **Python bridge**: Already working (prototype only)

### 4. What GPU features are required?

| Feature | Minimum | Recommended |
|---------|---------|-------------|
| CUDA Compute | 8.0 (Ampere) | 9.0 (Hopper) |
| GPU | RTX 3060 | RTX 4090 / A100 |
| CUDA Toolkit | 12.0 | 12.3+ |
| Memory | 8GB | 24GB+ |

---

## Recommendation

### For ExPhil's Use Case

1. **Short term**: Use Pure Nx `memory_efficient_attention` (already implemented)

2. **Medium term**: Monitor EXLA for cuDNN flash attention support
   - Check [jax.nn.dot_product_attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html) parity
   - XLA flag: `--xla_gpu_enable_cudnn_fmha=true`

3. **If needed**: Implement forward-only NIF using flash-attention-minimal as reference
   - Scope: 1-2 weeks
   - Only for inference (Dolphin play)
   - Fallback to MEA when GPU unavailable

4. **Avoid**: Full NIF with backward pass unless training speed is critical
   - High complexity
   - EXLA's cuDNN support likely coming

---

## References

### Papers
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682) (Rabe & Staats, 2021)

### Code
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - Official CUDA implementation
- [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) - ~100 lines reference
- [huggingface/candle-flash-attn-v1](https://github.com/huggingface/candle-flash-attn-v1) - Rust integration example
- [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) - C++ header-only API

### Documentation
- [XLA Custom Calls](https://openxla.org/xla/custom_call) - FFI documentation
- [JAX FFI Guide](https://docs.jax.dev/en/latest/ffi.html) - Foreign function interface
- [Extending JAX with C++/CUDA](https://dfm.io/posts/extending-jax/) - Tutorial
- [Rustler](https://github.com/rusterlium/rustler) - Rust NIF bindings

### Nx/EXLA
- [Nx Documentation](https://hexdocs.pm/nx/)
- [EXLA Documentation](https://hexdocs.pm/exla/)
- [elixir-nx/nx GitHub](https://github.com/elixir-nx/nx)

---

## Implementation Status (Jan 2026)

### What's Complete

| Component | Status | Location |
|-----------|--------|----------|
| Pure Nx memory-efficient attention | ✓ Done | `lib/exphil/networks/attention.ex` |
| Python bridge (flash_attn) | ✓ Done | `priv/python/flash_attention_server.py` |
| Forward-only NIF (CPU) | ✓ Done | `native/flash_attention_nif/` |
| Forward-only NIF (CUDA) | Written, untested | `native/flash_attention_nif/cuda/` |
| Benchmark script | ✓ Done | `scripts/benchmark_attention.exs` |

### Benchmark Results (CPU)

```
Sequence Length: 64
Standard (O(n²))            | 1829μs  | 1.0x baseline
Chunked (chunk=32)          | 3321μs  | 1.82x slower
Memory-Efficient (O(n))     | 4291μs  | 2.35x slower
NIF FlashAttention (CPU)    | 12690μs | 6.94x slower (data copy overhead)
```

The NIF is slower on CPU due to Elixir↔Rust data marshalling. It's designed for CUDA acceleration.

### Benchmark Results (GPU - RTX 4090, Jan 2026)

**Key Finding:** The NIF is slower than EXLA even on GPU due to host↔device memory transfers.

```
Sequence Length: 32
Standard (O(n²))            | 713μs   | 1.0x baseline
Chunked (chunk=32)          | 555μs   | 1.28x faster
Memory-Efficient (O(n))     | 1131μs  | 1.59x slower
NIF FlashAttention (CUDA)   | 1185μs  | 1.66x slower

Sequence Length: 64
Standard (O(n²))            | 897μs   | 1.0x baseline
NIF FlashAttention (CUDA)   | 2591μs  | 2.89x slower

Sequence Length: 128
Standard (O(n²))            | 1545μs  | 1.0x baseline
NIF FlashAttention (CUDA)   | 6356μs  | 4.11x slower

Sequence Length: 256
Standard (O(n²))            | 2373μs  | 1.0x baseline
NIF FlashAttention (CUDA)   | 14584μs | 6.14x slower
```

**Why is the NIF slower?**

The NIF approach requires:
1. `Nx.backend_copy(Nx.BinaryBackend)` - GPU→CPU transfer
2. `Nx.to_binary()` - Convert to raw bytes
3. `cudaMemcpy(..., HostToDevice)` - CPU→GPU transfer
4. Run CUDA kernel
5. `cudaMemcpy(..., DeviceToHost)` - GPU→CPU transfer
6. `Nx.from_binary()` - Reconstruct tensor

EXLA's standard attention stays entirely on GPU - no host transfers. This makes EXLA dramatically faster despite not using FlashAttention's O(n) memory algorithm.

**Conclusion:** The NIF approach is not viable for GPU performance. For true FlashAttention benefits, we need:
1. **XLA custom calls** - Stay on GPU, no host transfers (see EXLA_GPU_CUSTOM_CALLS.md)
2. **cuDNN FMHA** - Wait for EXLA to expose `--xla_gpu_enable_cudnn_fmha`
3. **Accept EXLA standard attention** - Already fast enough for our seq lengths (≤256)

For ExPhil's use case (seq_len=60 frames), EXLA's standard attention at ~900μs is acceptable for 60 FPS gameplay (16.67ms budget).

### Next Steps

1. ~~**Test CUDA kernel on GPU**~~ ✓ Done - NIF works but is slower than EXLA (see GPU benchmarks above)
2. ~~**Wire up training CLI**~~ ✓ Done - `--flash-attention-nif` flag added
3. ~~**Add memory profiling**~~ ✓ Done - Peak memory shown in benchmark
4. **Explore XLA custom calls** - GPU-to-GPU path without host transfers (see EXLA_GPU_CUSTOM_CALLS.md)
5. **Monitor EXLA cuDNN support** - `--xla_gpu_enable_cudnn_fmha` would give native FlashAttention

### Integration with Dolphin

For real-time play at 60 FPS (16.67ms per frame):
- EXLA standard attention (GPU): ~0.9ms for seq=64 ✓ **Best option**
- EXLA standard attention (CPU): ~2-5ms (acceptable)
- NIF with CUDA: ~2.5ms for seq=64 (slower due to host transfers)
- Target: Leave headroom for embedding, policy network, action sampling

**Updated recommendation:** Use EXLA's standard attention on GPU. The NIF approach is not beneficial due to host↔device memory transfer overhead. For seq_len=60 (1 second of Melee frames), EXLA achieves ~900μs which leaves plenty of headroom in the 16.67ms frame budget.

The NIF may still be useful for:
- CPU-only deployments where EXLA isn't available
- Future XLA custom call integration (GPU-to-GPU, no transfers)

---

## Future Work

### Half-Precision (f16) Support

The current CUDA kernel only supports f32. Adding f16 support would provide:

- **2x memory reduction** - f16 uses half the memory of f32
- **2x compute throughput** - Tensor cores operate on f16 natively
- **Better cache utilization** - More values fit in shared memory

Implementation requires:
1. Add f16 kernel variant in `cuda/flash_attention.cu`
2. Add FFI declaration in `lib.rs` with `#[cfg(feature = "cuda")]`
3. Add Elixir binding `forward_f16/4` in `flash_attention.ex`
4. Convert tensors with `Nx.as_type(tensor, :f16)` before calling

Reference: The official [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) primarily targets f16/bf16 for maximum Tensor Core utilization.

### Backward Pass CUDA Kernel

The backward pass currently uses CPU fallback. A CUDA implementation would:

- Enable GPU-accelerated training with FlashAttention
- Require implementing Algorithm 4 from the FlashAttention-2 paper
- Need ~3x more code than forward pass

See `docs/FLASH_ATTENTION_BACKWARD.md` for algorithm details.

### bf16 Support

For Ampere+ GPUs, bf16 offers better numerical stability than f16 for training:
- Same memory footprint as f16
- Larger dynamic range (same exponent bits as f32)
- Native Tensor Core support on Ampere+
