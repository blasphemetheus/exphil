# XLA Custom Call Integration — The Highest-Impact Optimization

## Executive Summary

All NIF-based GPU kernel approaches (Rust-CUDA, Triton, Futhark, ThunderKittens) are **2-3x slower** than CUDA C via XLA custom calls. The bottleneck is not kernel quality — it's the data path:

```
NIF path:    Nx tensor (GPU) → Nx.to_binary (CPU) → NIF → cudaMemcpy (GPU) → kernel → cudaMemcpy (CPU) → Nx.from_binary (GPU)
XLA path:    Nx tensor (GPU) → XLA custom call → kernel (GPU, zero-copy) → result stays on GPU
```

XLA custom calls eliminate **all CPU-GPU data transfers**. Tensors stay on GPU throughout. This is why our CUDA C kernels in EXLA (3.2ms) beat every NIF approach (6-10ms) despite running the same kernel code.

**Recommendation:** Invest in XLA custom call integration for production kernels. This is the single highest-impact optimization available — it would make all kernel languages equally fast by removing the NIF overhead bottleneck.

---

## Current State

### What We Have

ExPhil's EXLA fork (`/home/nixos/nx/exla`) already has extensive custom call infrastructure:

| Component | Status | Location |
|-----------|--------|----------|
| **Makefile CUDA support** | Working | `Makefile` — auto-discovers `custom_calls/*.cu` |
| **47+ fused scan kernels** | Production | `c_src/exla/custom_calls/fused_*.cu` |
| **gpu_add prototype** | Working | `c_src/exla/custom_calls/gpu_add.cu` |
| **Value.ex bindings** | Working | `lib/exla/mlir/value.ex` — 30+ custom calls |
| **XLA FFI registration** | Working | `XLA_FFI_REGISTER_HANDLER` with platform "CUDA" |
| **bf16/f32 mixed precision** | Working | `precision.cuh` — conditional compilation |
| **Runtime callbacks** | Working | `runtime_callback.cc` — Elixir ↔ GPU bridge |
| **Test coverage** | Good | `test/exla/gpu_custom_call_test.exs` |

### What Works Today

Adding a new custom call kernel requires:

1. **Write kernel** — `c_src/exla/custom_calls/my_kernel.cu`
2. **Register FFI handler** — `XLA_FFI_REGISTER_HANDLER(..., "CUDA", handler)`
3. **Add MLIR binding** — `Value.my_operation(...)` using `stablehlo.custom_call`
4. **Rebuild EXLA** — `EXLA_FORCE_REBUILD=true mix compile`

The Makefile auto-discovers `.cu` files and compiles them with nvcc when CUDA is detected.

### Proven Pattern

Every fused scan kernel follows this pattern:

```cpp
// 1. CUDA kernel
__global__ void my_kernel(const float* input, float* output, int batch, ...) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;
    // ... kernel logic ...
}

// 2. FFI handler (runs on CPU, receives GPU pointers)
static ffi::Error my_kernel_impl(
    cudaStream_t stream,                    // CUDA stream from XLA
    ffi::Buffer<ffi::F32> input,           // Already on GPU!
    ffi::Result<ffi::Buffer<ffi::F32>> out // Pre-allocated on GPU!
) {
    const float* in_ptr = input.typed_data();   // GPU pointer, zero-copy
    float* out_ptr = out->typed_data();         // GPU pointer, zero-copy

    my_kernel<<<grid, block, 0, stream>>>(in_ptr, out_ptr, ...);
    return ffi::Error::Success();
}

// 3. Registration
XLA_FFI_DEFINE_HANDLER_SYMBOL(my_handler, my_kernel_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_my_kernel_f32", "CUDA", my_handler);
```

```elixir
# 4. Elixir binding (lib/exla/mlir/value.ex)
def my_operation(%Value{function: func} = input, out_typespec) do
  operands = [input]
  result_types = typespecs_to_mlir_types([out_typespec])
  attributes = [
    call_target_name: attr_string("exla_my_kernel_f32"),
    api_version: attr_i32(4)
  ]
  op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)
  |> one!()
end
```

---

## Architecture

### Data Flow Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                 NIF-based approach (current)                │
│                                                             │
│  EXLA tensor ──► Nx.to_binary ──► CPU binary ──► NIF       │
│       (GPU)        (GPU→CPU)        (CPU RAM)    (CPU)      │
│                                                    │        │
│                                              cudaMemcpy     │
│                                              (CPU→GPU)      │
│                                                    │        │
│                                              CUDA kernel    │
│                                                    │        │
│                                              cudaMemcpy     │
│                                              (GPU→CPU)      │
│                                                    │        │
│  EXLA tensor ◄── Nx.from_binary ◄── CPU binary ◄──┘        │
│       (GPU)        (CPU→GPU)          (CPU RAM)             │
│                                                             │
│  Total transfers: 4 (GPU→CPU, CPU→GPU, GPU→CPU, CPU→GPU)   │
│  Overhead: ~5-7ms for typical sizes                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 XLA custom call approach                     │
│                                                             │
│  EXLA tensor ──► XLA runtime ──► FFI handler ──► kernel     │
│       (GPU)      (schedules)     (gets stream)   (GPU)      │
│                                                    │        │
│  EXLA tensor ◄────────────────────────────────────┘        │
│       (GPU)         (stays on GPU the entire time)          │
│                                                             │
│  Total transfers: 0                                         │
│  Overhead: XLA dispatch only (~0.1-0.5ms)                   │
└─────────────────────────────────────────────────────────────┘
```

### Performance Impact

| Approach | Kernel Time | Data Transfer | Total e2e | vs XLA CC |
|----------|------------|---------------|-----------|-----------|
| XLA custom call | ~0.5ms | 0ms | ~3ms | 1.0x |
| Rust NIF | ~0.5ms | ~5-7ms | ~8.5ms | ~2.6x |
| Triton NIF | ~0.5ms | ~5-7ms | ~8ms | ~2.5x |
| Futhark NIF | ~0.5ms | ~5-7ms | ~6.3ms | ~2.0x |
| TK NIF | ~0.5ms | ~5-7ms | ~8ms | ~2.5x |
| CuPy Port | ~0.5ms | ~8-10ms | ~10ms | ~3x |

The kernel execution time is nearly identical across all approaches. The entire performance difference is data transfer.

---

## Integration Approaches

### Approach 1: Add Kernels to EXLA Fork (Current, Proven)

**How:** Drop `.cu` files into `c_src/exla/custom_calls/`, add `Value.ex` bindings, rebuild.

**Pros:**
- Already works — 47+ kernels in production
- Zero-copy GPU tensors
- Automatic mixed precision (bf16/f32)
- Integrated with XLA's memory management and scheduling

**Cons:**
- Requires rebuilding EXLA (`EXLA_FORCE_REBUILD=true mix compile`, ~15-30 min)
- Kernels are tied to EXLA version
- Can't add kernels at runtime (compile-time only)

**Best for:** Production kernels with stable APIs (fused scans, attention).

### Approach 2: EXLA Plugin API (Future)

**How:** EXLA would expose a runtime registration API:

```elixir
# Hypothetical API
EXLA.register_custom_kernel("my_kernel", "/path/to/libmy_kernel.so",
  inputs: [{:f32, {batch, seq_len, hidden}}],
  outputs: [{:f32, {batch, seq_len, hidden}}]
)

defn fast_scan(a, b) do
  EXLA.custom_call("my_kernel", [a, b])
end
```

**Pros:**
- No EXLA rebuild needed
- Kernels can be developed independently
- Plugin ecosystem possible

**Cons:**
- Doesn't exist yet — would require significant EXLA changes
- XLA's FFI registration is compile-time, not runtime
- Need to solve: how to register handlers after XLA client starts

**Feasibility:** Medium. XLA does support dynamic library loading (`dlopen`), but the FFI handler registration happens at static initialization time. A workaround would be to have EXLA load user `.so` files before creating the XLA client, allowing their `__attribute__((constructor))` functions to register handlers.

### Approach 3: Direct GPU Buffer Access (Workaround)

**How:** Access EXLA's underlying GPU buffer pointer, pass to cudarc:

```elixir
# Hypothetical — EXLA would need to expose this
{:ok, a_ptr} = EXLA.Backend.get_device_pointer(a_tensor)
{:ok, b_ptr} = EXLA.Backend.get_device_pointer(b_tensor)

# Launch kernel directly via cudarc (Rust NIF)
result_ptr = MyKernel.launch(a_ptr, b_ptr, batch, seq_len, hidden)

# Wrap pointer back as EXLA tensor
result = EXLA.Backend.from_device_pointer(result_ptr, shape, type)
```

**Pros:**
- Could work with any kernel language (Rust, C, Triton cubins)
- No EXLA rebuild for new kernels
- Keeps tensors on GPU

**Cons:**
- EXLA doesn't expose buffer pointers (would need PR)
- Bypasses XLA's memory management (risk of double-free, use-after-free)
- No stream synchronization guarantees
- Fragile — depends on EXLA internals

**Feasibility:** Low. Too many safety concerns for production use.

### Approach 4: Nx.Defn Compiler Hook (Cleanest Long-term)

**How:** Intercept specific operations at the `Nx.Defn` compiler level and route to custom calls:

```elixir
# In EXLA.Defn compiler
defp cached_recur_operator(%{op: :custom_linear_scan}, nodes, state) do
  [a, b] = nodes
  if state.platform == :cuda do
    Value.fused_linear_scan(a, b, out_typespec)
  else
    # CPU fallback
    standard_scan(a, b)
  end
end
```

**Pros:**
- Transparent to user code (just use `Nx.Defn` as normal)
- Platform-aware (GPU kernel on CUDA, fallback on CPU)
- Integrates with XLA's optimization passes

**Cons:**
- Requires EXLA changes for each new kernel
- Tight coupling between Nx operations and kernel implementations

**Status:** This is already how the 47 fused scan kernels work. The pattern is proven.

---

## Implementation Roadmap

### Phase 1: Validate with fused_linear_scan (DONE)

The `fused_linear_scan` kernel already exists as an XLA custom call:
- Kernel: `c_src/exla/custom_calls/fused_linear_scan.cu`
- Binding: `Value.fused_linear_scan/3`
- Invoked by: `Edifice.CUDA.FusedScan.linear_scan/2`
- Performance: 3.2ms (vs 8.5ms NIF)

### Phase 2: Add selective_scan as XLA Custom Call

The kernel is ready at `native/xla_selective_scan/selective_scan_kernel.cu`. Steps:

1. Copy kernel to `c_src/exla/custom_calls/selective_scan.cu`
2. Convert from legacy CustomCall interface (`void** buffers`) to typed FFI
3. Add `Value.selective_scan/5` binding
4. Add pattern match in EXLA.Defn compiler
5. Rebuild EXLA, test against Rust NIF for correctness

**Expected improvement:** 10.96ms (Rust NIF) → ~5ms (XLA CC) for selective scan.

### Phase 3: FlashAttention Custom Call

Create `flash_attention_fwd.cu` and `flash_attention_bwd.cu`:

1. Implement tiled attention algorithm (FlashAttention-2)
2. Save logsumexp for backward pass
3. Register as `exla_flash_attn_fwd` / `exla_flash_attn_bwd`
4. Add `Value.flash_attention_forward/4` and `Value.flash_attention_backward/7`
5. Add `Nx.Defn.Kernel.custom_grad` wrapper for autograd

**Expected improvement:** Current EXLA attention is O(N^2) memory. FlashAttention would be O(N) memory, enabling much longer sequences.

### Phase 4: Plugin API (Long-term)

Design and implement runtime kernel registration:

1. Add `EXLA.load_custom_kernels/1` that dlopen's user `.so` files
2. Call before XLA client creation to allow FFI registration
3. Provide `mix exla.compile_kernel` task for building from `.cu` to `.so`
4. Document the plugin API

---

## Key Technical Details

### XLA FFI API Versions

| Version | Name | Interface | Used By |
|---------|------|-----------|---------|
| 1 | Legacy | `void f(void* out, void** in)` | Deprecated |
| 2 | CustomCall | `void f(stream, void** buffers, opaque)` | `xla_selective_scan` |
| 4 | Typed FFI | `ffi::Error f(stream, Buffer<T>...)` | All EXLA custom calls |

**Always use version 4** (typed FFI). It provides type safety, automatic buffer validation, and cleaner error handling.

### Critical API Differences

```cpp
// GPU (CUDA) handlers — use C API
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "name", "CUDA", handler);

// CPU (Host) handlers — use C++ API
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "name", "Host", handler);
```

Using the wrong API function causes silent registration failure.

### Mixed Precision

All EXLA custom calls support bf16/f32 via `precision.cuh`:

```cpp
#include "precision.cuh"
// USE_BF16 defined at compile time
// io_type = __nv_bfloat16 or float
// FFI_IO_TYPE = ffi::BF16 or ffi::F32
// IO_LOAD/IO_STORE macros handle conversion
```

Each kernel is compiled twice (f32 and bf16 variants) and registered with different names:
- `exla_fused_linear_scan_f32`
- `exla_fused_linear_scan_bf16`

---

## Comparison: NIF vs XLA Custom Call

| Aspect | NIF (Rust/C) | XLA Custom Call |
|--------|-------------|-----------------|
| Data path | CPU round-trip | Zero-copy GPU |
| Performance | ~8-10ms | ~3ms |
| Build | `cargo build` / `make` | EXLA rebuild (15-30min) |
| Hot reload | Yes (NIF reload) | No (full rebuild) |
| Language | Any (Rust, C, Triton) | CUDA C/C++ only |
| Memory mgmt | Manual (cudaMalloc/Free) | XLA handles it |
| Stream sync | Manual | Automatic |
| Error handling | Manual | FFI error types |
| bf16 support | Manual | Built-in |
| Autograd | Manual | Via custom_grad |
| CPU fallback | Separate code path | XLA handles it |

---

## Conclusion

**XLA custom calls are the right path for production kernels.** The 2-3x NIF overhead is a fundamental architectural limitation (CPU-GPU data copies), not a tuning issue. No amount of kernel optimization can overcome it.

The infrastructure is already in place — 47+ kernels prove the pattern works. The next step is converting key standalone kernels (selective_scan, future FlashAttention) from NIF to XLA custom call format.

For prototyping and development, NIFs remain valuable:
- **Rust-CUDA NIF** — best for Elixir-native development with type safety
- **Triton AOT** — best for writing new kernels quickly (15 lines Python)
- **CuPy Port** — best for experimenting with parallel scan algorithms

Once a kernel is proven via NIF prototyping, convert it to XLA custom call for production performance.

---

## Related Docs

| Doc | Contents |
|-----|----------|
| [EXLA_GPU_CUSTOM_CALLS.md](../internals/EXLA_GPU_CUSTOM_CALLS.md) | EXLA FFI architecture and prototype |
| [GPU_KERNEL_OPTIONS.md](GPU_KERNEL_OPTIONS.md) | Kernel language survey |
| [KERNEL_LANGUAGE_COMPARISON.md](KERNEL_LANGUAGE_COMPARISON.md) | Benchmark results |
| [RUST_CUDA_KERNEL_EXPLORATION.md](RUST_CUDA_KERNEL_EXPLORATION.md) | Rust-CUDA NIF findings |
| [TRITON_KERNEL_EXPLORATION.md](TRITON_KERNEL_EXPLORATION.md) | Triton AOT findings |
| [CUDA_COMPUTE_EXPLORATION.md](CUDA_COMPUTE_EXPLORATION.md) | CuPy/CCCL findings |
| [THUNDERKITTENS_EXPLORATION.md](THUNDERKITTENS_EXPLORATION.md) | ThunderKittens findings |
