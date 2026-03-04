# Julia CUDA.jl Kernel Exploration

## Overview

Evaluation of Julia's CUDA.jl and KernelAbstractions.jl for writing GPU kernels that integrate with ExPhil's Elixir training pipeline.

**Kernel:** `fused_linear_scan` — `h = a*h + b` over timesteps.

**Integration:** Port-based (msgpack over length-prefixed stdio), same pattern as PyTorch port.

## Implementation

### Files

| File | Purpose |
|------|---------|
| `native/julia_scan/Project.toml` | Julia package dependencies |
| `native/julia_scan/kernels.jl` | CUDA.jl + KernelAbstractions kernels |
| `native/julia_scan/linear_scan_server.jl` | Msgpack stdio server |
| `native/julia_scan/setup.sh` | Dependency installation |
| `lib/exphil/bridge/julia_port.ex` | Elixir GenServer port |
| `scripts/benchmark_julia_scan.exs` | Benchmark script |

### Kernel Code Comparison

**CUDA C (reference, 149 lines total, ~20 lines kernel):**
```c
__global__ void fused_linear_scan_kernel(
    const io_type* a_vals, const io_type* b_vals,
    const io_type* h0, io_type* output,
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || h >= hidden) return;
    float h_state = IO_LOAD(h0, b * hidden + h);
    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float a = IO_LOAD(a_vals, idx);
        float bv = IO_LOAD(b_vals, idx);
        h_state = a * h_state + bv;
        IO_STORE(output, idx, h_state);
    }
}
```

**Julia CUDA.jl (~25 lines kernel):**
```julia
function linear_scan_cuda_kernel!(output, a_vals, b_vals, h0, batch, seq_len, hidden)
    b = blockIdx().x
    h = threadIdx().x + (blockIdx().y - Int32(1)) * blockDim().x
    if b > batch || h > hidden return nothing end
    h_state = h0[h, b]
    @inbounds for t in Int32(1):seq_len
        a = a_vals[h, t, b]; bv = b_vals[h, t, b]
        h_state = a * h_state + bv
        output[h, t, b] = h_state
    end
    return nothing
end
```

**Julia KernelAbstractions (~20 lines kernel):**
```julia
@kernel function linear_scan_ka_kernel!(output, a_vals, b_vals, h0, seq_len)
    I = @index(Global)
    # ... index decomposition ...
    h_state = h0[h_idx, b_idx]
    @inbounds for t in Int32(1):seq_len
        h_state = a_vals[h_idx, t, b_idx] * h_state + b_vals[h_idx, t, b_idx]
        output[h_idx, t, b_idx] = h_state
    end
end
```

### Code Size

| Component | CUDA C | Julia |
|-----------|--------|-------|
| Kernel | ~20 lines | ~25 lines |
| Launch/glue | ~40 lines | ~10 lines |
| FFI boilerplate | ~50 lines | 0 (Port handles it) |
| Server protocol | N/A | ~80 lines |
| Elixir integration | ~50 lines (NIF module) | ~160 lines (GenServer) |
| **Total** | **~160 lines** | **~275 lines** |

Julia requires more total code due to the server protocol, but the kernel itself is nearly identical in size and much easier to read.

## Setup

```bash
# Add julia to shell.nix, then:
cd native/julia_scan
bash setup.sh

# Test
mix run scripts/benchmark_julia_scan.exs
```

**Known issues:**
- Julia's CUDA.jl may conflict with shell.nix CUDA 12.8 — set `JULIA_CUDA_USE_BINARYBUILDER=false`
- First run takes 30-60s for JIT compilation
- MsgPack serialization adds overhead for large tensors

## Results

Benchmarked on NVIDIA T400 4GB, NixOS/WSL2, batch=4, seq_len=60, hidden=64. 5 warmup, 30 timed, median.

| Metric | Value |
|--------|-------|
| Correctness (atol) | 5.0e-7 (both CUDA.jl and KA) |
| Julia CUDA.jl kernel (μs) | 588 |
| Julia KA kernel (μs) | 582 |
| Julia CUDA.jl e2e (μs) | 9,551 |
| Julia KA e2e (μs) | 9,545 |
| CUDA C reference (μs) | 3,230 |
| Julia kernel / CUDA C | 0.18x (Julia kernel is faster!) |
| Julia e2e / CUDA C | 2.96x (port overhead kills it) |
| Pure Nx fallback (μs) | 461,868 |

**Key finding:** Julia's GPU kernel (588μs) is **faster** than CUDA C via XLA (3,230μs) for kernel-only timing. The CUDA C number includes XLA dispatch overhead. But the Elixir → Julia port serialization adds ~9ms, making end-to-end 3x slower.

**Port overhead breakdown:** Of the 9,551μs e2e, ~588μs is kernel execution and ~8,963μs (94%) is msgpack serialization + stdio transfer + Julia-side array construction.

## Assessment

### Pros
- Nearly identical kernel syntax to CUDA C — easy to port
- KernelAbstractions.jl is vendor-neutral (CUDA/ROCm/oneAPI/Metal)
- Rich ecosystem: profiling, visualization, debugging tools
- Interactive REPL makes kernel development faster
- Array abstractions are ergonomic (no manual memory management)

### Cons
- 30-60s JIT cold-start (amortized by long-lived server)
- Port serialization overhead (~0.5-2ms for medium tensors)
- Julia↔Elixir data copy goes through CPU (no GPU-direct path)
- Extra process + protocol complexity vs NIF
- Large runtime (~200MB Julia + packages)

### Verdict

**Best alternative for kernel prototyping.** Julia's GPU kernel (588μs) is actually faster than CUDA C via XLA (3,230μs) at kernel-level timing — the XLA dispatch adds overhead that Julia avoids. KernelAbstractions (582μs) matches CUDA.jl performance while being vendor-neutral (CUDA/ROCm/oneAPI/Metal).

The dealbreaker is port serialization: 94% of end-to-end time is msgpack encode/decode + stdio transfer. For prototyping kernels this doesn't matter (iterate fast in Julia REPL, then port to CUDA C for production). For production use, a NIF-based integration or shared-memory IPC would be needed to eliminate the serialization bottleneck.

**JIT warmup:** ~20s cold start (acceptable for long-lived server). NixOS "non-official build" warning is harmless — all tests pass, CUDA works correctly.
