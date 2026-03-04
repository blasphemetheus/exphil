# Rust-CUDA Kernel Exploration (Rustler NIF)

## Overview

Evaluated Rust-CUDA + Rustler as a GPU kernel integration path for ExPhil. Uses `cudarc` (Rust CUDA bindings) + NVRTC (runtime PTX compilation) wrapped in a Rustler NIF for zero-serialization Elixir integration.

**Test kernel:** `fused_linear_scan` — `h = a*h + b` (same kernel used across all language comparisons)

## Architecture

```
Elixir (Nx tensor) → Nx.to_binary() → Rustler NIF → cudarc/NVRTC → CUDA kernel → cudarc → binary → Nx.from_binary()
```

Key components:
- **Rustler 0.35**: Erlang NIF bindings for Rust (mature, well-maintained)
- **cudarc 0.12**: Safe Rust wrapper over CUDA driver API + NVRTC
- **NVRTC**: Runtime CUDA kernel compilation (kernel source as const string in Rust)
- **OnceCell**: Lazy CUDA context + PTX compilation (first call only)

## Implementation

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `native/rust_linear_scan_nif/Cargo.toml` | ~35 | Dependencies (rustler, cudarc, bytemuck) |
| `native/rust_linear_scan_nif/src/lib.rs` | ~95 | NIF entry points (linear_scan, ping, cuda_available) |
| `native/rust_linear_scan_nif/src/kernel.rs` | ~200 | CUDA kernel + cudarc launch + CPU fallback |
| `lib/exphil/native/rust_linear_scan.ex` | ~100 | Elixir NIF wrapper (Nx ↔ binary conversion) |
| `scripts/benchmark_rust_scan.exs` | ~190 | Multi-seq_len benchmark |

### CUDA Kernel

```cuda
// One thread per (batch, hidden), sequential loop over timesteps
extern "C" __global__ void fused_linear_scan_kernel(
    const float* a_vals, const float* b_vals, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || h >= hidden) return;

    float h_state = h0[b * hidden + h];
    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        h_state = a_vals[idx] * h_state + b_vals[idx];
        output[idx] = h_state;
    }
}
```

### Data Flow

1. Elixir calls `RustLinearScan.linear_scan(a, b, h0)` with Nx tensors
2. Tensors → `Nx.backend_copy(BinaryBackend)` → `Nx.to_binary()` (CPU memory)
3. Rustler receives `Binary` references (zero-copy from BEAM)
4. `bytemuck::cast_slice` reinterprets bytes as `&[f32]` (zero-copy)
5. `cudarc::htod_sync_copy` → GPU memory
6. NVRTC-compiled kernel launched
7. `cudarc::dtoh_sync_copy` → CPU memory
8. `OwnedBinary` → Erlang binary → `Nx.from_binary()` → Nx tensor

## Building

```bash
cd native/rust_linear_scan_nif
cargo build --release --features cuda
cp target/release/librust_linear_scan_nif.so ../../priv/native/
```

## Cloned Pattern

This NIF follows the exact same pattern as `native/selective_scan_nif/` (the production Mamba SSM NIF), adapted for the simpler linear scan signature:

| selective_scan_nif | rust_linear_scan_nif | Change |
|-------------------|---------------------|--------|
| 5 inputs (x, dt, A, B, C) | 3 inputs (a, b, h0) | Simpler recurrence |
| Shape: (batch, seq_len, hidden, state) | Shape: (batch, seq_len, hidden) | No state dim |
| forward + backward kernels | Forward only | Benchmark-only, no training |
| 289 lines lib.rs | ~95 lines lib.rs | No backward pass |

## Developer Experience

**Strengths:**
- Type-safe end-to-end (Rust + CUDA kernel type checking)
- NIF = zero serialization overhead (vs Port-based Julia/Mojo)
- `cargo build` is reproducible and hermetic
- CPU fallback when CUDA not available (graceful degradation)
- Familiar Rustler pattern already used in production (selective_scan_nif)

**Weaknesses:**
- Still requires `Nx.to_binary()` / `Nx.from_binary()` (breaks Nx computation graph)
- CUDA kernel is a string literal (no syntax highlighting, no compile-time checks)
- NVRTC compilation on first call (~100ms cold start)
- More boilerplate than Triton or Futhark for the same kernel

## Benchmark Results

See `scripts/benchmark_rust_scan.exs` output. Expected: within 10% of CUDA C (XLA) since both use the same sequential scan algorithm — the only overhead is NIF data transfer (Nx → binary → GPU → binary → Nx) vs XLA's in-graph execution.

## Verdict

**Best Elixir-native integration path.** Rust-CUDA + Rustler provides the tightest coupling between Elixir and GPU kernels — no Python, no external processes, no serialization. The selective_scan_nif production NIF proves the pattern works at scale.

For new kernels, Triton AOT (compile to cubin, load via C NIF) may be more productive due to less boilerplate and automatic memory tiling. But for Elixir-first teams already comfortable with Rust, this is the natural choice.
