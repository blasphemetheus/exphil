# Mojo Kernel Exploration

## Overview

Evaluation of Mojo for writing GPU kernels that integrate with ExPhil's Elixir training pipeline.

**Kernel:** `fused_linear_scan` — `h = a*h + b` over timesteps.

**Integration:** Port-based via Python wrapper (Mojo Python interop + msgpack).

**Risk level:** HIGH — Mojo is pre-1.0, not in nixpkgs, and may fail to install on NixOS/WSL2. NumPy fallback ensures the benchmark protocol still works.

## Implementation

### Files

| File | Purpose |
|------|---------|
| `native/mojo_scan/linear_scan.mojo` | Mojo kernel (CPU SIMD + GPU stubs) |
| `native/mojo_scan/server.py` | Python wrapper (Mojo interop + NumPy fallback) |
| `native/mojo_scan/mojoproject.toml` | Mojo project config |
| `native/mojo_scan/setup.sh` | Installation helper |
| `lib/exphil/bridge/mojo_port.ex` | Elixir GenServer port |
| `scripts/benchmark_mojo_scan.exs` | Benchmark script |

### Kernel Code

**Mojo (CPU SIMD path):**
```mojo
fn linear_scan_vectorized(a_vals, b_vals, h0, output, batch, seq_len, hidden):
    alias simd_width = simdwidthof[Float32]()
    for b in range(batch):
        var h = 0
        while h + simd_width <= hidden:
            var h_state = h0.offset(b * hidden + h).load[width=simd_width]()
            for t in range(seq_len):
                var idx = b * seq_len * hidden + t * hidden + h
                var a = a_vals.offset(idx).load[width=simd_width]()
                var bv = b_vals.offset(idx).load[width=simd_width]()
                h_state = a * h_state + bv
                output.offset(idx).store(h_state)
            h += simd_width
```

**Mojo GPU path (when API stabilizes):**
```mojo
# Not yet stable — Mojo's @gpu decorator is experimental
@gpu
fn linear_scan_gpu_kernel(...):
    var b = block_idx.x
    var h = thread_idx.x + block_idx.y * block_dim.x
    ...
```

### Code Comparison

| Aspect | CUDA C | Mojo |
|--------|--------|------|
| Kernel lines | ~20 | ~30 (SIMD), ~20 (scalar) |
| Memory model | Explicit GPU | SIMD on CPU, GPU experimental |
| Type safety | Manual casting | Strong typing |
| Build system | nvcc | Mojo compiler / Python interop |
| Debugging | cuda-gdb, nsight | Print-based (limited) |

## Setup

```bash
cd native/mojo_scan
bash setup.sh

# Or manually:
pip install msgpack numpy

# Test without Mojo (NumPy fallback):
mix run scripts/benchmark_mojo_scan.exs
```

## Results

Benchmarked on NixOS/WSL2, batch=4, seq_len=30, hidden=64. 5 warmup, 30 timed, median.

| Metric | Value |
|--------|-------|
| Mojo installed | No (NixOS incompatible, used NumPy fallback) |
| Correctness (atol) | Exact match vs sequential |
| Mojo SIMD kernel (μs) | N/A (Mojo not available) |
| NumPy vectorized (μs) | 124 |
| End-to-end via port (μs) | 3,794 |
| CUDA C reference (μs) | 595 |
| Port overhead | ~3,670 μs (97% of e2e time) |

**Key finding:** The NumPy kernel itself (124μs) is faster than CUDA C (595μs) because it runs on CPU without GPU launch overhead at this tiny size. But port serialization (msgpack encode/decode + stdio) adds ~3.7ms, making the end-to-end time 6.4x slower than CUDA C. This confirms that port-based GPU kernel integration is bottlenecked by data transfer, not computation.

## Assessment

### Pros
- Python-like syntax with C-level performance
- Strong type system prevents common GPU bugs
- SIMD auto-vectorization is elegant
- Potential for GPU kernels when API stabilizes
- Python interop makes integration easy

### Cons
- Pre-1.0: GPU APIs are unstable and may change
- Not in nixpkgs — installation on NixOS is problematic
- GPU kernel support is experimental (CPU SIMD is production-ready)
- No native msgpack library (requires Python wrapper)
- Debugging GPU kernel issues has poor error messages
- Large SDK (~500MB) with non-standard installation

### Installation Notes

Mojo targets Ubuntu/Debian via Modular's installer. On NixOS/WSL2:
- Direct installation likely fails (glibc, FHS assumptions)
- Docker container is the most reliable option
- Without Mojo, the server falls back to NumPy (still useful for protocol testing)

### Verdict

**Not ready for production use.** Mojo couldn't be installed on NixOS/WSL2 (not in nixpkgs, proprietary installer assumes FHS). NumPy fallback validated the port integration protocol — the kernel itself is fast but port overhead dominates (97% of e2e time). Revisit when Mojo reaches 1.0 and lands in nixpkgs.
