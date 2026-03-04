# GPU Kernel Language Options for ExPhil

## Context

ExPhil uses CUDA C kernels via XLA custom calls for SSM recurrences (`h = a*h + b`). We benchmarked four alternatives (Julia, Futhark, Mojo, Bend) and CUDA C won at all tested sizes. This document surveys the broader landscape to identify what else is worth exploring.

**Hardware:** NVIDIA T400 4GB (Turing, sm_75), NixOS/WSL2, CUDA 12.8.

## Completed Benchmarks

| Backend | Kernel (μs) | E2E (μs) | Integration | Notes |
|---------|------------|----------|-------------|-------|
| CUDA C (XLA) | — | 3,230 | XLA custom call | Production baseline |
| Rust-CUDA | — | ~8,500 | Rustler NIF | cudarc + NVRTC, ~2.5x CUDA C |
| Triton AOT | — | TBD | C NIF (cubin) | AOT compile, zero Python runtime |
| Julia CUDA.jl | 588 | 9,551 | Port (msgpack) | Kernel fast, port overhead 94% |
| Julia KA | 582 | 9,545 | Port (msgpack) | Vendor-neutral, same perf |
| Futhark (NIF) | — | 6,338 | NIF (.so) | Parallel prefix scan, 2.55x CUDA C |
| NumPy (CPU) | 124 | 3,794 | Port (msgpack) | CPU only, port overhead 97% |
| Bend (Rust) | — | 119,185 | CLI | 24.5x slower than Nx, learning exercise |
| Nx (CPU) | — | 461,868 | Native | Pure Elixir reference |

**Key finding:** All NIF-based approaches (Rust, Triton, Futhark) are 2-3x slower than XLA due to Nx → binary → GPU → binary → Nx data transfer. The bottleneck is not kernel quality but the data path. Port-based integration (Julia, Mojo/NumPy) adds another 3x on top from serialization.

See [KERNEL_LANGUAGE_COMPARISON.md](KERNEL_LANGUAGE_COMPARISON.md) for detailed results.

---

## Tier 1: Worth Benchmarking

These have realistic Elixir integration paths, run on NixOS/WSL2, and bring something new.

### ~~1. Triton (OpenAI)~~ — COMPLETED

**Status:** Benchmarked and working. See [TRITON_KERNEL_EXPLORATION.md](TRITON_KERNEL_EXPLORATION.md).

AOT-compiled cubin loaded via C NIF. 15 lines of Python kernel code, zero Python at runtime. Performance limited by NIF data transfer (same as all NIF approaches), not kernel quality. Triton is available in nixpkgs (`python3Packages.triton`).

### 2. cuda.compute (NVIDIA CCCL)

NVIDIA's Python library providing optimized parallel algorithms (scan, reduce, sort) with custom types/operators. JIT-compiles to architecture-specific kernels.

| Aspect | Details |
|--------|---------|
| GPU support | CUDA only |
| Maturity | Medium — topped GPU MODE kernel leaderboard early 2026 |
| NixOS | `pip install cuda-compute` |
| Code size | ~10 lines (define operator, call scan) |

**Integration:** Python Port (msgpack), or AOT compile generated kernels → NIF.

**Why benchmark:** Define the associative operator `(a1,b1) * (a2,b2) = (a1*a2, a1*b2+b1)` and NVIDIA's library handles the parallel scan. Least custom code of any option. Quick to test via existing Port pattern.

### 3. ThunderKittens (Stanford Hazy Research)

CUDA-embedded C++ DSL for AI kernels. Tile-level abstractions over tensor cores and shared memory. v2.0 (Feb 2026) with Blackwell, FP8, multi-GPU.

| Aspect | Details |
|--------|---------|
| GPU support | CUDA (primary), HIP/ROCm (via HipKittens) |
| Maturity | Medium-high — production at Cursor, Together AI |
| NixOS | Header-only C++, just needs nvcc |
| Code size | ~15-20 lines with tile abstractions |

**Integration:** Compile to .so → C NIF or XLA custom call. Same build as existing CUDA C kernels (header-only library).

**Why benchmark:** Built by Hazy Research (the Mamba/SSM group). Tile abstractions reduce boilerplate 3-5x. Tensor core usage is automatic. The Mamba/SSM use case is exactly what they designed for.

### ~~4. Rust-CUDA + Rustler~~ — COMPLETED

**Status:** Benchmarked and working. See [RUST_CUDA_KERNEL_EXPLORATION.md](RUST_CUDA_KERNEL_EXPLORATION.md).

Rustler NIF with cudarc + NVRTC. Same pattern as production `selective_scan_nif`. ~2.5x overhead vs CUDA C (XLA) due to NIF data transfer. Best Elixir-native integration path — type-safe, no Python, `cargo build`.

### 5. CUTLASS / CuTe (NVIDIA)

NVIDIA's template library for high-performance linear algebra. CuTe DSL (Python, 2025) for writing kernels without C++.

| Aspect | Details |
|--------|---------|
| GPU support | CUDA only, Hopper/Blackwell optimized |
| Maturity | Very high (NVIDIA's own library) |
| NixOS | Header-only C++ with nvcc, CuTe DSL via pip |
| Code size | Varies — templates are verbose but powerful |

**Integration:** C++ headers in existing CUDA build, or CuTe DSL AOT → cubin → NIF.

**Why benchmark:** Only if fusing scan with matmul (Mamba-2 SSD). CUTLASS shines at matmul-heavy kernels, not simple recurrences.

---

## Tier 2: Viable but Lower Priority

These work but don't offer compelling advantages over Triton + CUDA C for `fused_linear_scan`.

| Tool | GPU Support | Why Lower Priority |
|------|-----------|-------------------|
| **Taichi** | CUDA, Vulkan, Metal | Optimized for physics/spatial, not sequential recurrences |
| **CuPy RawKernel** | CUDA | Writing CUDA C strings in Python — we already have CUDA C |
| **Numba CUDA** | CUDA | CUDA semantics in Python with JIT overhead — Triton is strictly better |
| **HIP (AMD)** | ROCm, CUDA via hipcc | On NVIDIA hardware, same binary as CUDA C. Only useful for AMD GPUs |
| **SYCL / oneAPI** | Intel, NVIDIA, AMD | Portability play — complex toolchain for no perf gain on NVIDIA |
| **Pallas (JAX)** | CUDA via Triton, TPU | Calls Triton underneath — use Triton directly |

---

## Tier 3: Not Practical for ExPhil

| Tool | Why Skip |
|------|----------|
| **Halide** | Image processing DSL, wrong paradigm for sequential recurrences |
| **Chapel** | HPC/distributed focus, GPU support is recent bolt-on |
| **Kokkos** | HPC portability library, massive overhead for single-GPU kernel |
| **OpenCL** | 13-67% slower than CUDA on NVIDIA, declining relevance |
| **WGPU/WebGPU** | Shader language via Vulkan, no tensor cores, no warp primitives |
| **Zig + CUDA** | Two layers of immature interop for no gain |
| **Nim + CUDA** | Arraymancer GPU support less mature than any other option |
| **cuTile (NVIDIA)** | Requires Blackwell GPU (sm_100+), T400 is Turing |

---

## Remaining Benchmark Plan

| Priority | Tool | Integration | Effort | Expected Outcome |
|----------|------|-------------|--------|-----------------|
| ~~**P0**~~ | ~~Triton~~ | ~~AOT cubin → C NIF~~ | ~~Medium~~ | **DONE** — works, ~2-3x CUDA C (NIF overhead) |
| **P1** | cuda.compute | Python Port or AOT | Low | Optimized parallel scan with custom assoc operator |
| **P2** | ThunderKittens | Header-only C++ → XLA CC | Medium | Potentially faster than raw CUDA C for SSM/attention |
| ~~**P3**~~ | ~~Rust-CUDA + Rustler~~ | ~~Rustler NIF~~ | ~~High~~ | **DONE** — works, ~2.5x CUDA C (NIF overhead) |
| **P4** | CUTLASS/CuTe | C++ headers | Medium | Only if fusing scan + matmul (Mamba-2 SSD) |

## Future TODOs

- **P1: cuda.compute** — NVIDIA CCCL parallel scan with custom associative operator `(a1,b1)*(a2,b2) = (a1*a2, a1*b2+b1)`. Least custom code of any option (~10 lines). Quick to test via existing Port pattern.
- **P2: ThunderKittens** — Stanford Hazy Research tile-level C++ DSL for AI kernels. Built by the Mamba/SSM group. Tile abstractions reduce boilerplate 3-5x. Tensor core usage is automatic.
- **P4: CUTLASS/CuTe** — Only if fusing scan+matmul for Mamba-2 SSD. CUTLASS shines at matmul-heavy kernels, not simple recurrences.
- **XLA custom call integration** — The real performance win. All NIF approaches have ~2-3x overhead from data transfer. Integrating Triton/Rust kernels as XLA custom calls would keep tensors on GPU and eliminate this bottleneck.

---

## Related Docs

| Doc | Contents |
|-----|----------|
| [KERNEL_LANGUAGE_COMPARISON.md](KERNEL_LANGUAGE_COMPARISON.md) | Benchmark results and synthesis |
| [RUST_CUDA_KERNEL_EXPLORATION.md](RUST_CUDA_KERNEL_EXPLORATION.md) | Rust-CUDA Rustler NIF findings |
| [TRITON_KERNEL_EXPLORATION.md](TRITON_KERNEL_EXPLORATION.md) | Triton AOT C NIF findings |
| [JULIA_KERNEL_EXPLORATION.md](JULIA_KERNEL_EXPLORATION.md) | Julia CUDA.jl detailed findings |
| [FUTHARK_KERNEL_EXPLORATION.md](FUTHARK_KERNEL_EXPLORATION.md) | Futhark parallel prefix scan |
| [MOJO_KERNEL_EXPLORATION.md](MOJO_KERNEL_EXPLORATION.md) | Mojo/NumPy findings |
| [BEND_KERNEL_EXPLORATION.md](BEND_KERNEL_EXPLORATION.md) | Bend learning exercise |
| [MOJO_NIXOS_INSTALLATION.md](MOJO_NIXOS_INSTALLATION.md) | Getting Mojo running on NixOS/WSL2 |
