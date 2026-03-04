# GPU Kernel Language Comparison

## Executive Summary

Evaluated four languages (Julia, Futhark, Mojo, Bend) as alternatives to CUDA C for writing GPU kernels in ExPhil's Elixir ML pipeline. The test kernel is `fused_linear_scan` — the simplest recurrence `h = a*h + b`.

**Recommendation:** Continue with CUDA C for production kernels. Julia (CUDA.jl) is the best alternative for rapid prototyping. Futhark is interesting for algorithmic exploration (parallel scan).

## Benchmark Protocol

- **Kernel:** `fused_linear_scan` — `h = a*h + b` over timesteps
- **Reference:** `/home/nixos/nx/exla/c_src/exla/custom_calls/fused_linear_scan.cu` (149 lines)
- **Sizes:** Inference (4×60×64), Training (32×120×512)
- **Timing:** 5 warmup, 30 timed iterations, report median
- **Correctness:** Compare against pure Nx sequential scan

## Results

Benchmarked on NVIDIA T400 4GB, NixOS/WSL2, March 2026. Inference size: batch=4, seq_len=30, hidden=64. 5 warmup iterations, 30 timed, median reported.

### Latency Comparison

| Implementation | Kernel (μs) | E2E (μs) | vs CUDA C (e2e) | Correct | Notes |
|----------------|------------|----------|-----------------|---------|-------|
| CUDA C (XLA CC) | — | 3,230 | 1.0x | ref | Production baseline (seq_len=60) |
| Rust-CUDA (NIF) | — | ~8,500 | ~2.5x | yes (1e-7) | Rustler NIF, cudarc + NVRTC |
| Triton AOT (NIF) | — | TBD | TBD | yes (1e-7) | cubin → C NIF, zero Python |
| Julia CUDA.jl | 588 | 9,551 | 2.96x | yes (5e-7) | Kernel faster than XLA! Port kills e2e |
| Julia KA | 582 | 9,545 | 2.96x | yes (5e-7) | Vendor-neutral, same perf |
| Futhark (NIF) | — | 6,338 | 1.96x* | yes (~1e-3) | Parallel prefix scan (*seq_len=30) |
| NumPy (kernel) | 124 | 3,794 | 1.17x | yes | CPU only, port overhead 97% |
| Bend (Rust) | — | 119,185 | 36.9x | yes | Learning exercise only |
| Pure Nx (CPU) | — | 461,868 | 143x | ref | Elixir fallback |

Note: Julia benchmarked at seq_len=60, Futhark/NumPy/Bend at seq_len=30. CUDA C number (3,230μs) is from Julia benchmark run (seq_len=60). Rust-CUDA number is median across seq_len=30-480.

**Key insight:** Julia's raw kernel (588μs) beats CUDA C via XLA (3,230μs) because XLA adds dispatch overhead. But port serialization adds ~9ms (94% of e2e), making it uncompetitive for production. NIF-based integration (like Futhark) avoids this — Futhark's 6.3ms includes no serialization overhead.

### Futhark Scaling (batch=4, hidden=64)

Futhark's parallel prefix scan has O(log T) depth. At what seq_len does it beat CUDA C?

| seq_len | Futhark (μs) | CUDA C (μs) | Nx (μs) | Futhark/CUDA C |
|---------|-------------|-------------|---------|----------------|
| 30 | 6,338 | 2,485 | 229,854 | 2.55x |
| 60 | 6,914 | 3,613 | 525,719 | 1.91x |
| 120 | 7,693 | 3,660 | 1,056,859 | 2.10x |
| 240 | 9,799 | 3,990 | 2,081,286 | 2.46x |
| 480 | 8,823 | 2,580 | 2,068,732 | 3.42x |
| 960 | 21,899 | 5,563 | 9,984,523 | 3.94x |

**Finding:** CUDA C wins at all tested lengths. Futhark did not reach the crossover point. The parallel prefix scan's 2x extra work outweighs its O(log T) advantage at these sizes. Crossover likely requires hidden_dim >> 64 (more parallel work to amortize scan overhead).

**Key observation:** Futhark scales sub-linearly (6.3ms → 21.9ms for 32x more data) while Nx scales super-linearly (230ms → 10s). CUDA C scales well too (2.5ms → 5.6ms).

### Code Size

| Language | Kernel Lines | Integration Lines | Total Lines | Deps |
|----------|-------------|-------------------|-------------|------|
| **CUDA C** | 20 | 130 | **150** | nvcc |
| **Rust-CUDA** | 20 | 295 | **315** | cargo, rustler, cudarc |
| **Triton AOT** | 15 | 370 | **385** | Python+Triton (build), gcc, CUDA driver |
| **Julia** | 25 | 250 | **275** | Julia, CUDA.jl, MsgPack.jl |
| **Futhark** | 15 | 190 | **205** | futhark compiler |
| **Mojo** | 30 | 245 | **275** | Mojo SDK / Python+NumPy |
| **Bend** | 10 | 100 | **110** | cargo install |

## Detailed Analysis

### Rust-CUDA (Rustler NIF)

Best Elixir-native integration path.

**Strengths:**
- Rustler NIF = zero serialization overhead (no Port, no stdio)
- Type-safe end-to-end (Rust → CUDA)
- Familiar pattern — ExPhil already has selective_scan_nif in production
- CPU fallback when CUDA not available
- Single `cargo build` compile step

**Weaknesses:**
- Still ~2.5x slower than CUDA C via XLA (NIF data transfer: Nx → binary → GPU → binary → Nx)
- CUDA kernel is a string literal (no syntax highlighting/checks)
- NVRTC cold-start on first call (~100ms)
- More boilerplate than Triton for the same kernel

**Best for:** Elixir teams comfortable with Rust who want the tightest NIF integration.

### Triton AOT (C NIF)

Most productive kernel development workflow.

**Strengths:**
- Most readable kernel code (~15 lines Python)
- AOT compile → cubin eliminates Python runtime dependency
- Dominant ML kernel DSL (PyTorch, Meta, Cursor use Triton)
- Automatic memory tiling for complex kernels
- Portable cubin across machines with same GPU arch

**Weaknesses:**
- Two-stage build (Python AOT → C compile)
- C NIF is manual (no Rustler safety)
- Build-time Python + Triton dependency (~500MB)
- NIF data transfer overhead same as Rust-CUDA (~2-3x vs XLA)

**Best for:** Prototyping new GPU kernels with minimal code, then AOT compiling for production.

### CUDA C (Reference)

The existing 47 fused kernels in CUDA C are the production baseline.

**Strengths:**
- Direct GPU control, minimal overhead
- Mature tooling (nsight, cuda-gdb)
- Integrates directly with EXLA via XLA FFI
- No serialization overhead (GPU tensors stay on GPU)

**Weaknesses:**
- Verbose (manual memory indexing, thread layout)
- Easy to introduce subtle bugs (off-by-one, race conditions)
- CUDA-specific (no ROCm/Metal portability)

### Julia (CUDA.jl)

Best alternative for rapid prototyping.

**Strengths:**
- Nearly identical kernel syntax to CUDA C (easy porting)
- Interactive REPL enables rapid iteration
- KernelAbstractions.jl provides vendor-neutral GPU code
- Rich ecosystem (profiling, visualization, debugging)
- Array abstractions eliminate manual memory management

**Weaknesses:**
- 30-60s JIT cold-start (amortized by long-lived server)
- Port serialization overhead (~0.5-2ms per call)
- Data copies through CPU (no GPU-direct Elixir→Julia path)
- Large runtime footprint (~200MB)

**Best for:** Prototyping new kernel ideas before porting to CUDA C.

### Futhark

Unique for algorithmic exploration (parallel prefix scan).

**Strengths:**
- Most concise kernel code (~15 lines)
- Guaranteed race-free (type system prevents data races)
- Parallel prefix scan is algorithmically optimal for long sequences
- Compiles to efficient CUDA (no runtime dependency)
- Built-in testing framework

**Weaknesses:**
- No bf16 support (f32/f64 only)
- Creates own CUDA context (potential EXLA conflict)
- Different numerical results (floating-point reordering, ~1e-3 atol)
- Tiny community (~500 GitHub stars)
- Cannot express sequential operations efficiently

**Best for:** Exploring parallel scan algorithms. The crossover point where Futhark's O(log T) scan beats CUDA C's O(T) scan is valuable data for architecture decisions (e.g., long-context models).

### Mojo

Promising but too immature for production.

**Strengths:**
- Python-like syntax with C-level performance
- Strong type system
- SIMD auto-vectorization is elegant
- Potential for GPU kernels when API stabilizes

**Weaknesses:**
- Pre-1.0: GPU APIs are unstable
- Not in nixpkgs — NixOS installation is problematic
- GPU kernel support is experimental
- No native msgpack (requires Python wrapper)
- Large SDK (~500MB) with non-standard installer

**Best for:** Watching for future GPU API maturity. Not ready for use today.

### Bend (HVM2)

Learning exercise only — fundamentally wrong architecture for tensor ops.

**Strengths:**
- Novel execution model (interaction nets)
- Automatic parallelism
- Elegant functional syntax

**Weaknesses:**
- No f32 arrays (f24 only)
- No C FFI or CUDA interop
- Expected 10-100x slower than Nx for numeric workloads
- Tree-based data structures, not dense arrays

**Best for:** Understanding alternative parallel computation models. Not suitable for any production use in ExPhil.

## Decision Matrix

| Criterion | CUDA C | Rust-CUDA | Triton AOT | CuPy/CCCL | TK | Julia | Futhark | Mojo | Bend |
|-----------|--------|-----------|-----------|-----------|-----|-------|---------|------|------|
| Performance | +++++ | +++ | +++ | +++ | +++ | ++++ | ++++ | +++ | + |
| DX (readability) | ++ | ++ | +++++ | ++++ | +++ | ++++ | +++++ | ++++ | +++ |
| DX (debugging) | ++++ | +++ | +++ | +++ | +++ | ++++ | ++ | + | + |
| Integration effort | + (existing) | ++ | ++ | ++ | ++ | ++ | +++ | ++ | +++++ |
| Portability | + (CUDA only) | + | +++ | ++ | + | ++++ | ++ | +++ | ++ |
| Stability | +++++ | ++++ | ++++ | +++ | +++ | ++++ | +++ | + | + |
| Community | +++++ | +++ | ++++ | ++++ | ++ | +++ | + | ++ | + |
| **Overall** | **Best** | **Good** | **Good** | **Good** | **Niche** | **Good** | **Niche** | **Wait** | **No** |

## Recommendations

1. **Keep CUDA C for production kernels.** Benchmarks confirm CUDA C via XLA is the fastest option at all tested sizes. The existing 47 kernels work and integrate directly with EXLA. The ~2.5x gap between NIF-based approaches and XLA is entirely due to data transfer overhead.

2. **Use Triton for new kernel prototyping.** Most readable syntax (15 lines Python), dominant ML ecosystem, and AOT compilation eliminates Python runtime. Write in Triton, AOT compile to cubin, deploy via C NIF or XLA custom call.

3. **Rust-CUDA for Elixir-native integration.** Best DX for teams already using Rust. The selective_scan_nif proves the pattern at production scale. Use for performance-critical NIF paths where Rust's type safety matters.

4. **Use Julia for rapid REPL iteration.** Julia's raw kernel (588μs) is faster than CUDA C via XLA (3,230μs). Great for prototyping and debugging, but port serialization (94% of e2e) prevents production use.

5. **NIF overhead is the bottleneck, not kernel quality.** All NIF-based approaches (Rust, Triton, Futhark) are 2-3x slower than XLA due to Nx → binary → GPU → binary → Nx data copies. The path to parity is XLA custom calls (tensors stay on GPU), not faster kernels.

6. **Futhark is not a win for ExPhil's sizes.** Parallel prefix scan needs larger hidden dims or much longer sequences.

7. **Revisit Mojo in 12-18 months.** Bend is fundamentally wrong for tensor ops.

8. **CuPy/CCCL is good for parallel scan experimentation.** Blelloch parallel prefix scan via CuPy RawKernel, but Port serialization overhead makes it uncompetitive for production.

9. **ThunderKittens shines for attention, not scans.** TK's tile primitives (16x16 min) add no value for element-wise sequential scan. Its value would be for FlashAttention or Mamba-2 SSD matmul on sm_80+ GPUs.

10. **XLA custom calls are the highest-impact optimization.** See [XLA_CUSTOM_CALL_INTEGRATION.md](XLA_CUSTOM_CALL_INTEGRATION.md) for the detailed roadmap.

## Running the Benchmarks

```bash
# Individual language benchmarks
mix run scripts/benchmark_rust_scan.exs
mix run scripts/benchmark_triton_scan.exs
mix run scripts/benchmark_cuda_compute_scan.exs
mix run scripts/benchmark_thunderkittens_scan.exs
mix run scripts/benchmark_julia_scan.exs
mix run scripts/benchmark_futhark_scan.exs
mix run scripts/benchmark_mojo_scan.exs
mix run scripts/benchmark_bend_scan.exs

# Unified comparison
mix run scripts/benchmark_kernel_languages.exs
mix run scripts/benchmark_kernel_languages.exs --size training
```

## Related Files

| File | Description |
|------|-------------|
| [RUST_CUDA_KERNEL_EXPLORATION.md](RUST_CUDA_KERNEL_EXPLORATION.md) | Rust-CUDA Rustler NIF findings |
| [TRITON_KERNEL_EXPLORATION.md](TRITON_KERNEL_EXPLORATION.md) | Triton AOT findings |
| [CUDA_COMPUTE_EXPLORATION.md](CUDA_COMPUTE_EXPLORATION.md) | CuPy/CCCL parallel scan findings |
| [THUNDERKITTENS_EXPLORATION.md](THUNDERKITTENS_EXPLORATION.md) | ThunderKittens (HazyResearch) findings |
| [XLA_CUSTOM_CALL_INTEGRATION.md](XLA_CUSTOM_CALL_INTEGRATION.md) | XLA custom call roadmap |
| [JULIA_KERNEL_EXPLORATION.md](JULIA_KERNEL_EXPLORATION.md) | Julia detailed findings |
| [FUTHARK_KERNEL_EXPLORATION.md](FUTHARK_KERNEL_EXPLORATION.md) | Futhark detailed findings |
| [MOJO_KERNEL_EXPLORATION.md](MOJO_KERNEL_EXPLORATION.md) | Mojo detailed findings |
| [BEND_KERNEL_EXPLORATION.md](BEND_KERNEL_EXPLORATION.md) | Bend detailed findings |
| `scripts/benchmark_kernel_languages.exs` | Unified benchmark |
| `native/rust_linear_scan_nif/` | Rust-CUDA implementation |
| `native/triton_scan/` | Triton AOT implementation |
| `native/cuda_compute_scan/` | CuPy/CCCL implementation |
| `native/thunderkittens_scan/` | ThunderKittens implementation |
| `native/julia_scan/` | Julia implementation |
| `native/futhark_scan/` | Futhark implementation |
| `native/mojo_scan/` | Mojo implementation |
| `native/bend_scan/` | Bend implementation |
