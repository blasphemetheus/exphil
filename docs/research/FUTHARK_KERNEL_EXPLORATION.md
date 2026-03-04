# Futhark Kernel Exploration

## Overview

Evaluation of Futhark for writing GPU kernels, specifically its parallel prefix scan primitive for the linear recurrence `h = a*h + b`.

**Key insight:** Futhark's `scan` primitive generates a Blelloch-style parallel prefix scan — fundamentally different from CUDA C's sequential per-thread approach. The linear recurrence `h = a*h + b` forms a monoid under the composition operator `(a1, b1) ∘ (a2, b2) = (a1*a2, a2*b1 + b2)`, enabling O(log T) parallel depth.

**Integration:** NIF via compiled C library, same pattern as Edifice's `edifice_cuda_nif.c`.

## Implementation

### Files

| File | Purpose |
|------|---------|
| `native/futhark_scan/linear_scan.fut` | Futhark source (~30 lines) |
| `native/futhark_scan/futhark_scan_nif.c` | C NIF wrapper |
| `native/futhark_scan/Makefile` | Build: futhark → C → .so |
| `lib/exphil/native/futhark_scan.ex` | Elixir NIF module |
| `scripts/benchmark_futhark_scan.exs` | Multi-seq_len benchmark |

### Kernel Code

The entire Futhark kernel is ~15 lines:

```futhark
let combine (a1, b1) (a2, b2) = (a1 * a2, a2 * b1 + b2)

let linear_scan_1d [n] (a_seq: [n]f32) (b_seq: [n]f32) (h0_val: f32): [n]f32 =
  let b_seq'[0] = a_seq[0] * h0_val + b_seq[0]
  let scanned = scan combine (1.0f32, 0.0f32) (zip a_seq b_seq')
  in map (\(_a, b) -> b) scanned
```

Compare to 20+ lines of CUDA C for the same operation.

### Algorithm Comparison

| Aspect | CUDA C (sequential) | Futhark (parallel prefix) |
|--------|---------------------|---------------------------|
| Depth | O(T) | O(log T) |
| Work | O(T) | O(2T) |
| Parallelism | batch × hidden | batch × hidden × log(T) |
| Best for | Short sequences (T < 256) | Long sequences (T > 256) |
| FP precision | Exact sequential order | Non-deterministic (reordering) |

### Code Size

| Component | CUDA C | Futhark |
|-----------|--------|---------|
| Kernel | ~20 lines | ~15 lines |
| Build system | nvcc flags | Makefile + futhark cmd |
| NIF wrapper | ~50 lines | ~130 lines |
| Elixir module | ~50 lines | ~60 lines |
| **Total** | ~120 lines | ~205 lines |

## Setup

```bash
# Add futhark to shell.nix, then:
cd native/futhark_scan
make          # Compile .fut → .c → .so
make install  # Copy to priv/native/
make test     # Run Futhark's built-in tests

# Benchmark
mix run scripts/benchmark_futhark_scan.exs
```

## Results

Benchmarked on NVIDIA T400 4GB, NixOS/WSL2, batch=4, hidden=64. 5 warmup, 30 timed, median.

### Correctness

| Metric | Value |
|--------|-------|
| Max absolute difference vs sequential | ~1e-3 (floating-point reordering from parallel scan) |
| Acceptable for ML training | Yes (gradient noise is larger) |

### Latency at Multiple Sequence Lengths

| seq_len | Futhark (μs) | CUDA C (μs) | Nx CPU (μs) | Futhark/CUDA C | Winner |
|---------|-------------|-------------|-------------|----------------|--------|
| 30 | 6,338 | 2,485 | 229,854 | 2.55x | CUDA C |
| 60 | 6,914 | 3,613 | 525,719 | 1.91x | CUDA C |
| 120 | 7,693 | 3,660 | 1,056,859 | 2.10x | CUDA C |
| 240 | 9,799 | 3,990 | 2,081,286 | 2.46x | CUDA C |
| 480 | 8,823 | 2,580 | 2,068,732 | 3.42x | CUDA C |
| 960 | 21,899 | 5,563 | 9,984,523 | 3.94x | CUDA C |

### Crossover Analysis

Futhark's parallel scan does 2x total work but in O(log T) depth. **The crossover was not reached** at these sizes. CUDA C won at all tested sequence lengths.

Key observations:
- Futhark's overhead is ~4ms constant (context setup, kernel launch) — dominates at small seq_len
- Futhark scales sub-linearly: 6.3ms → 21.9ms for 32x more data (3.5x increase)
- CUDA C also scales well: 2.5ms → 5.6ms for 32x more data (2.2x increase)
- Nx (CPU) scales super-linearly: 230ms → 10s (43x increase)
- The gap *widened* at larger seq_len (1.91x at 60 → 3.94x at 960)

**Why no crossover?** With hidden=64, each thread in CUDA C does 64 parallel sequential scans — very efficient. Futhark's parallel prefix scan needs the *time* dimension to be the bottleneck, which requires hidden >> batch (so there's not enough batch-level parallelism for CUDA C). Crossover likely requires hidden_dim > 256 and seq_len > 2048.

## Assessment

### Pros
- Extremely concise: entire parallel scan in ~15 lines
- Guaranteed race-free: Futhark's type system prevents data races
- Compiles to efficient CUDA via Futhark compiler
- No runtime dependency (compiles to C + CUDA)
- Parallel prefix scan is algorithmically optimal for long sequences
- Built-in testing framework

### Cons
- No bf16 support (f32 and f64 only)
- Creates its own CUDA context (may conflict with EXLA)
- Parallel scan has different numerical results (floating-point non-associativity)
- Tiny community (~500 GitHub stars) — limited support for edge cases
- Cannot express sequential operations efficiently (always parallel)
- Build requires Futhark compiler (Haskell binary, ~50MB)

### Verdict

**Not a win for ExPhil's current sizes.** CUDA C is 2-4x faster at all tested sequence lengths (30-960) with batch=4, hidden=64. Futhark's parallel prefix scan is algorithmically elegant (15 lines vs 20 for CUDA C) and provably race-free, but the 2x work overhead and CUDA context setup cost outweigh the O(log T) depth advantage at these sizes.

**Worth revisiting if:** ExPhil moves to hidden_dim > 256 and seq_len > 2048 (e.g., long-context Mamba variants or full-game sequence modeling).
