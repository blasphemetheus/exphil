# Bend (HVM2) Kernel Exploration

## Overview

Evaluation of Bend for GPU computation — a learning exercise to understand its interaction net execution model, NOT a practical kernel replacement.

**Kernel:** `fused_linear_scan` — `h = a*h + b` over timesteps, expressed as a fold.

**Integration:** Standalone binary (no C FFI, no CUDA interop).

## Implementation

### Files

| File | Purpose |
|------|---------|
| `native/bend_scan/linear_scan.bend` | Bend source (~80 lines) |
| `native/bend_scan/test.sh` | Run tests across backends |
| `lib/exphil/bridge/bend_port.ex` | Simple CLI wrapper |
| `scripts/benchmark_bend_scan.exs` | Exploration script |

### Kernel Code

```bend
def linear_scan(h_state, a_seq, b_seq):
  match a_seq:
    case List/Cons:
      match b_seq:
        case List/Cons:
          h_new = (a_seq.head * h_state) + b_seq.head
          rest = linear_scan(h_new, a_seq.tail, b_seq.tail)
          return List/Cons { head: h_new, tail: rest }
        case List/Nil:
          return List/Nil
    case List/Nil:
      return List/Nil
```

This is purely functional — the recurrence is expressed as a recursive function over linked lists. Bend automatically determines what can be parallelized (in this case, very little — the scan is inherently sequential).

**Implementation note:** Bend 0.2.x doesn't support `let` destructuring inside `fold`/`match`, and HVM2's linearity checker rejects shared data across branches. The final implementation uses separate `a_seq`/`b_seq` lists instead of a zipped pairs list.

### Code Size

| Component | Lines |
|-----------|-------|
| Bend kernel + helpers | ~80 |
| Test script | ~40 |
| Elixir wrapper | ~100 |
| **Total** | ~220 |

Extremely concise for the kernel itself (~10 lines for the core scan), but the lack of binary I/O means no practical integration.

## Setup

```bash
# Install Bend (requires Rust toolchain, already in shell.nix)
cargo install bend-lang

# Test
cd native/bend_scan && bash test.sh

# Explore
mix run scripts/benchmark_bend_scan.exs
```

## Results

Benchmarked on NixOS/WSL2, small test size (4 elements, not full tensor — Bend has no array I/O).

| Metric | Value |
|--------|-------|
| Correctness | Verified: [1.000, 1.900, 2.710, 3.439] matches sequential |
| Bend (Rust backend) | 119,185 μs |
| Bend (C backend) | Correct results, similar speed |
| Bend (CUDA backend) | Failed (expected — HVM2 GPU is experimental) |
| Nx reference | 4,864 μs |
| Bend/Nx ratio | **24.5x slower** |

**Note:** This comparison is generous to Bend — it only processes 4 scalar values while Nx processes a full 4×30×64 tensor. The per-element overhead of HVM2's interaction net evaluation is orders of magnitude higher than dense array ops.

## Execution Model

### How Bend differs from CUDA

| Aspect | CUDA C | Bend (HVM2) |
|--------|--------|-------------|
| Parallelism model | Explicit threads/blocks | Automatic (interaction nets) |
| Memory model | Shared/global/register | Graph reduction |
| Data types | IEEE 754 f32/f64 | f24 (24-bit float) |
| GPU execution | CUDA kernels | Interaction net evaluation |
| Best for | Dense tensors | Tree-structured recursion |
| Worst for | Highly sequential | Dense array operations |

### Why Bend is wrong for scan operations

1. **No dense arrays**: Bend represents data as trees/lists, not contiguous memory. Every array element is a linked list node — O(n) access vs O(1).

2. **f24 precision**: 24-bit floats lose significant precision for ML workloads. No f32 or f64 support.

3. **Overhead**: HVM2's interaction net evaluation has high per-operation overhead compared to raw CUDA warps.

4. **No tensor parallelism**: Bend parallelizes tree reductions, not dense element-wise operations. A scan over a list is inherently sequential in any model.

### Where Bend might shine

- **Tree-structured algorithms**: Monte Carlo tree search, game tree evaluation
- **Symbolic computation**: Term rewriting, lambda calculus evaluation
- **Massively recursive programs**: Fractal generation, parallel parsing

These don't overlap with ExPhil's tensor computation needs.

## Assessment

### Pros
- Elegant functional syntax
- Automatic parallelism (no manual thread management)
- Novel execution model worth understanding
- Easy to install (single cargo install)

### Cons
- No f32 arrays (f24 only)
- No C FFI or CUDA interop
- No binary tensor I/O
- Expected 10-100x slower than Nx for numeric workloads
- HVM2 GPU backend is experimental
- Tiny community, minimal documentation

### Verdict

**Not suitable for GPU kernel development in ExPhil.** Bend's interaction net model is optimized for tree-structured parallelism, not dense tensor operations. Valuable as a learning exercise to understand alternative parallel computation models.

The key takeaway: automatic parallelism sounds appealing but requires the right data structures (trees, not arrays) and the right operations (independent subproblems, not sequential scans).
