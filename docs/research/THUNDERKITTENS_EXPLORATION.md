# ThunderKittens Exploration

## Overview

Explored ThunderKittens (Stanford HazyResearch) as a kernel development framework for ExPhil's SSM recurrences. ThunderKittens is a CUDA-embedded C++ DSL providing tile-level abstractions over tensor cores, shared memory, and warp-level primitives.

**Key finding:** ThunderKittens requires sm_80+ (Ampere) GPUs and is designed for tile-parallel operations (attention, matmul). For the simple sequential scan `h = a*h + b`, TK provides no advantage. Its value would be for fused attention kernels in Mamba-2/Jamba architectures.

## What Is ThunderKittens?

ThunderKittens (v2.0, Feb 2026) provides:
- **Register tiles** (rt): 16x16 or larger tiles stored in registers across a warp
- **Shared memory tiles** (st): Tiles for inter-warp communication
- **Global memory tiles** (gt): Async loads from DRAM with hardware prefetching
- **Tile operations**: load, store, mma (tensor core matmul), reduce, map
- **Warp group abstractions**: Coordinate multiple warps for producer-consumer patterns

Used in production at Cursor, Together AI for attention and MoE kernels.

## Why TK Doesn't Help for fused_linear_scan

The `fused_linear_scan` kernel (`h[t] = a[t]*h[t-1] + b[t]`) is:
1. **Sequential over time** — each timestep depends on the previous
2. **Element-wise** — no matmul or tensor core operations
3. **Simple memory access** — linear reads, no tiling needed

TK's primitives (tile matmul, shared memory management, tensor cores) are irrelevant for this kernel. The sequential scan naturally maps to one CUDA thread per (batch, hidden) pair.

## Where TK Would Help in ExPhil

TK shines for operations that ExPhil uses in attention-based backbones:

| Operation | TK Advantage | ExPhil Context |
|-----------|-------------|----------------|
| **Multi-head attention** | Fused QKV matmul + softmax + V multiply | Attention, Jamba, Zamba |
| **Mamba-2 SSD matmul** | Fused chunk-wise associative scan + matmul | Mamba-2 backbone |
| **Causal attention mask** | Hardware-accelerated mask application | All attention variants |
| **FlashAttention** | Memory-efficient attention with tiling | Future FlashAttn custom call |

## Integration Architecture

```
native/thunderkittens_scan/
├── kernel.cu                     # CUDA kernel (.cu, compiled with nvcc)
├── thunderkittens_scan_nif.c     # C NIF (loads .so via dlopen)
└── Makefile                      # Two-stage build

The kernel is compiled to a shared library (libthunderkittens_kernel.so)
and loaded by the C NIF via dlopen/dlsym at runtime.
```

### Build Pattern

```bash
# Standard build (sm_75, CUDA fallback kernel)
cd native/thunderkittens_scan
make && make install

# With ThunderKittens headers (sm_80+)
make tk TK_PATH=/path/to/ThunderKittens CUDA_ARCH=sm_80
```

## Hardware Requirements

| GPU | Compute Capability | TK Support |
|-----|-------------------|------------|
| T400 (Turing) | sm_75 | No — CUDA fallback only |
| A100 (Ampere) | sm_80 | Yes — full TK support |
| RTX 3090 | sm_86 | Yes |
| RTX 4090 (Ada) | sm_89 | Yes |
| H100 (Hopper) | sm_90 | Yes (advanced features) |
| B200 (Blackwell) | sm_100 | Yes (cuTile integration) |

Our T400 (sm_75) cannot use ThunderKittens. RunPod A100 instances can.

## Files

| File | Description |
|------|-------------|
| `native/thunderkittens_scan/kernel.cu` | CUDA kernel with TK fallback |
| `native/thunderkittens_scan/thunderkittens_scan_nif.c` | C NIF via dlopen |
| `native/thunderkittens_scan/Makefile` | Two-stage build |
| `lib/exphil/native/thunderkittens_scan.ex` | Elixir NIF wrapper |
| `scripts/benchmark_thunderkittens_scan.exs` | Multi-seq_len benchmark |

## Performance Expectations

Since TK uses the same CUDA kernel as our reference for `fused_linear_scan`, performance is determined by NIF overhead:

| Integration | Expected e2e | vs CUDA C (XLA) |
|-------------|-------------|-----------------|
| TK NIF (sm_75 fallback) | ~8-10ms | ~2.5-3x |
| TK NIF (sm_80+ native) | ~8-10ms | ~2.5-3x (same kernel for simple scan) |
| CUDA C via XLA | ~3ms | 1.0x |

The NIF data transfer overhead (Nx → binary → GPU → binary → Nx) dominates, not kernel quality.

## Recommendations

1. **Don't use TK for `fused_linear_scan`** — no benefit over raw CUDA C for element-wise sequential scan.

2. **Consider TK for FlashAttention custom call** — when implementing FlashAttention as an XLA custom call, TK's tile abstractions would reduce the kernel code from ~500 lines (raw CUDA) to ~50-100 lines with automatic tensor core usage.

3. **Consider TK for Mamba-2 SSD matmul** — the chunk-wise matmul in `Edifice.CUDA.MambaSSD` could benefit from TK's fused matmul primitives on A100+ GPUs.

4. **Requires A100+ hardware** — only testable on RunPod or similar cloud GPU instances.

## Running

```bash
# Build (uses CUDA fallback on sm_75)
cd native/thunderkittens_scan
make && make install

# Benchmark
mix run scripts/benchmark_thunderkittens_scan.exs
mix run scripts/benchmark_thunderkittens_scan.exs --batch 32 --hidden 512
```
