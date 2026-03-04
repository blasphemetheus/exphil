# CUDA Compute / CCCL Exploration

## Overview

Explored NVIDIA's CCCL (CUDA C++ Core Libraries) Python bindings for implementing the `fused_linear_scan` kernel (`h = a*h + b`). CCCL provides optimized parallel algorithms (scan, reduce, sort) with custom types and operators. The key value proposition is defining the associative operator and letting NVIDIA's library handle the optimized parallel scan.

## Approach

Since `cuda-cccl` Python bindings (v0.5.1) are still experimental, we used **CuPy** as a stable GPU computing platform with custom CUDA kernels via `RawKernel`. This provides:

1. **Sequential scan** — Same algorithm as our CUDA C kernel, one thread per (batch, hidden), sequential loop over timesteps
2. **Parallel prefix scan** — Blelloch algorithm with the associative operator `(a1,b1)*(a2,b2) = (a1*a2, a1*b2+b1)`, O(log T) depth

## Integration Architecture

```
┌─────────────┐     msgpack/stdio    ┌──────────────────┐
│   Elixir    │◄───────────────────►│     Python        │
│  GenServer  │   Port ({packet,4})  │  CuPy RawKernel  │
│  (Port)     │                      │  CUDA GPU         │
└─────────────┘                      └──────────────────┘
```

Same Port-based pattern as Mojo/Julia. Tensor data serialized as f32 binary via msgpack.

## Files

| File | Description |
|------|-------------|
| `native/cuda_compute_scan/server.py` | Python server with CuPy sequential + parallel kernels |
| `lib/exphil/bridge/cuda_compute_port.ex` | Elixir GenServer Port wrapper |
| `scripts/benchmark_cuda_compute_scan.exs` | Multi-seq_len benchmark |

## The Parallel Prefix Scan

The linear recurrence `h[t] = a[t] * h[t-1] + b[t]` can be expressed as a scan with associative operator:

```
Operator: (a1, b1) ⊕ (a2, b2) = (a1·a2, a2·b1 + b2)
Identity: (1.0, 0.0)
```

This transforms the sequential O(T) recurrence into an O(log T) parallel scan (Blelloch algorithm), at the cost of 2x more arithmetic operations.

### When does parallel beat sequential?

The parallel scan wins when:
- **seq_len is large** (more parallelism to exploit)
- **hidden_dim is small** (sequential scan already saturates GPU with batch × hidden threads)
- **GPU has many SMs** (more parallel units to keep busy)

For ExPhil's typical sizes (batch=4, seq_len=60, hidden=64), sequential scan creates only 256 threads — the GPU is underutilized. The parallel scan could help by exposing more parallelism, but the Blelloch algorithm's shared memory requirements and synchronization overhead may negate this.

## Key Findings

### Performance Expectations

| Integration | Overhead Source | Expected Impact |
|-------------|----------------|-----------------|
| Port (msgpack) | Serialization + stdio | ~3-5ms per call |
| CuPy data copy | NumPy → GPU → NumPy | ~1-2ms per call |
| CUDA kernel | Actual computation | ~0.5-3ms depending on size |
| **Total e2e** | | **~5-10ms** (vs 3.2ms CUDA C via XLA) |

Port + CuPy overhead is expected to make this ~2-3x slower than CUDA C via XLA, similar to all other Port-based approaches.

### cuda-cccl Python API (Experimental)

When `cuda-cccl` stabilizes, the kernel reduces to ~10 lines:

```python
from cuda.parallel.experimental import algorithms as algos

# Define associative operator
@algos.scan.binary_op
def linear_scan_op(left, right):
    return (left[0] * right[0], right[0] * left[1] + right[1])

# Run parallel scan
result = algos.inclusive_scan(pairs, linear_scan_op)
```

This is the "least custom code" approach — NVIDIA handles all the parallel scan optimization internally, including work-efficient Blelloch algorithm, bank-conflict avoidance, and multi-SM load balancing.

### CuPy RawKernel Approach (Stable)

The CuPy RawKernel approach writes the CUDA kernel as a string (same as cudarc NVRTC in Rust-CUDA), compiled at runtime:

```python
kernel = cp.RawKernel(CUDA_SOURCE, "kernel_name")
kernel((grid,), (block,), (args...))
```

This gives full CUDA control with CuPy's array management handling GPU memory allocation and transfer.

## Comparison with Other Approaches

| Aspect | CUDA C (XLA) | CuPy/CCCL (Port) | Rust-CUDA (NIF) |
|--------|-------------|-------------------|-----------------|
| Data path | GPU → GPU (zero-copy) | CPU → Port → GPU → Port → CPU | CPU → NIF → GPU → NIF → CPU |
| Kernel quality | Identical | Identical | Identical |
| Overhead | XLA dispatch only | Port + CuPy + GPU copy | NIF + GPU copy |
| Expected speed | ~3ms | ~8-10ms | ~8.5ms |
| Code complexity | Low (XLA CC) | Medium (Port + Python) | Medium (Rustler) |

## Recommendations

1. **Port-based CuPy adds too much overhead for production** — same conclusion as Mojo/Julia. The kernel is fine; the data path is the bottleneck.

2. **cuda-cccl is worth watching** — when the Python API stabilizes, it becomes the lowest-code-effort way to write custom parallel scan operators. Could be combined with XLA custom calls for zero-copy GPU integration.

3. **The parallel prefix scan is algorithmically interesting** — O(log T) depth vs O(T), but only wins at very long sequences or with highly parallel hardware.

## Running

```bash
# Install CuPy
pip install cupy-cuda12x  # or add to shell.nix

# Run benchmark
mix run scripts/benchmark_cuda_compute_scan.exs
mix run scripts/benchmark_cuda_compute_scan.exs --seq-lengths 30,60,120,240,480,960
```
