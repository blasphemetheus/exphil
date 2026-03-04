# Triton AOT Kernel Exploration

## Overview

Evaluated OpenAI's Triton as a GPU kernel DSL for ExPhil. Triton compiles Python to Triton IR → LLVM IR → PTX → cubin. We use **AOT (ahead-of-time) compilation** to produce a cubin at build time, then load it at runtime from a C NIF via CUDA driver API. **Zero Python dependency at runtime.**

**Test kernel:** `fused_linear_scan` — `h = a*h + b` (same kernel used across all language comparisons)

## Architecture

```
[Build time]  Python + Triton → cubin (compile_aot.py)
[Runtime]     Elixir → C NIF → cuModuleLoadData(cubin) → cuLaunchKernel → result
```

Key insight: Triton is a **build tool**, not a runtime dependency. The AOT-compiled cubin is a standard CUDA binary loaded via the driver API, just like NVRTC-compiled PTX.

## Implementation

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `native/triton_scan/linear_scan_kernel.py` | ~40 | Triton kernel (`@triton.jit`) |
| `native/triton_scan/compile_aot.py` | ~90 | AOT compile to cubin + metadata header |
| `native/triton_scan/triton_scan_nif.c` | ~230 | C NIF: loads cubin, launches kernel |
| `native/triton_scan/linear_scan_kernel_meta.h` | ~7 | Auto-generated: kernel name, num_warps, shared_mem |
| `native/triton_scan/Makefile` | ~40 | Build pipeline |
| `lib/exphil/native/triton_scan.ex` | ~90 | Elixir NIF wrapper |
| `scripts/benchmark_triton_scan.exs` | ~200 | Multi-backend benchmark |

### Triton Kernel (15 lines of actual logic)

```python
@triton.jit
def linear_scan_kernel(a_ptr, b_ptr, h0_ptr, output_ptr, batch, seq_len, hidden):
    pid = tl.program_id(0)
    batch_idx = pid // hidden
    hidden_idx = pid % hidden

    h_state = tl.load(h0_ptr + batch_idx * hidden + hidden_idx)

    for t in range(seq_len):
        idx = batch_idx * seq_len * hidden + t * hidden + hidden_idx
        a_t = tl.load(a_ptr + idx)
        b_t = tl.load(b_ptr + idx)
        h_state = a_t * h_state + b_t
        tl.store(output_ptr + idx, h_state)
```

Compare with CUDA C (same logic, more boilerplate):
```c
extern "C" __global__ void fused_linear_scan_kernel(...) {
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

### AOT Compilation

```bash
python3 compile_aot.py --sm 75   # T400 Turing
# Output: linear_scan_kernel.cubin (7912 bytes)
#         linear_scan_kernel_meta.h (kernel name, num_warps, shared_mem)
```

Uses `triton.compiler.compile()` with `GPUTarget("cuda", 75, 32)` — no torch dependency needed for AOT.

### C NIF Loading

The C NIF uses CUDA driver API to load the cubin:
1. `cuModuleLoadData(&module, cubin_data)` — load cubin into CUDA context
2. `cuModuleGetFunction(&function, module, "linear_scan_kernel")` — get kernel handle
3. `cuLaunchKernel(function, grid, block, ...)` — launch kernel

Grid layout: one block per program (batch × hidden programs), each block has `num_warps × 32` threads (managed by Triton).

## Building

```bash
cd native/triton_scan

# Full build (needs Python + Triton for AOT step)
make && make install

# NIF-only build (cubin already compiled)
make nif && make install
```

### NixOS Setup

Triton is available in nixpkgs as `python3Packages.triton`. CUDA toolkit must be in PATH for ptxas:

```nix
# In shell.nix buildInputs:
(python3.withPackages (ps: with ps; [ msgpack numpy triton ]))  # For AOT compilation
cudaPackages.cudatoolkit  # For ptxas (already present)
```

## Benchmark Results

See `scripts/benchmark_triton_scan.exs` output. Expected performance:
- Triton AOT ≈ Rust-CUDA NIF (both do Nx → binary → GPU → binary → Nx)
- Both ~2-3x slower than CUDA C (XLA) which keeps tensors on GPU
- Both ~50-100x faster than pure Nx CPU

The overhead is **not from the kernel** (Triton generates efficient PTX) but from the NIF data transfer pattern. XLA custom calls avoid this by keeping tensors in GPU memory.

## Developer Experience

**Strengths:**
- Most readable kernel code (Python syntax, 15 lines)
- AOT compilation eliminates Python runtime dependency
- Dominant ML kernel DSL (PyTorch, Meta, Cursor, Together AI use Triton)
- Automatic memory tiling for complex kernels (less relevant for sequential scan)
- cubin is portable across machines with same GPU architecture

**Weaknesses:**
- Build-time Python + Triton dependency (~500MB)
- C NIF is manual (no Rustler-level safety or convenience)
- Two-stage build (AOT Python → C compile) vs single cargo build for Rust
- Less control over thread layout than raw CUDA C
- nixpkgs Triton requires CUDA toolkit in PATH for ptxas

## Comparison with Rust-CUDA

| Aspect | Triton AOT | Rust-CUDA (Rustler) |
|--------|-----------|-------------------|
| Kernel language | Python | CUDA C string in Rust |
| Kernel lines | ~15 | ~20 |
| Integration | C NIF | Rustler NIF |
| Build tool | Python + Triton + gcc | cargo |
| Runtime deps | libcuda.so only | libcuda.so only |
| Type safety | None (Python) | Rust (NIF layer) |
| Debugging | Triton autotuner | CUDA-GDB via cudarc |
| DX for new kernels | Best (Python + auto-tune) | Good (familiar to Rust devs) |

## Verdict

**Best kernel DX for new kernels.** Triton's Python DSL is the most readable and productive way to write GPU kernels. AOT compilation eliminates the runtime dependency, making it production-viable.

For ExPhil's Elixir pipeline, the integration path is:
1. **Prototype** kernels in Triton (fast iteration in Python)
2. **AOT compile** to cubin (one-time build step)
3. **Load** via C NIF (zero Python at runtime)

The main limitation is the same as Rust-CUDA: NIF data transfer overhead. For production use, both Triton and Rust-CUDA kernels should be integrated via **XLA custom calls** (keeping tensors on GPU) rather than NIFs.
