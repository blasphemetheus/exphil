# Custom GPU Kernels for Mamba SSM

This document outlines our options and plan for implementing custom GPU kernels to achieve 60 FPS inference for the Mamba selective scan.

## Current Status

| Implementation | Inference Time | 60 FPS? | Notes |
|----------------|----------------|---------|-------|
| Blelloch (Nx/XLA) | ~55ms | No | Current best in pure Elixir |
| Hillis-Steele (Nx/XLA) | ~56ms | No | Slightly slower |
| Cumsum (Nx/XLA) | ~2500ms | No | XLA cumsum is pathologically slow |
| SSD (Nx/XLA) | ~800ms | No | Too many separate XLA ops |
| ONNX INT8 | ~0.5ms | **Yes** | Requires export, less flexible |
| **Custom Triton** | TBD | Target | Goal of this effort |

**Target:** <16.67ms for 60 FPS real-time play

---

## Options Analysis

### Option 1: Triton Kernel (Recommended)

**What:** Python DSL that compiles to optimized PTX/CUBIN

**Pros:**
- 10x less code than raw CUDA
- Automatic memory coalescing and tiling
- `tl.associative_scan` built-in for parallel scans
- Easy to iterate and experiment
- Works on all NVIDIA GPUs (including 4090)

**Cons:**
- Python dependency at runtime (or compile to cubin)
- Learning curve for Triton-specific patterns

**Effort:** 1-2 days for working kernel, 1 week for optimized

**Files:**
- `priv/triton/selective_scan.py` - Kernel implementation
- `lib/exphil/bridge/triton_port.ex` - Elixir Port wrapper

### Option 2: Custom XLA Operation

**What:** C++/CUDA code registered as XLA CustomCall

**Pros:**
- Seamless EXLA integration (no Python at runtime)
- Can be distributed as part of the package
- Best long-term solution

**Cons:**
- Complex build setup (CUDA toolkit, XLA headers)
- Harder to debug and iterate
- Must match XLA's memory layout expectations

**Effort:** 3-5 days

**Files:**
```
native/
├── selective_scan.cu       # CUDA kernel
├── selective_scan_xla.cc   # XLA CustomCall wrapper
├── CMakeLists.txt
└── Makefile
```

### Option 3: Rust NIF with cudarc (Also Implemented)

**What:** Rust FFI to CUDA via cudarc crate

**Pros:**
- Memory-safe CUDA wrapper
- No Python dependency at runtime
- Compiles into release (single binary)
- Good Elixir integration via Rustler
- Runtime PTX compilation via NVRTC

**Cons:**
- Requires Rust toolchain + CUDA toolkit to build
- cudarc is less mature than direct CUDA

**Effort:** 2-3 days

**Files (created):**
```
native/selective_scan_nif/
├── Cargo.toml              # Rust dependencies
├── src/
│   ├── lib.rs              # NIF entry points
│   └── kernel.rs           # CUDA kernel + cudarc interface
lib/exphil/native/
└── selective_scan.ex       # Elixir wrapper
```

**Building:**
```bash
cd native/selective_scan_nif
cargo build --release --features cuda
cp target/release/libselective_scan_nif.so ../../priv/native/
```

**Usage:**
```elixir
alias ExPhil.Native.SelectiveScan

if SelectiveScan.available?() and SelectiveScan.cuda_available?() do
  result = SelectiveScan.scan(x, dt, a, b, c)
end
```

### Option 4: Pure CUDA with NIF

**What:** Raw CUDA kernel called via Erlang NIF

**Pros:**
- Maximum control and performance
- No Python dependency

**Cons:**
- Most complex option
- Manual memory management
- Build complexity

**Effort:** 1 week+

---

## Recommended Plan

```
Phase 1: Triton Prototype (Current)
├── Write Triton kernel                     ✅ priv/triton/selective_scan.py
├── Benchmark on GPU                        ⬜ Run python selective_scan.py
├── Verify correctness                      ⬜ Run python selective_scan.py --test
└── Measure speedup vs XLA                  ⬜ Compare to Nx benchmarks

Phase 2: Elixir Integration
├── Create Port-based bridge                ⬜ lib/exphil/bridge/triton_port.ex
├── Add to MambaTriton module               ⬜ lib/exphil/networks/mamba_triton.ex
├── Benchmark end-to-end                    ⬜ Add to benchmark scripts
└── Test in training loop                   ⬜ Verify gradients work

Phase 3: Optimization
├── Profile with Triton autotuner           ⬜ Find optimal block sizes
├── Add backward pass kernel                ⬜ For training
├── Tune for 4090 specifically              ⬜ Ada Lovelace optimizations
└── Consider tensor core usage              ⬜ FP16/BF16 paths

Phase 4: Production (Optional)
├── Compile to standalone cubin             ⬜ Remove Python dependency
├── Or: Port to custom XLA op               ⬜ Best long-term solution
└── Package for distribution                ⬜ Hex package with native code
```

---

## Phase 1: Triton Kernel

### Running the Kernel

```bash
# On GPU pod
cd /app

# Install Triton (if not present)
pip install triton torch

# Run benchmark
python priv/triton/selective_scan.py

# Run correctness test
python priv/triton/selective_scan.py --test
```

### Expected Output

```
============================================================
Selective Scan Benchmark
============================================================

Config: batch=32, seq=60, hidden=512, state=16
Device: cuda

Benchmarking PyTorch reference...
  Reference: 45.23 ms
Benchmarking Triton kernel...
  Triton: 2.15 ms
  Speedup: 21.0x

60 FPS threshold: 16.67 ms
Reference meets 60 FPS: NO
Triton meets 60 FPS: YES  ← Goal
```

### Kernel Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton Kernel Grid                        │
│                                                              │
│  grid = (batch, hidden)                                      │
│  Each block handles one (batch_idx, hidden_idx) pair         │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Block (b, h)                                         │    │
│  │                                                      │    │
│  │   for t in 0..seq_len:                              │    │
│  │     1. Load x[b,t,h], dt[b,t,h], B[b,t,:], C[b,t,:] │    │
│  │     2. Discretize: A_bar = exp(dt * A)              │    │
│  │     3. Recurrence: h = A_bar * h + B_bar * x        │    │
│  │     4. Output: y[b,t,h] = sum(C * h)                │    │
│  │                                                      │    │
│  │   State h lives in registers (fast!)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why Triton is Faster

1. **Fused Operations:** Discretization + scan + output in one kernel
2. **Register State:** Hidden state `h` lives in registers, not global memory
3. **Coalesced Loads:** Triton handles memory coalescing automatically
4. **No Python Overhead:** Compiled to PTX, runs natively on GPU
5. **No XLA Dispatch:** Single kernel launch vs many separate ops

---

## Phase 2: Elixir Integration

### Port-Based Bridge

```elixir
# lib/exphil/bridge/triton_port.ex
defmodule ExPhil.Bridge.TritonPort do
  use GenServer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def selective_scan(x, dt, a, b, c) do
    GenServer.call(__MODULE__, {:scan, x, dt, a, b, c}, :infinity)
  end

  @impl true
  def init(_opts) do
    python_path = Application.app_dir(:exphil, "priv/triton/selective_scan.py")
    port = Port.open({:spawn, "python3 #{python_path} --serve"}, [
      :binary,
      :exit_status,
      {:packet, 4}
    ])
    {:ok, %{port: port}}
  end

  @impl true
  def handle_call({:scan, x, dt, a, b, c}, _from, state) do
    # Serialize tensors to binary
    msg = %{
      op: "scan",
      x: Nx.to_binary(x), x_shape: Nx.shape(x) |> Tuple.to_list(),
      dt: Nx.to_binary(dt), dt_shape: Nx.shape(dt) |> Tuple.to_list(),
      A: Nx.to_binary(a), A_shape: Nx.shape(a) |> Tuple.to_list(),
      B: Nx.to_binary(b), B_shape: Nx.shape(b) |> Tuple.to_list(),
      C: Nx.to_binary(c), C_shape: Nx.shape(c) |> Tuple.to_list()
    }

    Port.command(state.port, Msgpax.pack!(msg))

    receive do
      {^state.port, {:data, data}} ->
        response = Msgpax.unpack!(data)
        result = Nx.from_binary(response["result"], :f32)
                 |> Nx.reshape(List.to_tuple(response["shape"]))
        {:reply, {:ok, result}, state}
    after
      30_000 -> {:reply, {:error, :timeout}, state}
    end
  end
end
```

### MambaTriton Module

```elixir
# lib/exphil/networks/mamba_triton.ex
defmodule ExPhil.Networks.MambaTriton do
  @moduledoc """
  Mamba using custom Triton kernel for the selective scan.

  Requires Python with Triton installed.
  Falls back to Blelloch scan if Triton unavailable.
  """

  alias ExPhil.Bridge.TritonPort

  def build(opts) do
    # Same architecture as regular Mamba, but uses Triton for SSM
    # ...
  end

  def selective_scan(x, dt, a, b, c) do
    case TritonPort.selective_scan(x, dt, a, b, c) do
      {:ok, result} -> result
      {:error, _} ->
        # Fallback to Nx implementation
        ExPhil.Networks.Mamba.selective_scan(x, dt, a, b, c)
    end
  end
end
```

---

## Hardware: NVIDIA 4090

The RTX 4090 is fully capable of running custom kernels:

| Spec | Value |
|------|-------|
| Architecture | Ada Lovelace |
| CUDA Cores | 16,384 |
| Tensor Cores | 512 (4th gen) |
| Compute Capability | 8.9 |
| Memory | 24 GB GDDR6X |
| Memory Bandwidth | 1 TB/s |
| FP32 Performance | 82.6 TFLOPS |
| FP16 Tensor | 165 TFLOPS |

**Triton Compatibility:** Full support, compiles to sm_89 target

**Optimization Opportunities:**
- FP16/BF16 for 2x throughput on tensor cores
- Shared memory (128 KB per SM) for state caching
- L2 cache (72 MB) helps with sequential access patterns

---

## Comparison: All Options

| Approach | Inference | Training | Effort | Dependencies | Status |
|----------|-----------|----------|--------|--------------|--------|
| Nx/XLA Blelloch | 55ms | Works | Done | None | ✅ Done |
| ONNX INT8 | 0.5ms | No | Medium | ONNX Runtime | ✅ Documented |
| **Triton** | ~2ms? | Needs backward | Medium | Python, Triton | ✅ **Starter created** |
| **Rust NIF** | ~2ms? | Needs work | Medium | Rust, cudarc | ✅ **Starter created** |
| Custom XLA | ~2ms? | Works | High | CUDA toolkit | ⬜ Future |

**Both Paths Ready to Test:**
1. **Triton** - `python priv/triton/selective_scan.py` - Quick iteration
2. **Rust NIF** - `cd native/selective_scan_nif && cargo build --release` - Production path

---

## Next Steps

1. **Run Triton benchmark on 4090:**
   ```bash
   python priv/triton/selective_scan.py
   ```

2. **If speedup is good (>10x), proceed to Phase 2**

3. **If not, profile and optimize:**
   - Use Triton autotuner
   - Try different block sizes
   - Consider parallel scan within kernel

4. **For training:** Need to implement backward pass kernel

---

## References

- [Triton Documentation](https://triton-lang.org/)
- [Triton Scan Tutorial](https://triton-lang.org/main/getting-started/tutorials/)
- [Official Mamba CUDA Kernels](https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan)
- [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention) - Similar optimizations
- [cudarc (Rust CUDA)](https://github.com/coreylowman/cudarc)
