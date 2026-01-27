# Custom GPU Kernels for Mamba SSM

This document outlines our custom GPU kernel implementation for the Mamba selective scan.

## Summary

**Mission accomplished!** The Rust NIF achieves 60 FPS inference (10.96ms on RTX 4090).

After benchmarking all backbones, we found that **only Mamba needed optimization**:
- Attention: 0.014ms (PyTorch SDPA already uses Flash Attention)
- LSTM: 5.34ms (PyTorch already uses cuDNN)
- Mamba (Nx/XLA): 55ms ❌ → **Mamba (Rust NIF): 10.96ms** ✅

No additional kernel work is needed for 60 FPS inference.

---

## Current Status

| Implementation | Inference Time | 60 FPS? | Notes |
|----------------|----------------|---------|-------|
| **Rust NIF (CUDA)** | **10.96ms** | **Yes** | ✅ Production ready, no Python |
| ONNX INT8 | ~0.5ms | **Yes** | Requires export, inference only |
| XLA Custom Call | ~5ms (est.) | **Yes** | 🚧 Requires EXLA FFI support |
| Blelloch (Nx/XLA) | ~55ms | No | Current default in pure Elixir |
| Hillis-Steele (Nx/XLA) | ~56ms | No | Slightly slower |
| PyTorch Port | ~273ms | No | IPC overhead too high |
| Cumsum (Nx/XLA) | ~2500ms | No | XLA cumsum is pathologically slow |

**Target:** <16.67ms for 60 FPS real-time play ✅ **ACHIEVED with Rust NIF**

---

## Options Analysis

### Option 1: Triton Kernel (For Experimentation)

**What:** Python DSL that compiles to optimized PTX/CUBIN

**Purpose:** Rapid iteration and algorithm experimentation. **Not for production.**

**Pros:**
- 10x less code than raw CUDA
- Automatic memory coalescing and tiling
- `tl.associative_scan` built-in for parallel scans
- Easy to iterate and experiment
- Quick feedback loop for kernel optimization

**Cons:**
- Python dependency
- IPC overhead makes it slow for production (~273ms vs ~5ms kernel time)
- Not suitable for training or inference

**Use case:** Prototype kernel optimizations, then port to Rust NIF or Custom XLA op.

**Files:**
- `priv/triton/selective_scan.py` - Kernel implementation + benchmark
- `priv/python/pytorch_scan_server.py` - Port server (for testing only)
- `lib/exphil/bridge/pytorch_port.ex` - Elixir Port wrapper (for testing only)

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

### Option 4: Custom XLA Operation (Best for Performance)

**What:** C++/CUDA code registered as XLA CustomCall, called from EXLA

**Pros:**
- **No data transfer** - tensor stays on GPU the entire time
- Seamless Nx/Axon integration
- XLA can fuse with surrounding operations
- Works with existing training code

**Cons:**
- Complex build setup (XLA headers, CUDA toolkit)
- Need to match XLA's tensor layout
- Harder to debug

**Effort:** 1-2 weeks

**Why this is the best option:**
```
Nx/XLA current path:
  [GPU Tensor] → XLA parallel scan (slow) → [GPU Tensor]

Custom XLA op:
  [GPU Tensor] → Our CUDA kernel (fast) → [GPU Tensor]

No CPU involvement, no data transfer!
```

**Implementation approach:**
```cpp
// native/xla_selective_scan/selective_scan.cu
__global__ void selective_scan_kernel(...) {
  // Same kernel as Triton/Rust versions
}

// Register with XLA
XLA_REGISTER_CUSTOM_CALL_TARGET(SelectiveScan, "CUDA");
```

```elixir
# In Elixir
defn selective_scan_custom(x, dt, a, b, c) do
  custom_call(
    &EXLA.Defn.custom_call/4,
    ["SelectiveScan", {x, dt, a, b, c}],
    result_shape: Nx.shape(x)
  )
end
```

### Option 5: Pure CUDA with NIF

**What:** Raw CUDA kernel called via Erlang NIF

**Pros:**
- Maximum control and performance
- No Python dependency

**Cons:**
- Still has GPU↔CPU transfer overhead (tensor comes from EXLA)
- Complex build
- Manual memory management

**Effort:** 1 week+

**Note:** This is essentially what the Rust NIF does, just in C instead of Rust.

---

## Recommended Plan

```
Phase 1: Validate Algorithm ✅ COMPLETE
├── Write Triton kernel                     ✅ priv/triton/selective_scan.py
├── Benchmark on GPU                        ✅ PyTorch: 5ms (10x faster than XLA!)
├── Verify correctness                      ✅ Matches reference implementation
└── Test Elixir Port integration            ✅ Works but 273ms (IPC overhead)

Key Finding: The scan itself is fast (~5ms). XLA overhead is the problem.

Phase 2: Choose Integration Path
├── Option A: ONNX (works now)              ✅ 0.5ms, documented in INFERENCE.md
├── Option B: Rust NIF                      ⬜ Build and benchmark
└── Option C: Custom XLA Op                 ⬜ Best performance, most effort

Phase 3: Rust NIF Path (if chosen)
├── Build the NIF                           ⬜ cd native/selective_scan_nif && cargo build
├── Benchmark GPU↔CPU transfer              ⬜ Measure in-process overhead
├── Add backward pass                       ⬜ For training support
└── Integrate with MambaNIF module          ⬜ lib/exphil/networks/mamba_nif.ex

Phase 4: Custom XLA Op Path (best performance)
├── Set up XLA build environment            ⬜ XLA headers, CUDA toolkit
├── Write CUDA kernel                       ⬜ native/xla_selective_scan/
├── Register as XLA CustomCall              ⬜ XLA_REGISTER_CUSTOM_CALL_TARGET
├── Create EXLA bindings                    ⬜ EXLA.Defn.custom_call
└── Benchmark (should match PyTorch ~5ms)   ⬜ No data transfer!
```

## Why Data Transfer Matters

```
PyTorch benchmark (all on GPU):
  Create tensors → Scan (5ms) → Done
  Total: ~5ms ✅

Elixir Port (crosses process boundary):
  EXLA GPU → CPU copy (100ms) → msgpack (20ms) → Port IPC (5ms)
  → Python decode (10ms) → CPU → PyTorch GPU (50ms) → Scan (5ms)
  → PyTorch GPU → CPU (50ms) → msgpack → Port → Elixir → Nx
  Total: ~273ms ❌

Custom XLA Op (stays on GPU):
  EXLA GPU tensor → Our CUDA kernel (5ms) → EXLA GPU tensor
  Total: ~5ms ✅
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

| Approach | Inference | Training | Data Transfer | Effort | Status |
|----------|-----------|----------|---------------|--------|--------|
| Nx/XLA Blelloch | 55ms | Works | None (on GPU) | Done | ✅ Done |
| ONNX INT8 | 0.5ms | No | None (on GPU) | Medium | ✅ Documented |
| PyTorch Port | 273ms | No | **12MB IPC** | Done | ❌ Too slow |
| **Rust NIF** | ~5-10ms? | Needs work | GPU↔CPU in-process | Medium | ⬜ **Build & test** |
| **Custom XLA Op** | ~5ms? | Works | **None (on GPU)** | High | ⬜ **Best option** |

### Key Finding: Data Transfer is the Bottleneck

PyTorch achieves ~5ms for the scan itself (proven via benchmark). But:

- **PyTorch Port**: 273ms - IPC + msgpack + GPU transfers kill performance
- **Rust NIF**: Would be faster (no IPC) but still has GPU↔CPU transfers
- **Custom XLA Op**: Best - tensor stays on GPU, no transfers

### Recommended Paths

1. **For inference NOW**: Use ONNX export (~0.5ms) - already works
2. **For training speedup**: Build Rust NIF or Custom XLA op
3. **Long-term best**: Custom XLA op (tensor stays on GPU)

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
