# FlashAttention Backward Pass Implementation Guide

*Enabling GPU-accelerated attention for training, not just inference*

---

## Executive Summary

The FlashAttention NIF currently supports **forward-only** inference. This document outlines two approaches to enable **backward pass** support for training:

| Approach | Effort | Axon Compatible | GPU Efficiency | Recommended For |
|----------|--------|-----------------|----------------|-----------------|
| **Option A: NIF + Manual Grad** | 2-3 weeks | No (custom loop) | 2 GPU-CPU copies | Immediate speedup |
| **Option B: XLA Custom Call** | 4-6 weeks | Yes | Zero-copy | Long-term/upstream |

---

## The Core Problem

NIFs break the Nx computation graph because they require `Nx.to_binary()`:

```
EXLA tensor (GPU) → Nx.to_binary() → [GRAPH SEVERED] → NIF → Nx.from_binary() → EXLA tensor
                          ↑
           Autodiff cannot cross this boundary
```

`Nx.Defn.custom_grad` requires the forward expression to be a defn-compatible operation. NIFs are NOT defn-compatible because `Nx.to_binary()` is not a valid defn operation.

---

## Option A: NIF + Manual Gradients

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Manual Training Loop                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FORWARD PASS:                                                               │
│  ┌──────────┐    ┌───────────────────────┐    ┌────────────────────────┐    │
│  │ Embedding │──►│ FlashAttn NIF Forward │──►│ Policy Head + Loss     │    │
│  │  (Nx)     │   │ returns {O, logsumexp}│   │       (Nx)             │    │
│  └──────────┘    └───────────────────────┘   └────────────┬───────────┘    │
│       ▲                     ▲                              │                │
│       │                     │                              ▼                │
│  BACKWARD PASS:             │                         ┌─────────┐          │
│  ┌──────────┐    ┌──────────┴──────────┐              │ d_loss  │          │
│  │ d_embed  │◄───│ FlashAttn NIF Bwd   │◄─────────────┤         │          │
│  │ (Nx.grad)│    │ returns {dQ,dK,dV}  │              └─────────┘          │
│  └──────────┘    └─────────────────────┘                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Works

We manually "stitch" gradient flow by:
1. Computing partial gradients through Nx-based layers (embedding, policy head)
2. Calling the NIF backward pass for the attention block
3. Chaining the gradients together manually

### Reference Implementation: SelectiveScan NIF

The existing SelectiveScan NIF (`native/selective_scan_nif/src/kernel.rs`) demonstrates this pattern:

```rust
// Forward with state saving (lines 134-192)
extern "C" __global__ void selective_scan_forward_with_states_kernel(
    const float* x, const float* dt, const float* A, const float* B, const float* C,
    float* out,
    float* h_all,  // ← Saved hidden states for backward
    ...
)

// Backward kernel (lines 194-298)
extern "C" __global__ void selective_scan_backward_kernel(
    const float* dy,      // Gradient from output
    const float* x,       // Saved input
    const float* h_all,   // Saved hidden states
    const float* dt, const float* A, const float* B, const float* C,
    float* dx,            // Gradient outputs
    float* d_dt,
    float* dB,
    float* dC,
    ...
)
```

### FlashAttention-2 Backward Algorithm

Based on the [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):

**Forward pass must save:**
| Tensor | Shape | Purpose |
|--------|-------|---------|
| O (output) | [batch, seq, heads, head_dim] | Attention output |
| L (logsumexp) | [batch, heads, seq] | Row-wise log(sum(exp(scores))) |

**Backward pass inputs:**
| Input | Shape | Source |
|-------|-------|--------|
| dO | [batch, seq, heads, head_dim] | Gradient from downstream |
| Q, K, V | [batch, seq, heads, head_dim] | Original inputs (saved) |
| O | [batch, seq, heads, head_dim] | Forward output (saved) |
| L | [batch, heads, seq] | Logsumexp (saved) |

**Backward pass outputs:**
| Gradient | Shape |
|----------|-------|
| dQ | [batch, seq, heads, head_dim] |
| dK | [batch, seq, heads, head_dim] |
| dV | [batch, seq, heads, head_dim] |

**Computational cost:** 2.5x FLOPs of forward (recomputes attention scores in chunks).

### Files to Modify

#### 1. `native/flash_attention_nif/src/lib.rs`

Add new NIF functions:

```rust
/// Forward pass that saves logsumexp for backward
#[rustler::nif]
fn forward_with_logsumexp<'a>(
    env: Env<'a>,
    q_data: Binary<'a>,
    k_data: Binary<'a>,
    v_data: Binary<'a>,
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    causal: bool,
) -> NifResult<(Atom, Binary<'a>, Binary<'a>)> {
    // Returns (output, logsumexp)
    // logsumexp shape: [batch, num_heads, seq_len]
}

/// Backward pass computing gradients
#[rustler::nif]
fn backward_f32<'a>(
    env: Env<'a>,
    d_out: Binary<'a>,     // Gradient from output
    q_data: Binary<'a>,    // Saved Q
    k_data: Binary<'a>,    // Saved K
    v_data: Binary<'a>,    // Saved V
    output: Binary<'a>,    // Saved output
    logsumexp: Binary<'a>, // Saved logsumexp
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    causal: bool,
) -> NifResult<(Atom, Binary<'a>, Binary<'a>, Binary<'a>)> {
    // Returns (dQ, dK, dV)
}
```

#### 2. `native/flash_attention_nif/cuda/flash_bwd.cu` (New File)

```cuda
// FlashAttention-2 backward kernel
// Reference: https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_bwd_kernel.h

extern "C" __global__ void flash_attention_backward_kernel(
    const float* __restrict__ dO,     // [batch, seq, heads, head_dim]
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ L,      // [batch, heads, seq] logsumexp
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int batch, int seq_len, int num_heads, int head_dim,
    float softmax_scale, bool causal
) {
    // Tiled backward pass implementation
    // See Algorithm 4 in FlashAttention-2 paper
}
```

#### 3. `lib/exphil/native/flash_attention.ex`

Add Elixir wrappers:

```elixir
defmodule ExPhil.Native.FlashAttention do
  use Rustler, otp_app: :exphil, crate: "flash_attention_nif"

  # Existing forward-only (inference)
  def forward(q, k, v, opts \\ [])

  # NEW: Forward that saves state for backward
  @doc """
  Forward pass that returns logsumexp for backward pass.

  Returns `{:ok, output, logsumexp}` where:
  - output: [batch, seq, heads, head_dim] attention output
  - logsumexp: [batch, heads, seq] row-wise log(sum(exp(scores)))
  """
  def forward_with_states(q, k, v, opts \\ []) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    causal = Keyword.get(opts, :causal, true)

    q_bin = Nx.to_binary(q)
    k_bin = Nx.to_binary(k)
    v_bin = Nx.to_binary(v)

    case forward_with_logsumexp_nif(q_bin, k_bin, v_bin,
                                     batch, seq_len, num_heads, head_dim, causal) do
      {:ok, out_bin, lse_bin} ->
        output = Nx.from_binary(out_bin, :f32)
                 |> Nx.reshape({batch, seq_len, num_heads, head_dim})
        logsumexp = Nx.from_binary(lse_bin, :f32)
                    |> Nx.reshape({batch, num_heads, seq_len})
        {:ok, output, logsumexp}
      error -> error
    end
  end

  # NEW: Backward pass
  @doc """
  Backward pass computing gradients w.r.t. Q, K, V.

  ## Arguments
  - d_out: Gradient from downstream [batch, seq, heads, head_dim]
  - q, k, v: Original inputs (must be saved from forward)
  - output: Forward output (must be saved)
  - logsumexp: From forward_with_states

  Returns `{:ok, dq, dk, dv}` gradients.
  """
  def backward(d_out, q, k, v, output, logsumexp, opts \\ []) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    causal = Keyword.get(opts, :causal, true)

    case backward_f32_nif(
      Nx.to_binary(d_out),
      Nx.to_binary(q), Nx.to_binary(k), Nx.to_binary(v),
      Nx.to_binary(output), Nx.to_binary(logsumexp),
      batch, seq_len, num_heads, head_dim, causal
    ) do
      {:ok, dq_bin, dk_bin, dv_bin} ->
        shape = {batch, seq_len, num_heads, head_dim}
        dq = Nx.from_binary(dq_bin, :f32) |> Nx.reshape(shape)
        dk = Nx.from_binary(dk_bin, :f32) |> Nx.reshape(shape)
        dv = Nx.from_binary(dv_bin, :f32) |> Nx.reshape(shape)
        {:ok, dq, dk, dv}
      error -> error
    end
  end

  # NIF stubs
  defp forward_with_logsumexp_nif(_q, _k, _v, _b, _s, _h, _d, _c),
    do: :erlang.nif_error(:not_loaded)
  defp backward_f32_nif(_do, _q, _k, _v, _o, _l, _b, _s, _h, _d, _c),
    do: :erlang.nif_error(:not_loaded)
end
```

#### 4. `lib/exphil/training/flash_training.ex` (New File)

Custom training loop that uses NIF for attention:

```elixir
defmodule ExPhil.Training.FlashTraining do
  @moduledoc """
  Custom training loop using FlashAttention NIF for both forward and backward passes.

  This bypasses Axon.Loop because NIFs cannot participate in autodiff. Instead,
  we manually chain gradients through:
  1. Nx-based embedding layer (has gradients via Nx.grad)
  2. FlashAttention NIF (manual forward/backward)
  3. Nx-based policy head (has gradients via Nx.grad)
  """

  alias ExPhil.Native.FlashAttention

  @doc """
  Train one step with FlashAttention NIF.

  ## Architecture assumed:
  state → [Embedding] → Q,K,V → [FlashAttn] → [PolicyHead] → logits → loss
  """
  def train_step(params, batch, opts \\ []) do
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-4)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    # 1. Embedding layer (pure Nx - we'll need gradients later)
    {q, k, v} = compute_qkv(batch.states, params.embedding)

    # 2. FlashAttention forward (NIF - saves state for backward)
    {:ok, attn_out, logsumexp} = FlashAttention.forward_with_states(q, k, v)

    # 3. Policy head forward (pure Nx)
    logits = compute_policy_head(attn_out, params.policy)

    # 4. Compute loss
    loss = cross_entropy_loss(logits, batch.actions)

    # =========================================================================
    # BACKWARD PASS
    # =========================================================================

    # 5. Gradient through policy head (Nx.grad)
    #    d_loss/d_attn_out
    d_attn_out = policy_head_backward(logits, batch.actions, params.policy)

    # 6. Gradient through FlashAttention (NIF backward)
    #    d_loss/d_Q, d_loss/d_K, d_loss/d_V
    {:ok, dq, dk, dv} = FlashAttention.backward(
      d_attn_out, q, k, v, attn_out, logsumexp
    )

    # 7. Gradient through embedding layer (Nx.grad)
    #    d_loss/d_embedding_params
    d_embedding = embedding_backward(dq, dk, dv, batch.states, params.embedding)

    # =========================================================================
    # OPTIMIZER STEP
    # =========================================================================

    gradients = %{
      embedding: d_embedding,
      policy: policy_head_gradients(logits, batch.actions, attn_out, params.policy)
    }

    new_params = apply_gradients(params, gradients, learning_rate)

    {new_params, loss}
  end

  # ... helper functions for each layer's forward/backward
end
```

### Advantages

- Uses existing NIF infrastructure
- Full control over implementation
- Can be done incrementally (CPU first, then CUDA)
- No upstream changes required

### Disadvantages

- Cannot use `Axon.Loop` (requires custom training loop)
- Manual gradient wiring is error-prone
- Still has 2 GPU-CPU round trips per forward+backward
- Must maintain separate training code path

---

## Option B: XLA Custom Call

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        XLA Computation Graph                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌───────────────────────┐    ┌────────────────────────┐    │
│  │ Embedding │──►│ stablehlo.custom_call │──►│ Policy Head            │    │
│  │  (XLA)    │   │ "flash_attn_fwd"      │   │       (XLA)            │    │
│  └──────────┘    │ (GPU kernel, no copy) │   └────────────────────────┘    │
│                  └───────────────────────┘                                  │
│                                                                              │
│  Autodiff: Nx.Defn.custom_grad wraps the custom_call                        │
│  → Backward uses "flash_attn_bwd" custom_call                               │
│  → Tensors NEVER leave GPU                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Is Ideal

1. **Zero GPU-CPU copies**: Tensors stay on device throughout
2. **Works with Axon**: Can use `Axon.Loop`, checkpointing, etc.
3. **Composable**: Just another defn function
4. **Upstream benefit**: Could be contributed to EXLA for everyone

### Current EXLA State

EXLA supports CPU custom calls (QR, LU, Eigh) via XLA FFI:

```cpp
// deps/exla/c_src/exla/custom_calls/qr_f32.cc
XLA_FFI_DEFINE_HANDLER_SYMBOL(qr_cpu_custom_call_f32, qr_cpu_custom_call_f32_impl, ...);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_f32", "Host", ...);
//                                                                        ↑
//                                                               CPU only!
```

**GPU custom calls are NOT currently supported.** The registration uses `"Host"` device.

### Implementation Path

#### Step 1: Add CUDA Compilation to EXLA Makefile

```makefile
# deps/exla/Makefile (modified)
CUDA_SRCS := $(wildcard c_src/exla/custom_calls/*.cu)
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)

%.o: %.cu
    nvcc -c -o $@ $< $(CUDA_FLAGS)

$(PRIV_DIR)/libexla.so: $(OBJS) $(CUDA_OBJS)
    ...
```

#### Step 2: Create CUDA Kernel Files

```cpp
// deps/exla/c_src/exla/custom_calls/flash_attention_fwd.cu

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Forward kernel
static ffi::Error flash_attn_fwd_impl(
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> k,
    ffi::Buffer<ffi::F32> v,
    ffi::Result<ffi::Buffer<ffi::F32>> output,
    ffi::Result<ffi::Buffer<ffi::F32>> logsumexp
) {
    // Launch CUDA kernel
    // Tensors are already on GPU - no copies!
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(flash_attn_fwd, flash_attn_fwd_impl, ...);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "flash_attn_fwd", "CUDA", flash_attn_fwd);
//                                                               ↑
//                                                         GPU device!
```

#### Step 3: Add Elixir Bindings via EXLA.MLIR

```elixir
# lib/exla/mlir/value.ex (modified)

def flash_attention(%Value{function: func} = q, k, v, opts) do
  output_type = # same as q
  lse_type = # [batch, heads, seq]

  attributes = [
    call_target_name: attr_string("flash_attn_fwd"),
    api_version: attr_i32(4)
  ]

  [output, lse] = op(func, "stablehlo.custom_call", [q, k, v],
                      [output_type, lse_type], attributes: attributes)
  {output, lse}
end
```

#### Step 4: Create defn Wrapper with custom_grad

```elixir
defmodule ExPhil.Networks.XLAFlashAttention do
  import Nx.Defn

  @doc """
  FlashAttention via XLA custom call - works with Axon autodiff!
  """
  defn flash_attention(q, k, v, opts \\ []) do
    causal = opts[:causal] || true

    # Forward pass returns {output, logsumexp}
    {output, logsumexp} = EXLA.custom_call("flash_attn_fwd", [q, k, v],
                                            causal: causal)

    # Define custom gradient
    Nx.Defn.Kernel.custom_grad(output, [q, k, v], fn d_out ->
      {dq, dk, dv} = EXLA.custom_call("flash_attn_bwd",
                                       [d_out, q, k, v, output, logsumexp],
                                       causal: causal)
      [dq, dk, dv]
    end)
  end
end
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `deps/exla/Makefile` | Modify | Add CUDA compilation |
| `deps/exla/c_src/exla/custom_calls/flash_attention_fwd.cu` | Create | Forward CUDA kernel |
| `deps/exla/c_src/exla/custom_calls/flash_attention_bwd.cu` | Create | Backward CUDA kernel |
| `deps/exla/c_src/exla/custom_calls/flash_attention.h` | Create | Shared header |
| `deps/exla/lib/exla/mlir/value.ex` | Modify | Add flash_attention op |
| `lib/exphil/networks/xla_flash_attention.ex` | Create | defn wrapper |

### Advantages

- Zero GPU-CPU copies (tensors stay on device)
- Works with Axon.Loop, checkpointing, etc.
- Clean integration - just a defn function
- Could be contributed upstream to benefit the Nx ecosystem

### Disadvantages

- Requires forking/modifying EXLA
- More complex build system (CUDA in Elixir deps)
- C++/CUDA development required
- Upstream acceptance uncertain

---

## Recommended Approach

### Phase 1: Hybrid (Current State)

Use what we have now:
- **Training**: Pure Nx `memory_efficient_attention` (slower but has gradients)
- **Inference**: FlashAttention NIF (fast, no gradients needed)

This works today with no additional implementation.

### Phase 2: NIF + Manual Gradients (Option A)

1. Add backward kernel to FlashAttention NIF
2. Create CPU fallback first (easier to debug)
3. Port to CUDA kernel
4. Build custom training loop
5. Benchmark against pure Nx training

**Expected speedup**: ~2x for attention-heavy models

### Phase 3: XLA Custom Call (Option B)

1. Contribute GPU custom call support to EXLA (or fork)
2. Register FlashAttention as XLA custom op
3. Create defn wrapper with custom_grad
4. Enable full Axon integration

**Expected speedup**: ~3x (zero-copy advantage)

---

## Appendix: FlashAttention-2 Backward Algorithm

From Algorithm 4 in the [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):

```
Input: Q, K, V ∈ R^(N×d), O ∈ R^(N×d), dO ∈ R^(N×d), L ∈ R^N (logsumexp)
Output: dQ, dK, dV ∈ R^(N×d)

1. Initialize dQ, dK, dV to zeros
2. For each block of queries Qi (tiled for memory efficiency):
   a. Load Qi, Oi, dOi, Li
   b. Compute Di = rowsum(dOi ⊙ Oi)  // Used for gradient scaling
   c. For each block of keys/values Kj, Vj:
      i.   Compute Sij = Qi @ Kj^T / sqrt(d)  // Recompute attention scores
      ii.  Apply causal mask if needed
      iii. Compute Pij = exp(Sij - Li)  // Softmax (using saved logsumexp)
      iv.  dVj += Pij^T @ dOi
      v.   dPij = dOi @ Vj^T
      vi.  dSij = Pij ⊙ (dPij - Di)  // Gradient through softmax
      vii. dQi += dSij @ Kj / sqrt(d)
      viii.dKj += dSij^T @ Qi / sqrt(d)

3. Return dQ, dK, dV
```

Key insight: The backward pass must **recompute** attention scores (Sij) because they were not saved (O(n) vs O(n²) memory). This is why backward is 2.5x the FLOPs of forward.

---

## References

- [FlashAttention-2 Paper (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [Official FlashAttention Repository](https://github.com/Dao-AILab/flash-attention)
- [Stanford CRFM FlashAttention-2 Summary](https://crfm.stanford.edu/2023/07/17/flash2.html)
- [EXLA Custom Calls (QR, LU)](https://github.com/elixir-nx/nx/tree/main/exla/c_src/exla/custom_calls)
- [XLA FFI Documentation](https://github.com/openxla/xla/blob/main/xla/ffi/api/ffi.h)

---

*Last updated: 2026-01-28*
