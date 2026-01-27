# Kernel Optimization Opportunities

This document explores custom GPU kernel opportunities across all ExPhil backbones, building on the success of the Mamba selective scan optimization (55ms → 11ms).

## Executive Summary

| Backbone | Current Time | Optimization | Expected Time | Priority |
|----------|--------------|--------------|---------------|----------|
| Mamba | 55ms | ✅ Rust NIF CUDA | **10.96ms** | Done |
| Attention | ~40ms | Flash Attention | ~5-10ms | **High** |
| LSTM/GRU | ~20ms | Fused CUDA kernel | ~8-12ms | Medium |
| Jamba | ~80ms | Both optimizations | ~15-20ms | High |
| MLP | ~5ms | XLA handles well | ~5ms | Low |

---

## 1. Attention: Flash Attention

### Current Bottleneck

Our `scaled_dot_product_attention` implementation:
```elixir
# O(N²) memory and compute
scores = Nx.dot(query, [2], [0], key, [2], [0])  # [batch, seq, seq]
scores = Nx.softmax(scores, axis: -1)            # Materializes N² matrix
output = Nx.dot(scores, [2], [0], value, [1], [0])
```

**Problems:**
1. **Memory**: Stores full N² attention matrix (~14MB for seq=60, hidden=512)
2. **Memory bandwidth**: Multiple passes over the attention matrix
3. **No fusion**: Three separate kernel launches

### Flash Attention Solution

Flash Attention (Dao et al., 2022) computes exact attention without materializing the N² matrix:

```
┌─────────────────────────────────────────────────────────────┐
│ Standard Attention          │ Flash Attention              │
├─────────────────────────────┼──────────────────────────────┤
│ Q, K, V → [B, N, d]         │ Q, K, V → [B, N, d]          │
│ S = QK^T → [B, N, N] (HBM)  │ Split Q,K,V into blocks      │
│ P = softmax(S) → [B, N, N]  │ For each block:              │
│ O = PV → [B, N, d]          │   Compute local attention    │
│                              │   Update running softmax     │
│ Memory: O(N²)               │   Accumulate output          │
│                              │ Memory: O(N) ✓               │
└─────────────────────────────┴──────────────────────────────┘
```

**Key insight**: Softmax can be computed incrementally (online softmax trick).

### Implementation Options

#### Option A: FlashAttention-2 via Triton (Experimentation)

```python
# priv/triton/flash_attention.py
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    ...
):
    # Tiled attention computation
    # See: https://github.com/Dao-AILab/flash-attention
```

#### Option B: Rust NIF with cudarc (Production)

Same pattern as selective scan:
```rust
// native/flash_attention_nif/src/kernel.rs
const FLASH_ATTENTION_KERNEL: &str = r#"
extern "C" __global__ void flash_attention_fwd(
    const float* Q, const float* K, const float* V,
    float* O, float* L, float* M,  // L=rowsum, M=rowmax for online softmax
    int batch, int heads, int seq_len, int head_dim,
    int BLOCK_M, int BLOCK_N
) {
    // Block-wise attention with online softmax
    ...
}
"#;
```

#### Option C: Use existing FlashAttention library

cuDNN 8.9+ has Flash Attention built-in. We could:
1. Link against cuDNN from Rust NIF
2. Or use PyTorch's `F.scaled_dot_product_attention` via Port (for experimentation)

### Expected Improvement

| Metric | Standard | Flash Attention |
|--------|----------|-----------------|
| Memory | O(N²) = 14MB | O(N) = 240KB |
| Kernel launches | 3 | 1 |
| Time (est.) | ~40ms | ~5-10ms |

### Priority: **HIGH**

Flash Attention is the single biggest optimization opportunity after Mamba.

---

## 2. LSTM/GRU: Fused Recurrent Kernels

### Current Implementation

Axon's LSTM uses standard Nx operations:
```elixir
# Each timestep is a separate computation
for t <- 0..seq_len do
  {i, f, g, o} = compute_gates(x_t, h_prev)
  c = f * c_prev + i * g
  h = o * tanh(c)
end
```

**Problems:**
1. **Sequential**: Can't parallelize across time (inherent to RNNs)
2. **Kernel launch overhead**: Multiple small kernels per timestep
3. **No gate fusion**: i, f, g, o computed separately

### Optimization: Fused LSTM Kernel

cuDNN provides highly optimized LSTM/GRU implementations. We can:

#### Option A: cuDNN via Rust NIF

```rust
// native/fused_lstm_nif/src/lib.rs
use cudarc::cudnn::{Cudnn, RnnDescriptor, RnnMode};

pub fn lstm_forward(
    x: &[f32],       // [batch, seq, input]
    h0: &[f32],      // [layers, batch, hidden]
    c0: &[f32],      // [layers, batch, hidden]
    weights: &[f32], // Packed weights
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let cudnn = Cudnn::new()?;
    let rnn = RnnDescriptor::new(
        cudnn,
        RnnMode::LSTM,
        hidden_size,
        num_layers,
        ...
    )?;
    rnn.forward(x, h0, c0, weights)
}
```

#### Option B: Custom fused kernel

If cuDNN isn't flexible enough, write a custom kernel:

```cuda
__global__ void fused_lstm_kernel(
    const float* x,      // [batch, seq, input]
    const float* Wi, const float* Wh, const float* b,
    float* h, float* c,  // Hidden states
    float* out,          // Output sequence
    int batch, int seq_len, int input_size, int hidden_size
) {
    // Fused gate computation
    // All 4 gates (i,f,g,o) in one kernel
    // Pointwise ops fused with matmuls where possible
}
```

### Expected Improvement

| Metric | Current | Fused cuDNN |
|--------|---------|-------------|
| Kernel launches/step | ~10 | 1 |
| Time (est.) | ~20ms | ~8-12ms |

### Priority: **MEDIUM**

LSTM/GRU are already reasonably fast. cuDNN integration is straightforward but lower impact than Flash Attention.

---

## 3. Jamba: Combined Optimizations

### Current Architecture

```
Mamba Block 1  (~8ms with NIF)
Mamba Block 2  (~8ms)
Mamba Block 3  (~8ms)
Attention Block (~40ms)  ← Bottleneck!
Mamba Block 4  (~8ms)
...
Total: ~80ms for 6 layers
```

### Optimized Architecture

With both Mamba NIF and Flash Attention:
```
Mamba Block 1  (~3ms fused)
Mamba Block 2  (~3ms)
Mamba Block 3  (~3ms)
Flash Attention (~8ms)  ← Much faster
Mamba Block 4  (~3ms)
...
Total: ~15-20ms for 6 layers
```

### Implementation Plan

1. **Use Mamba NIF**: Already done, 10.96ms
2. **Add Flash Attention NIF**: New kernel
3. **Fuse normalization**: LayerNorm into Mamba/Attention kernels

### Priority: **HIGH** (after Flash Attention)

---

## 4. Cross-Cutting Optimizations

### 4.1 LayerNorm Fusion

LayerNorm is called after every block. Currently:
```
y = mamba(x)
y = layer_norm(y)  # Separate kernel
```

Fused:
```
y = mamba_with_layernorm(x)  # One kernel, data stays in registers
```

**Implementation**: Add LayerNorm to Mamba CUDA kernel's output stage.

### 4.2 Activation Fusion

GELU, SiLU, etc. are separate kernel launches:
```
x = linear(x)
x = silu(x)  # Separate kernel
```

These can be fused into the preceding linear layer.

### 4.3 Residual Connection Fusion

```
y = block(x)
y = y + x  # Separate kernel for element-wise add
```

Can be fused into the block's output.

### 4.4 Memory Format: NHWC vs NCHW

For 1D sequences, this is N(batch), L(length), C(channels):
- **NLC (current)**: Natural for Nx, good for sequential access
- **NCL**: Better for some CUDA kernels, worse memory coalescing

Generally NLC is fine for our use case.

---

## 5. Implementation Roadmap

### Phase 1: Flash Attention (1-2 weeks)
```
1. Write Triton prototype for experimentation
2. Benchmark against standard attention
3. Port to Rust NIF with cudarc
4. Integrate with Attention module
5. Benchmark full model
```

### Phase 2: Fused Operations (1 week)
```
1. Add LayerNorm to Mamba CUDA kernel
2. Fuse residual connections
3. Benchmark improvements
```

### Phase 3: cuDNN LSTM (optional, 1 week)
```
1. Add cuDNN bindings to cudarc
2. Implement fused LSTM forward
3. Benchmark vs current implementation
```

### Phase 4: XLA Custom Calls (future)
```
1. Wait for EXLA FFI support
2. Register kernels as XLA custom calls
3. Zero-copy tensor handling
```

---

## 6. Benchmark Targets

| Backbone | Current | Target | Status |
|----------|---------|--------|--------|
| MLP | 5ms | 5ms | ✅ Good |
| Mamba | 55ms | <15ms | ✅ **10.96ms** |
| Attention | 40ms | <15ms | 🎯 Next |
| LSTM | 20ms | <15ms | ⬜ Later |
| GRU | 18ms | <12ms | ⬜ Later |
| Jamba | 80ms | <20ms | ⬜ After attention |

**Goal**: All backbones under 16.67ms (60 FPS)

---

## 7. References

### Flash Attention
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [GitHub: Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

### CUDA Optimization
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)

### Triton
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton Flash Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

### cudarc
- [cudarc Rust crate](https://github.com/coreylowman/cudarc)
- [cudarc cuDNN support](https://docs.rs/cudarc/latest/cudarc/cudnn/)
