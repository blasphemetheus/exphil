# Kernel Optimization Opportunities

This document explores custom GPU kernel opportunities across all ExPhil backbones.

## Executive Summary

**Key Finding: Only Mamba needed optimization.** All other backbones are already fast on RTX 4090.

| Backbone | Measured Time | 60 FPS? | Optimization Needed? |
|----------|---------------|---------|---------------------|
| **Mamba (Nx/XLA)** | 55ms | ❌ | **Yes → Fixed!** |
| **Mamba (Rust NIF)** | **10.96ms** | ✅ | Done |
| Attention (standard) | 0.067ms | ✅ | No |
| Attention (SDPA/Flash) | 0.014ms | ✅ | No |
| LSTM | 5.34ms | ✅ | No |
| MLP | ~5ms | ✅ | No |

**Conclusion:** The Rust NIF for Mamba selective scan was the only necessary optimization. PyTorch's built-in SDPA already uses Flash Attention. LSTM uses cuDNN internally. No further kernel work needed for 60 FPS inference.

---

## 1. Attention: Already Fast ✅

### Benchmark Results (RTX 4090)

```
Standard attention: 0.067 ms
Flash/SDPA:         0.014 ms (4.62x speedup)
```

**Both are well under the 16.67ms target.** No custom kernel needed.

### Why It's Fast

1. **Short sequences**: seq_len=60 means only 3,600 attention scores (60²)
2. **PyTorch SDPA**: `torch.nn.functional.scaled_dot_product_attention` already uses Flash Attention
3. **RTX 4090**: Handles small attention matrices in microseconds

### Recommendation

Use PyTorch's SDPA for any attention-based models. It automatically selects:
- Flash Attention (when available)
- Memory-efficient attention (fallback)
- Math attention (CPU fallback)

```python
# Already optimal - no custom kernel needed
output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

### If Sequences Get Longer

For seq_len > 1000, consider:
1. Sliding window attention (already in our Attention module)
2. Flash Attention via Rust NIF (pattern same as Mamba NIF)

But for Melee's 60-frame windows, this is unnecessary.

---

## 2. LSTM/GRU: Already Fast ✅

### Benchmark Results (RTX 4090)

```
LSTM (PyTorch cuDNN): 5.34 ms
```

**Well under the 16.67ms target.** No custom kernel needed.

### Why It's Fast

PyTorch's LSTM already uses cuDNN internally, which provides:
1. Fused gate computation (i, f, g, o in one kernel)
2. Optimized memory access patterns
3. Tensor Core acceleration on modern GPUs

### Recommendation

Use PyTorch/EXLA LSTM as-is. The cuDNN backend is already optimal.

```python
# Already uses cuDNN - no custom kernel needed
lstm = torch.nn.LSTM(hidden, 256, batch_first=True)
output, (h, c) = lstm(x)
```

### Note on Axon LSTM

Axon's LSTM implementation in Nx/XLA may be slower than PyTorch's cuDNN version.
If LSTM becomes a bottleneck in Elixir training, consider:
1. Using the PyTorch Port for LSTM forward pass
2. Implementing a cuDNN NIF (similar pattern to Mamba NIF)

But for inference, 5.34ms is plenty fast.

---

## 3. Jamba: Already Optimized ✅

### Updated Architecture Performance

With Mamba NIF and PyTorch SDPA:
```
Mamba Block 1  (~2ms per block with NIF)
Mamba Block 2  (~2ms)
Mamba Block 3  (~2ms)
Attention Block (~0.07ms)  ← Already fast!
Mamba Block 4  (~2ms)
...
Total estimate: ~15ms for 6 layers ✅
```

### Recommendation

Jamba should work well with:
1. **Mamba blocks**: Use Rust NIF (10.96ms total for scan)
2. **Attention blocks**: Use PyTorch SDPA (0.014ms)

No additional kernel work needed.

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

## 5. Implementation Status

### Completed ✅

```
✅ Mamba Rust NIF (10.96ms) - The only bottleneck, now fixed
✅ Benchmarked Attention (0.067ms standard, 0.014ms SDPA) - Already fast
✅ Benchmarked LSTM (5.34ms) - Already fast
```

### Not Needed

```
❌ Flash Attention NIF - PyTorch SDPA already optimal (0.014ms)
❌ cuDNN LSTM NIF - PyTorch already uses cuDNN (5.34ms)
❌ Fused LayerNorm - Not a bottleneck
```

### Future (if needed)

```
⬜ XLA Custom Calls - When EXLA adds FFI support
   Would reduce Mamba NIF from 10.96ms to ~5ms by eliminating GPU↔CPU transfer
```

---

## 6. Final Benchmark Results (RTX 4090)

| Backbone | Measured | 60 FPS Target | Status |
|----------|----------|---------------|--------|
| MLP | ~5ms | < 16.67ms | ✅ Pass |
| **Mamba (NIF)** | **10.96ms** | < 16.67ms | ✅ **Pass** |
| Mamba (Nx/XLA) | 55ms | < 16.67ms | ❌ Use NIF instead |
| Attention (SDPA) | 0.014ms | < 16.67ms | ✅ Pass |
| LSTM | 5.34ms | < 16.67ms | ✅ Pass |

**All backbones now achieve 60 FPS!**

The Mamba Rust NIF was the only custom kernel work needed.

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
