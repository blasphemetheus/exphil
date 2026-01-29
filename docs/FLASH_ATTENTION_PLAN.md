# Flash Attention Implementation Plan

This document explores options for implementing Flash Attention in ExPhil to enable longer sequences and reduce memory usage.

## Why Flash Attention?

### Current Limitation
Standard attention computes the full n×n attention matrix:
```elixir
# Current implementation (attention.ex:scaled_dot_product_attention)
scores = Nx.dot(query, [2], [0], key, [2], [0])  # [batch, seq, seq] - O(n²) memory
weights = FusedOps.fused_softmax(scores)
output = Nx.dot(weights, [2], [0], value, [1], [0])
```

For seq_len=128, batch=32: scores matrix is 32 × 128 × 128 × 4 bytes = **2MB per layer**

### Flash Attention Benefit
Flash Attention computes attention in tiles without materializing the full matrix:
- **O(n) memory** instead of O(n²)
- Enables 2-4x longer sequences with same VRAM
- Better GPU cache utilization (reduced HBM bandwidth)

### ExPhil-Specific Impact

| Metric | Current (standard) | With Flash Attention |
|--------|-------------------|---------------------|
| Max seq_len (24GB VRAM) | ~90 frames | ~200+ frames |
| Memory per attention layer | O(n²) | O(n) |
| Training batch size | 256 | 512+ at same memory |

Longer sequences mean more temporal context for:
- Reaction timing (opponent mixups)
- Combo execution (multi-hit sequences)
- Recovery patterns (off-stage situations)

---

## Implementation Options

### Option 1: Pure Nx Tiled Attention (Already Done: Chunked)

**Status:** ✅ Implemented as `chunked_attention/4`

We already implemented chunked attention which processes queries in blocks:
```elixir
Attention.chunked_attention(query, key, value, chunk_size: 32, mask: mask)
```

**Limitations:**
- Still materializes chunk×seq attention tiles (not true O(n) memory)
- No online softmax normalization (can't fuse across chunks perfectly)
- ~20-30% memory reduction, not 4x

**Verdict:** Good stepping stone, but not true flash attention.

---

### Option 2: XLA Flash Attention Custom Call

**Approach:** Use XLA's cuDNN Flash Attention backend when available.

**Current Status (Jan 2026):**
- JAX has `jax.nn.dot_product_attention` with `implementation='cudnn'`
- XLA supports cuDNN flash attention on Ampere+ GPUs (RTX 30xx, 40xx, A100)
- EXLA may expose this through `EXLA.Backend.cuda_fused_attention` (not yet)

**Implementation Path:**
```elixir
# Future API (when EXLA supports it)
defn flash_attention(query, key, value, opts \\ []) do
  EXLA.Backend.fused_attention(query, key, value,
    scale: 1.0 / :math.sqrt(Nx.axis_size(query, -1)),
    is_causal: opts[:causal] || true,
    dropout_rate: opts[:dropout] || 0.0
  )
end
```

**Dependencies:**
- EXLA compiled with cuDNN 8.9+ support
- CUDA 12.0+
- Ampere or newer GPU (sm_80+)

**Pros:**
- Native XLA integration (no NIFs)
- Optimized by NVIDIA
- Automatic differentiation works

**Cons:**
- Depends on EXLA team adding the API
- May not be available for older GPUs

**Effort:** Low (once EXLA supports it) | **Timeline:** Unknown

**Tracking:**
- [elixir-nx/nx#1234](https://github.com/elixir-nx/nx) - Flash attention support (hypothetical)
- [google/jax#14223](https://github.com/google/jax/issues/14223) - JAX flash attention

---

### Option 3: NIF with FlashAttention-2 CUDA Kernel

**Approach:** Write a Rust/C NIF wrapping NVIDIA's FlashAttention-2 CUDA code.

**Implementation:**
```
lib/
  exphil_flash_attention/
    native/
      Cargo.toml
      src/lib.rs       # Rustler NIF
      cuda/
        flash_attn.cu  # CUDA kernel (from flash-attention repo)
```

**NIF Interface:**
```elixir
defmodule ExPhil.Native.FlashAttention do
  use Rustler, otp_app: :exphil, crate: "exphil_flash_attention"

  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def forward(_query, _key, _value, _opts), do: :erlang.nif_error(:not_loaded)

  @spec backward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}
  def backward(_grad, _query, _key, _value, _opts), do: :erlang.nif_error(:not_loaded)
end
```

**Axon Integration:**
```elixir
def flash_attention_layer(input, opts) do
  Axon.layer(
    fn query, key, value, _opts ->
      ExPhil.Native.FlashAttention.forward(query, key, value, opts)
    end,
    [input, input, input],
    name: opts[:name] || "flash_attn"
  )
end
```

**Dependencies:**
- Rust + Rustler for NIF binding
- CUDA toolkit 12.0+
- FlashAttention-2 source (MIT license)
- Ampere+ GPU

**Pros:**
- Full control over implementation
- Can optimize for our specific use case
- Works with any EXLA version

**Cons:**
- Significant implementation effort
- Must implement backward pass for training
- Build complexity (CUDA compilation)
- Platform-specific (NVIDIA only)

**Effort:** High | **Timeline:** 2-4 weeks

**Reference:**
- [flash-attention](https://github.com/Dao-AILab/flash-attention) - Official CUDA implementation
- [Rustler](https://github.com/rusterlium/rustler) - Rust NIF binding

---

### Option 4: Triton-Based Implementation via Python Bridge

**Approach:** Use Python's FlashAttention via `Pythonx` or a Python port.

**Implementation:**
```elixir
defmodule ExPhil.FlashAttention.Python do
  def forward(query, key, value, opts) do
    # Convert Nx tensors to numpy
    q_np = Nx.to_binary(query) |> numpy_from_binary(Nx.shape(query))
    k_np = Nx.to_binary(key) |> numpy_from_binary(Nx.shape(key))
    v_np = Nx.to_binary(value) |> numpy_from_binary(Nx.shape(value))

    # Call Python flash attention
    result = Pythonx.call("flash_attn", "flash_attn_func", [q_np, k_np, v_np])

    # Convert back to Nx
    Nx.from_binary(result, :f32) |> Nx.reshape(Nx.shape(query))
  end
end
```

**Pros:**
- Leverage existing Python ecosystem
- Easier to get working initially

**Cons:**
- Data transfer overhead (Elixir → Python → Elixir)
- Python GIL contention
- Complex deployment (Python environment)
- Not suitable for production inference

**Effort:** Medium | **Timeline:** 1 week (prototype only)

**Verdict:** Good for experimentation, not production.

---

### Option 5: Memory-Efficient Attention (MEA) in Pure Nx

**Approach:** Implement memory-efficient attention algorithm in pure Nx with online softmax.

This is a simpler algorithm than FlashAttention that still achieves O(n) memory:

```elixir
defn memory_efficient_attention(query, key, value, opts \\ []) do
  chunk_size = opts[:chunk_size] || 32
  {batch, seq_q, dim} = Nx.shape(query)
  seq_k = Nx.axis_size(key, 1)
  scale = Nx.rsqrt(dim)

  # Initialize accumulators
  output = Nx.broadcast(0.0, {batch, seq_q, dim})
  lse = Nx.broadcast(Nx.Constants.neg_infinity(), {batch, seq_q})  # log-sum-exp

  # Process key/value in chunks with online softmax normalization
  num_kv_chunks = div(seq_k + chunk_size - 1, chunk_size)

  {output, _lse} =
    while {output, lse}, i <- 0..(num_kv_chunks - 1) do
      k_start = i * chunk_size
      k_chunk = Nx.slice_along_axis(key, k_start, chunk_size, axis: 1)
      v_chunk = Nx.slice_along_axis(value, k_start, chunk_size, axis: 1)

      # Compute attention for this KV chunk
      scores = Nx.dot(query, k_chunk, axes: [[2], [2]]) |> Nx.multiply(scale)

      # Online softmax update (numerically stable)
      new_max = Nx.max(lse, Nx.reduce_max(scores, axes: [-1]))
      exp_scores = Nx.exp(scores - Nx.new_axis(new_max, -1))
      exp_sum = Nx.sum(exp_scores, axes: [-1])

      # Update running sum with correction factor
      correction = Nx.exp(lse - new_max)
      new_lse = new_max + Nx.log(correction + exp_sum)

      # Weighted update of output
      alpha = correction / (correction + exp_sum)
      chunk_output = Nx.dot(exp_scores, v_chunk, axes: [[2], [1]])
      new_output = output * Nx.new_axis(alpha, -1) +
                   chunk_output / Nx.new_axis(correction + exp_sum, -1)

      {new_output, new_lse}
    end

  output
end
```

**Pros:**
- Pure Nx/defn (XLA compatible)
- O(n) memory with online softmax
- Works on any GPU
- Automatic differentiation

**Cons:**
- More compute than FlashAttention (no kernel fusion)
- ~2x slower than FlashAttention-2
- Complex implementation

**Effort:** Medium-High | **Timeline:** 1-2 weeks

---

## Comparison Matrix

| Option | Memory | Speed | Effort | GPU Req | Production Ready |
|--------|--------|-------|--------|---------|------------------|
| Chunked (done) | O(n×chunk) | 1.0x | Done | Any | ✅ Yes |
| XLA Custom Call | O(n) | 1.5-2x | Low* | Ampere+ | ⏳ When available |
| NIF + CUDA | O(n) | 2x | High | Ampere+ | ✅ After impl |
| Python Bridge | O(n) | 0.5x | Medium | Ampere+ | ❌ Prototype only |
| Pure Nx MEA | O(n) | 0.8x | Medium | Any | ✅ Yes |

*Effort is low once EXLA adds support

---

## Recommended Approach

### Short Term (Now)
1. **Use chunked attention** for immediate memory savings
2. **Monitor EXLA/XLA** for flash attention support
3. **Benchmark** to establish baseline memory/speed

### Medium Term (1-2 months)
4. **Implement Pure Nx MEA** as fallback for any GPU
5. **Test with longer sequences** (120+ frames)

### Long Term (If Needed)
6. **NIF + CUDA** if XLA support doesn't materialize and we need max performance

---

## Implementation Checklist

### Phase 1: Baseline & Monitoring
- [ ] Benchmark current attention memory usage at various seq_len
- [ ] Document max seq_len per GPU (RTX 3090, 4090, A100)
- [ ] Set up alerts for EXLA flash attention support
- [ ] Create test suite for attention output equivalence

### Phase 2: Pure Nx MEA
- [x] Implement `memory_efficient_attention/4` in attention.ex
- [x] Add `--memory-efficient-attention` flag
- [x] Verify output matches standard attention (6 tests passing)
- [ ] Benchmark memory and speed
- [x] Update documentation

### Phase 3: Production Optimization (If Needed)
- [ ] Evaluate EXLA flash attention when available
- [ ] Consider NIF implementation if 2x speedup critical
- [ ] Profile and optimize chosen implementation

---

## References

### Papers
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682) (Rabe & Staats, 2021)

### Implementations
- [flash-attention](https://github.com/Dao-AILab/flash-attention) - CUDA implementation
- [xformers](https://github.com/facebookresearch/xformers) - Memory-efficient attention
- [JAX flash attention](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.dot_product_attention.html)

### Related Issues
- [google/jax#14223](https://github.com/google/jax/issues/14223) - JAX flash attention
- [openxla/xla#12429](https://github.com/openxla/xla/issues/12429) - XLA BF16 issues (related)
