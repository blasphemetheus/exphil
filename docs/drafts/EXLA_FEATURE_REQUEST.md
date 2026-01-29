# Feature Request Draft: GPU Custom Kernels for EXLA

**Target repo:** https://github.com/elixir-nx/nx
**Type:** Feature request / Discussion

---

## Title

Support for GPU Custom Calls and/or cuDNN Flash Attention

## Description

### Use Case

I'm building a real-time game AI that needs to run inference at 60 FPS (16.67ms budget per frame). The model uses attention mechanisms, and I'd like to leverage GPU-accelerated attention (FlashAttention/cuDNN FMHA) for lowest latency.

Currently, my options are:
1. **Pure Nx attention** - Works great, but at ~2-5ms on CPU may leave less headroom
2. **Separate NIF** - I've implemented a Rust+CUDA NIF, but it requires copying data between EXLA tensors and the NIF, adding overhead
3. **Python bridge** - Too slow for real-time (~10ms+ serialization overhead)

### Feature Request

I'm interested in two potential features (either would help):

#### Option A: Expose cuDNN Flash Attention

JAX has `jax.nn.dot_product_attention(q, k, v, implementation="cudnn")` which uses XLA's cuDNN FMHA integration. Could EXLA expose this?

```elixir
# Hypothetical API
defn attention(query, key, value, opts \\ []) do
  EXLA.Defn.dot_product_attention(query, key, value,
    implementation: :cudnn,  # or :xla (default)
    is_causal: opts[:causal]
  )
end
```

XLA supports this via `--xla_gpu_enable_cudnn_fmha=true`. I understand JAX's implementation has [known issues](https://github.com/jax-ml/jax/issues?q=dot_product_attention+cudnn), so this may not be ready for EXLA yet.

#### Option B: User-Provided Custom CUDA Kernels

A mechanism for users to register custom GPU kernels, similar to PyTorch's `torch.utils.cpp_extension`:

```elixir
# Hypothetical API - register a custom CUDA kernel
EXLA.CustomCall.register(:flash_attention_fwd,
  source: "path/to/flash_attention.cu",
  inputs: [:f16, :f16, :f16],  # Q, K, V types
  outputs: [:f16],             # Output type
  device: :cuda
)

# Use in defn
defn my_attention(q, k, v) do
  EXLA.CustomCall.invoke(:flash_attention_fwd, [q, k, v])
end
```

This would enable the community to experiment with custom GPU operations without forking EXLA.

### Current EXLA Custom Calls

I explored `exla/c_src/exla/custom_calls/` and see the pattern for CPU custom calls (QR, LU, Eigh using XLA FFI). These are great! However:

- GPU custom calls aren't currently supported (no `.cu` files, registration uses `"Host"` only)
- There's no mechanism for users to add their own kernels

### Related Discussion

[Issue #1461](https://github.com/elixir-nx/nx/issues/1461) explored "Special node acceleration via metadata" for operations like flash attention. José mentioned:

> "We would need to make EXLA itself extensible. Custom calls are one mechanism to achieve this."

This feature request is a follow-up to that discussion, with a concrete use case (real-time inference).

### Workaround

For now, I'm using a separate Rustler NIF that:
1. Converts EXLA tensors to binary (`Nx.backend_transfer` + `Nx.to_binary`)
2. Calls CUDA kernel via Rust FFI
3. Converts result back to Nx tensor

This works but adds ~1-2ms of data copy overhead that could be avoided with proper EXLA integration.

### Environment

- EXLA with CUDA backend
- RTX 3090 / A100 GPUs
- Use case: Real-time game AI inference at 60 FPS

### Questions

1. Is cuDNN FMHA support on the roadmap for EXLA?
2. Would PRs adding GPU custom call support be welcome?
3. Is there a recommended pattern for zero-copy tensor access from external CUDA code?

Thank you for the incredible work on Nx/EXLA!

---

## Notes for Posting

- Post to: https://github.com/elixir-nx/nx/issues/new
- Consider posting to Elixir Forum first for discussion
- Tag: `enhancement`, `exla`
