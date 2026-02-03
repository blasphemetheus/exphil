# Elixir Forum Post Draft

**URL:** https://elixirforum.com/c/elixir-chat/machine-learning/
**Category:** Machine Learning (Nx/Axon)

---

## Title

Real-time inference with custom GPU attention - best approach?

## Body

I'm building a game AI that needs 60 FPS inference (~16ms budget). The model uses attention, and I'm exploring ways to get GPU-accelerated attention working with EXLA.

**What I've tried:**
- Pure Nx attention works but uses ~2-5ms on CPU
- Built a Rust+CUDA NIF for FlashAttention, but data copy overhead between EXLA tensors and the NIF adds latency

**Questions:**
1. Is there a way to access EXLA tensor memory directly from external CUDA code (zero-copy)?
2. Any plans to expose cuDNN flash attention like JAX's `implementation="cudnn"` option?
3. Would the Nx team be open to PRs adding GPU custom call support to EXLA? (I see CPU custom calls exist for QR/LU/Eigh)

I found [Issue #1461](https://github.com/elixir-nx/nx/issues/1461) discussing custom operation acceleration but it was closed as exploratory. Happy to contribute if there's a path forward.

---

## Notes

- Post here first to gauge interest before opening GitHub issue
- If positive response, follow up with formal feature request from `EXLA_FEATURE_REQUEST.md`
- Tag: nx, exla, cuda, machine-learning
