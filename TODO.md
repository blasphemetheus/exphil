# TODO

## EXLA Upstream
- [x] PR to elixir-nx/nx: fix CallbackServer process leak — lazy-start only when `:runtime_call` nodes exist in graph (PR #1682)
- [ ] PR to elixir-nx/nx: expose `:allocator` option (`:bfc` / `:cuda_async` / `:default`), default `:bfc` for upstream
      Branch ready on fork: `feat/edifice-lazy-callback-allocator`
      After merge, update exphil/edifice configs to set `allocator: :cuda_async` explicitly

## Training Infrastructure
- [ ] Save full training config JSON alongside checkpoints so eval/resume don't need architecture flags
- [ ] Expose lazy batching mode as CLI flag (`--lazy-sequences`) for large datasets

## Fused CUDA Kernels
- [x] Run fused kernel A/B benchmark (`scripts/benchmark_fused_ab.sh`)
- [x] Fix fused selective_scan kernel crash — replaced 128KB/thread stack array with cudaMallocAsync workspace + fixed grad_B/grad_C bf16 type mismatch
- [ ] Fix fused delta_rule_scan perf — DeltaNet 65% slower with fused kernel (40 vs 115 batch/s)
      Likely: wrong block size, uncoalesced memory, or missing shared mem optimization
- [x] Per-kernel fused dispatch via runtime auto-tune (`Edifice.CUDA.AutoTune`)
      `auto_tune.ex` + 29 dispatch functions wired + 20 tests passing + smoke tested
      Note: benchmark can't run inside JIT — call `AutoTune.warmup()` before training, or defaults to fused
- [ ] Test remaining kernel variants (attention, flash_attention, liquid, lstm, etc.)

## Architecture Evaluation
- [ ] Jamba 20-epoch convergence test (with kCudaAsync allocator + val_split)
- [ ] Compare top architectures at 20 epochs (H3, Zamba, MinGRU, Mamba)

## Mode Collapse / Class Imbalance (Jamba eval 2026-03-04)

Jamba 20ep Mewtwo: 85.3% overall but all buttons predicted 0%, sticks collapse to neutral.
High accuracy is artifact of class imbalance (buttons pressed ~5% of frames).
Not Jamba-specific — any architecture will collapse the same way on this data distribution.

**Fixes (ordered by expected impact):**
- [ ] Focal loss (gamma=2.0, sweep 1.0-3.0) — downweight easy "no press" examples
- [ ] Per-head class weighting — inverse frequency (L ~8x, X ~15x)
- [ ] Action-conditional oversampling — 2-4x oversample frames with button presses
- [ ] Lower button sigmoid threshold at inference (quick win, doesn't fix training)
- [ ] Eval metric: pred/actual press-rate ratio (accuracy alone hides collapse)
