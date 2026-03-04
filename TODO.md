# TODO

## EXLA Fork
- [ ] PR to elixir-nx/nx: expose `:allocator` option (`:bfc` / `:cuda_async` / `:default`), default `:default` for upstream
- [ ] PR to elixir-nx/nx: fix CallbackServer process leak — lazy-start only when `:runtime_call` nodes exist in graph

## Training Infrastructure
- [ ] Save full training config JSON alongside checkpoints so eval/resume don't need architecture flags
- [ ] Expose lazy batching mode as CLI flag (`--lazy-sequences`) for large datasets

## Fused CUDA Kernels
- [x] Run fused kernel A/B benchmark (`scripts/benchmark_fused_ab.sh`)
- [ ] Fix fused selective_scan kernel crash — Mamba segfaults with `EDIFICE_DISABLE_FUSED=0` (batch_size 64+128)
- [ ] Fix fused delta_rule_scan perf — DeltaNet 65% slower with fused kernel (40 vs 115 batch/s)
      Likely: wrong block size, uncoalesced memory, or missing shared mem optimization
- [~] Per-kernel fused dispatch via runtime auto-tune (`Edifice.CUDA.AutoTune`)
      **Done:** `auto_tune.ex` written (use_fused?/2, warmup/1, report/0, disk cache, env var overrides)
      **Remaining — execute plan next session:**
      1. Modify `custom_call_available?/0` in `fused_scan.ex` — add `Process.get(:__edifice_force_fallback__)` check
      2. Replace `custom_call_available?()` with `AutoTune.use_fused?(:kernel_name, tensor)` in 29 dispatch functions
         - Lines 55,69,83,97,111,125,148,169,190,214,234,255,277,300,323,348,381 (scan kernels)
         - Lines 3344,3539,3736 (attention variants — replace `*_custom_call_available?()`)
         - Lines 3943,4037,4144,4261 (reservoir, titans, miras, gsa)
         - Lines 4392,4515,4650,4772,4914 (block scans — replace `block_custom_call_available?()` etc.)
      3. Write `test/edifice/cuda/auto_tune_test.exs` — cache, disk persistence, env var overrides
      4. Compile + run targeted tests in edifice
      5. Smoke test with ExPhil training (verify per-kernel log output)
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
