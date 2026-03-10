# TODO

## EXLA Upstream
- [x] PR to elixir-nx/nx: fix CallbackServer process leak — lazy-start only when `:runtime_call` nodes exist in graph (PR #1682)
- [ ] PR to elixir-nx/nx: expose `:allocator` option (`:bfc` / `:cuda_async` / `:default`), default `:bfc` for upstream
      Branch ready on fork: `feat/edifice-lazy-callback-allocator`
      After merge, update exphil/edifice configs to set `allocator: :cuda_async` explicitly
- [ ] PR to elixir-nx/nx: fused selective scan XLA custom call
      Branch: `feat/edifice-lazy-callback-allocator` (commit `aeddcda4`)
      Likely split from allocator PR into its own branch/PR

## XLA Selective Scan Integration
Status: Forward pass fully wired, backward pass implemented, all tests passing.

Branch `feat/edifice-lazy-callback-allocator` in blasphemetheus/nx fork contains:
- `exla/c_src/exla/custom_calls/fused_selective_scan.cu` — CUDA kernel + XLA FFI handler
- `exla/Makefile` — .cu compilation pattern rule
- `exla/lib/exla/mlir/value.ex` — `Value.fused_selective_scan/6` MLIR emitter
- `exla/lib/exla/defn.ex` — `:optional` handler for `:fused_selective_scan` on CUDA
- `exla/test/exla/defn/fused_selective_scan_test.exs` — 11 tests (forward, backward, CUDA, gradients)

Consumer in exphil: `lib/exphil/native/xla_selective_scan.ex`
- `selective_scan/5` — forward + custom_grad (differentiable, for training)
- `selective_scan_forward/5` — forward only (inference)
- Backward pass is pure-Nx (no CUDA backward kernel yet)

### Next Steps
- [x] Benchmark: compare XLA custom call vs NIF vs pure-Nx on RTX 5090
      Results (RTX 5090, large B=1 T=512 H=768 S=16):
      Custom call 0.97ms, CUDA fallback 3.85ms (3.95x slower), Host 5.90ms (6.1x slower)
- [ ] Wire exphil training to use `XLASelectiveScan.selective_scan/5` instead of NIF path
      Replace `ExPhil.Native.SelectiveScan.scan(x, dt, a, b, c)` calls with
      `ExPhil.Native.XLASelectiveScan.selective_scan(x, dt, a, b, c)` in the training loop.
      The new path is differentiable (custom_grad) so it works with value_and_grad directly.
- [x] CUDA backward kernel (in progress on nx fork branch)
- [x] bf16 support via precision.cuh (in progress on nx fork branch)
- [ ] Consider splitting into its own PR branch (separate from allocator changes)
- [ ] Upstream viability: discuss with elixir-nx maintainers whether custom call pattern
      is appropriate for nx repo or better as an external package

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
