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
- [ ] Fix `--button-pos-weight` CLI override ignored when `--preset` is used
      `get_cli_overrides()` in config.ex doesn't include button_pos_weight, so
      `defaults() |> merge(preset) |> merge(cli_overrides)` overwrites the parsed
      value with the default `:auto`. Add `parse_button_pos_weight` to cli_overrides.
- [x] Fix `compute_pos_weights` hang — was accessing `train_dataset.actions.buttons` which
      doesn't exist as a precomputed tensor. Rewrote to use `Data.stats().button_rates`
      (already computed) with new `compute_pos_weights_from_rates/2`. Instant now.
- [x] Fix EXLA/Expr backend mismatch in `focal_binary_cross_entropy` — `button_pos_weight`
      tensor captured in defn closure must be on BinaryBackend (GOTCHAS.md #3)
- [x] Fix JSON serialization crash — `button_pos_weight` Nx.Tensor in config.ex and registry.ex
      now converted to list before Jason.encode
- [x] Add per-button press-rate diagnostic after each epoch (predicted vs actual, COLLAPSE flag)
- [x] Update `compute_pos_weights` formula: sqrt((1-rate)/rate) instead of (1-rate)/rate.
      Raw inverse frequency caused 3-8x over-prediction; sqrt produces ~1-2x ratios.
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

## HuggingFace Melee Dataset (altf4 Public SLP Dataset v3 + mimic-melee)

erickfm uploaded altf4's Public SLP Dataset v3 to HuggingFace (~95k replays, CC0).
Also published pre-processed PyTorch shards (mimic-melee) for the MIMIC imitation bot.
See `docs/reference/HUGGINGFACE_DATASET_MAPPING.md` for full column-by-column mapping.

**Links:**
- Raw .slp replays: https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7
- Pre-processed .pt shards (2.59 TB, 1.81B frames): https://huggingface.co/datasets/erickfm/mimic-melee
- Subset .pt shards (26.7 GB, 18.7M frames): https://huggingface.co/datasets/erickfm/mimic-melee-subset
- Frame extractor tool: https://github.com/erickfm/slippi-frame-extractor
- Credit: altf4, nikki, yashichi (replays), erickfm (HF upload + processing)

**Note:** Discord message called these "frame-melee" but actual HF repos are "mimic-melee".
The mimic-melee .pt shards are pre-baked for MIMIC's pipeline (z-score normalized, K-means
discretized sticks, self-controller excluded) — NOT directly usable by ExPhil. Use raw .slp
files or run slippi-frame-extractor to get raw parquet.

### Phase 1: Raw .slp replays — direct use with existing pipeline (no new code)
- [ ] Download raw .slp dataset (or subset) from HF to local/cloud storage
- [ ] Test ingestion through existing Peppi → ReplayParser → Training.Data pipeline
- [ ] Assess replay quality distribution (ranked vs unranked, skill levels, character coverage)
- [ ] Filter by character for low-tier specialist training (Mewtwo, Ganondorf, Link, etc.)
- [ ] Run architecture benchmark on this dataset (larger + more diverse than current replays)

### Phase 2: Second ingestion path (new code needed)
Two options — run slippi-frame-extractor for raw parquet, or read mimic-melee .pt shards directly.

**Option A: Run slippi-frame-extractor → raw parquet (cleanest)**
- [ ] Run slippi-frame-extractor on downloaded .slp files to produce raw parquet
- [ ] Add Explorer-based parquet reader to ExPhil (`lib/exphil/data/parquet_loader.ex`)
- [ ] Map columns → ExPhil embedding format (all fields match, no speed gap — see mapping doc)
      Sticks use same [0,1] range as ExPhil, speeds are already decomposed into 5 fields
- [ ] Benchmark parquet load speed vs Peppi .slp parsing for same data
- [ ] Write `scripts/load_parquet_dataset.exs` for download + conversion

**Option B: Read mimic-melee .pt shards (larger dataset, pre-split)**
- [ ] Write .pt shard reader (Python helper or Rust NIF to extract tensors → Nx format)
- [ ] Un-normalize continuous features using norm_stats.json (multiply std + add mean)
- [ ] Decode categoricals using cat_maps.json
- [ ] **Blocker:** Self-controller inputs excluded from states — targets dict has discretized
      controls (60 K-means stick clusters, not raw values). May need to map cluster centers
      back to continuous or accept coarser targets.
- [ ] Streaming shard loader for the 2.59TB full dataset

**Shared:**
- [ ] Use subset for CI smoke tests or quick architecture iteration
- [ ] Consider enriching ExPhil embeddings with extra fields available in dataset:
      stage geometry (blastzones, platforms, edges: 19 dims), ECBs (16 dims/player),
      hitlag_left, invuln_left (frame counts vs boolean)

### Data quality & analysis
- [x] Compare dataset schema against ExPhil's ReplayParser — see HUGGINGFACE_DATASET_MAPPING.md
      **Result:** All 5 decomposed speed fields present. Stick ranges match. No critical gaps.
      Extra fields (ECBs, blastzones, platforms, hitlag) could enrich embeddings.
- [x] Verify controller input encoding — sticks in [0,1] with 0.5=center (same as ExPhil).
      8 buttons + main_x/y + c_x/y + l_shldr = 13-dim match. r_shoulder available but not embedded.
- [ ] Inspect actual .pt shard to confirm character/action/stage/facing/on_ground tensor keys
      (not in norm_stats since they're categorical/boolean, not z-scored)
- [ ] Assess character distribution across 95k replays — may help with class imbalance work
- [ ] Check game boundary handling: .pt shards use offsets array, parquet uses per-file splits

## Architecture Evaluation
- [ ] Jamba 20-epoch convergence test (with kCudaAsync allocator + val_split)
- [ ] Compare top architectures at 20 epochs (H3, Zamba, MinGRU, Mamba)

## Mode Collapse / Class Imbalance (Jamba eval 2026-03-04)

Jamba 20ep Mewtwo: 85.3% overall but all buttons predicted 0%, sticks collapse to neutral.
High accuracy is artifact of class imbalance (buttons pressed ~5% of frames).
Not Jamba-specific — any architecture will collapse the same way on this data distribution.

**Fixes (ordered by expected impact):**
- [x] Focal loss (gamma=2.0) — enabled by default, working
- [x] Per-head class weighting — sqrt inverse frequency via `compute_pos_weights_from_rates`
      Auto-computed from `Data.stats().button_rates`. Validated: L pred=13.4% vs actual=11.9%
- [x] Eval metric: per-button pred/actual press-rate ratio with COLLAPSE flag (shown after each epoch)
- [ ] Action-conditional oversampling — 2-4x oversample frames with button presses
- [ ] Lower button sigmoid threshold at inference (quick win, doesn't fix training)
- [ ] Sweep focal_gamma (1.0-3.0) and button_weight (1.0-5.0) on larger dataset
