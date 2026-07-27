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
- [x] Wire exphil training to use XLA custom call for selective scan
      Already done — Edifice.CUDA.FusedScan.selective_scan dispatches via Nx.Shared.optional.
      EXLA_EXTRA_CUDA_DIR compiles the kernel into libexla.so. No ExPhil wrapper needed.
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

## Make the bot multishine better (exposure bias)

Full context, evidence and caveats: [docs/planning/EXPOSURE_BIAS.md](docs/planning/EXPOSURE_BIAS.md).
Root cause is EXPOSURE BIAS, not the state-stream shift. The synthesized
recovery set already took self-initiated shines from 3.0–11.4/min to
29.4–58.4/min and max chain from 1 to 3–6 (n=3 both sides, no overlap).

Score every change with `scripts/analyze_shine_source.exs` — self-initiated
shines and max chain, **≥3 seeds**, never the off-manifold agreement metric
(44–77% run-to-run noise at one config). Rate and chain length do NOT track
each other; report both.

Ordered by expected value per minute on the laptop:

- [ ] **1. prev-action channel + synthesis TOGETHER.** Highest value and the
      only theoretically-motivated one. Melee registers shine on a press EDGE,
      and the recovery rules alternate on the previously-landed input — so
      synthesis currently teaches "tap X at reflector af 3+" to a policy that
      cannot SEE whether X is held. It can learn a marginal press probability,
      never the alternation. Plausible cause of max chain stalling at 3–6.
      Also reframes the earlier null: `--prev-action` alone did nothing
      BECAUSE there was no recovery data to condition on. No code — run
      `--prev-action --synth-recovery`. Laptop, ~6 min/seed.

- [-] **2. Widen synthesis beyond reflector states — DROPPED, measured
      pointless.** Compared every action's live af range against its fixture
      range. Only TWO actions ever exceed it, and grounded reflector 361 is
      99.6% of the total:

      | action | fixture max af | live max af | frames beyond |
      |---|---|---|---|
      | 361 | 14 | 27 | **1006** |
      | 42 | 9 | 13 | 4 |

      The reflector IS the trap and synthesis already covers it (extends to
      af 30 vs the live max of 27). Widening would add impossible states as
      well as useless ones: jumpsquat is a FIXED 3 frames, so synthesising
      af 4 there manufactures a state the game cannot produce. Only HOLDABLE
      actions can be extended at all.

- [ ] **3. Noise injection (DART-style).**
      `ExPhil.Training.Augmentation.add_noise/2` and `maybe_add_noise/2`
      already exist and are unused by the multishine trainer. Widens training
      into a TUBE around the trajectory rather than a line — complementary to
      synthesis, which only covers extensions of already-visited segments.
      Laptop, ~15 min to wire.

- [ ] **4. Teacher-driven recovery data (perturbation harness).** The
      closed-loop teacher holds 791 unbroken cycles. Start it from PERTURBED
      states and record how it actually recovers: ground truth instead of
      rule-generated labels, and it reaches states synthesis cannot (getting
      hit, ledge, tech). Needs Dolphin; laptop-capable. Biggest build of the
      set — the harness must set up an off-trajectory state, hand control to
      the teacher, and record only the recovery.

- [ ] **5. Scheduled sampling.** Feed the model its own predictions for a
      fraction of training frames — attacks exposure bias during TRAINING
      rather than via data, so it composes with all of the above. Nothing
      implements it today; largest code change of the laptop-viable set.

- [ ] Sweep `--synth-ratio` / `--synth-max-af` (1.0 and 30 were picked
      untested), and try a longer live eval — 2 min gives only ~90 shine
      opportunities, so chain 3–6 may be a sampling limit not a capability one.

## Multishine / State-Stream (task #8 follow-ups)

- [ ] **Widen the action_frame table — cheap, laptop-sized, unblocks the
      af_convention default.** "77 of 399" understates it: 399 counts every
      character's specials plus states unreachable in normal play. Measured
      against what we ACTUALLY encounter, across every live trace recorded
      2026-07-26: **98 distinct action states seen, 77 mapped, 21 missing**
      (17 universal <341, 4 character-specific). Missing:
      0, 38, 50, 55, 73, 83, 89, 90, 178, 179, 182, 252, 253, 262, 263,
      322, 324, 344, 352, 363, 364.

      Two of those (322, 324) carry af = -1 sentinels and are unmappable by
      construction. Most of the rest were SEEN but never mapped only because
      their run was killed mid-game, so the .slp was truncated and no pair
      could be diffed — the failure `--seconds` now prevents. So a chunk of
      this gap is already-collected data we threw away.

      What it takes: VARIETY, not volume. Yield per recording so far —
      Fox TAS loop +9, Mewtwo vs level-9 CPU +66, Fox teacher loop +2. A
      repetitive loop adds nothing; a real match adds dozens. Target the
      missing states directly: ledge (252/253), tech/knockdown (178-182),
      grabs and throws, and one varied match per character for its 341+
      specials.

      Cost: ~2 min to record + ~1 min to diff per pair, and it is
      interruptible. Five to ten varied recordings is well under an hour and
      should take mapped coverage of encountered states past 95%. THEN
      flipping `af_convention` to default-on becomes uncontroversial, because
      the "mixed convention" objection disappears once nearly everything
      encountered is mapped.

- [ ] **Play the multishine teacher live: teacher on one port, level-1 CPU on port 2.**
      The teacher currently gets validated against a fixture and via
      `demo_expert.exs`; running it against a passive low-level CPU gives a
      live behavioural check with an opponent present but not interfering
      (a level-1 CPU mostly idles, so a broken loop is unambiguous rather
      than "it got hit"). Use `--dummy cpu --dummy-cpu-level 1`.
      Also the cheapest way to record a NEW state-stream pair covering Fox
      multishine states under live conditions:
      `EXPHIL_STATE_TRACE=1 ... > pair.live-trace.log 2>&1`, then
      `scripts/diff_state_streams.exs` (needs 100% action/on_ground/y).
      Re-verify the teacher after ANY af-convention change (GOTCHAS #81):
      its live success partly rides on table-miss -> recovery-rule luck.

- [x] **Get a policy trained on Fox multishining.** No B2 trip needed — it
      retrains from scratch on a LAPTOP CPU in ~90 seconds:
      `mix run scripts/train_multishine_policy.exs` (GRU, hidden 256, 1
      layer, window 16, 1679 frames from `fox_multishine_closed.slp`).
      Reached loss 0.00161 in 12 epochs, matching the 0.00148 on record.
      Never wait for the GPU for this one.
      Verified with `eval_policy_on_fixture.exs` on that same fixture — a
      MEANINGFUL run, since it presses B 77.7% of frames (unlike the Mewtwo
      fixtures, see Mode Collapse below): **99.9% B/X agreement at delay 0,
      and press rates matching the fixture exactly (B=77.7% X=11.1%)**.
      Delay 1 scores only 66.7%, confirming delay 0 is the right convention.
      Reproduces the 99.3% parsed-space figure on record.

- [x] **Run that multishine policy LIVE with `af_convention: :live`** — DONE,
      NEGATIVE. `--live-af` and `--seconds` flags added to
      play_dolphin_async.exs. 2 min each vs a level-1 Fox CPU: 66 shines OFF
      vs 65 ON, median gap 81f vs 83f. No effect, and the flag was verified
      wired (embedding tensors change). Root cause turned out to be EXPOSURE
      BIAS, not features — see GOTCHAS #81's "ROOT CAUSE FOUND" section.
      Keep `af_convention` default OFF: coverage is 77/399, so enabling it
      puts SOME actions in parsed convention and the rest in live — a mixture
      matching neither training nor current inference. Revisit when coverage
      is near-total or a measured win exists.

- [ ] **DAgger round 1 for multishine — NEEDS THE GPU.** Attempted on the
      laptop 2026-07-26 and abandoned at 30 min. The aggregate is 16882
      frames (1679 fixture + 15203 relabelled rollout, the expert corrected
      **89% of visited frames**) and training runs **~7 min/EPOCH** vs ~6
      s/epoch for the 1679-frame baseline — a ~70x slowdown from 10x data.
      4 epochs in 30 min, loss bouncing 0.112/0.043/0.068/0.071, target 2e-3.
      So: initial multishine training is a laptop job (90 s), DAgger rounds
      are NOT. Re-record rollouts at home (2 min each, cheap) rather than
      vendoring them, then:
        mix run scripts/dagger_drill.exs --expert multishine \
          --rollouts "<a>.slp,<b>.slp" --out checkpoints/ms_dagger1.bin
      Gate the result with `scripts/eval_policy_on_rollout.exs` — compare
      off-manifold agreement before vs after on the SAME rollout (the
      absolute numbers are confounded; the delta is what means something).
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
