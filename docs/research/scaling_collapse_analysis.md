# Scaling Collapse: Why 200 Files Collapses but 100 Doesn't

## Problem

A 3.7M param Mamba SSM trains fine on 100 replay files (action diversity 16/35 by epoch 5) but collapses to predicting "neutral/do nothing" on 200 files. Every hyperparameter combination fails:
- f32 and bf16
- focal_gamma 0.5, 1.0, 3.0
- LR 1e-4, 3e-4
- constant, cosine_restarts schedules
- AdamW, RAdam
- entropy_weight 0.01, 0.1
- button_weight 2.0, 5.0
- Curriculum learning (resume from 100-file checkpoint)
- MLP backbone (also collapses — not architecture-specific)

150 files works (wobbly). 175 files TBD.

## Root Cause

**Distributional shift amplification under class imbalance at scale.**

~70% of frames are neutral (no buttons, sticks centered). At 100 files, the 30% action frames provide enough gradient signal. At 200 files, the neutral proportion increases (more diverse replays include more idle/neutral time), pushing past the tipping point where cross-entropy's gradient basin around "predict neutral" becomes inescapable.

This is NOT a hyperparameter problem. No amount of LR tuning, optimizer changes, or regularization fixes a data distribution issue. The model correctly learns that "predict neutral" minimizes loss for the majority of samples.

## Solutions (Priority Order)

### 1. Per-Frame Loss Weighting (Highest Impact, Lowest Effort)

Compute a weight for each frame: `weight = 1.0 if any_action else 0.25`. Multiply per-sample loss by this weight before reduction. This reduces the gradient contribution of neutral frames without changing the data.

**Implementation:** Add `frame_weights` tensor to each batch during batch creation. Pass to loss function. Multiply per-sample loss before `Nx.mean`.

### 2. Per-Head Loss Normalization

Don't let the button head (easy to minimize by predicting "none") dominate the total gradient. Normalize each head's loss contribution:
- Simple: `total_loss = mean(loss_head_i / stop_gradient(loss_head_i))`
- Better: Learned per-head weights (GradNorm / uncertainty weighting)

### 3. Frame Skip (Predict Every Nth Frame)

Training on every frame at 60Hz means 70%+ is redundant neutral. Predict every 3rd or 4th frame — drastically reduces neutral dominance. Action repeat for skipped frames.

**Implementation:** In lazy batch creation, stride the action targets by N frames instead of 1.

### 4. Action-Density Chunk Filtering

Compute `action_density = frames_with_action / total_frames` per sequence. Discard sequences where density < 0.1 (pure neutral segments). Or stratify: 50% of each batch from high-density, 50% from mixed.

**Implementation:** Already have StratifiedSampling callback — wire it into the batch stream.

### 5. Auxiliary "Action Change" Prediction

Add a head predicting whether the action changes in the next K frames. Forces the model to distinguish "about to act" from "continuing to idle."

### 6. Mixture Density Output

Replace single softmax per head with a mixture. Explicitly models multimodality — prevents mean-action (neutral) collapse.

## Key Insight from slippi-ai

vladfi1's project used:
- Action-state conditioning (game state features)
- Replay quality filtering (skilled players only)
- Frame skip
- Separation of "do I act?" from "what action?"

## Threshold Data

| Files | Frames | Batches/Epoch | Collapse? |
|-------|--------|--------------|-----------|
| 100 | 630K | 8,406 | No |
| 150 | 950K | 12,730 | Wobbly |
| 175 | 1.1M | ~14,200 | TBD |
| 200 | 1.26M | 15,760 | Yes |

## References
- [Towards Balanced BC from Imbalanced Datasets](https://arxiv.org/html/2508.06319v1)
- [slippi-ai](https://github.com/vladfi1/slippi-ai)
- [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115) (Kendall et al.)

## Pre-experiment audit (2026-05-19)

Audited the in-tree per-frame loss weighting implementation before re-running scaling experiments. Confirmed correctness of every wire-up; one criterion gap noted; one **premise reality-check failed** that warrants revisiting before declaring the fix valid or invalid at 200 files.

### Implementation correctness (all pass)

| Check | File | Verdict |
|-------|------|---------|
| 1a — Loss math `sum(L·w)/sum(w)` with all 6 heads, no stop_grad on weights | `lib/exphil/networks/policy/loss.ex:149-186` | ✅ correct |
| 1b — Train-loop reads `:frame_weights` unconditionally; eval path is correctly UN-weighted | `lib/exphil/training/imitation/train_loop.ex:255,298,348,438`, `lib/exphil/training/imitation/loss.ex:182,333` | ✅ correct |
| 1c — Bucket-8 stick center matches `controller_to_action` discretization | `lib/exphil/training/data.ex:2313-2322,802-839` | ✅ but see gap below |
| 1d — Both lazy (`Nx.Batch`) and eager batch paths attach `:frame_weights`; GPU transfer applied; prefetcher passes through | `lib/exphil/training/data.ex:2284-2300`, `lib/exphil/training/prefetcher.ex` | ✅ correct |
| 1e — Empirical batch distribution sane | (see below) | ⚠️ premise diverges from data |

### Criterion gap (1c)

`compute_frame_weights/2` checks button presses and main-stick deflection but **misses c-stick deflection and shoulder press**. Empirically (5 huggingface replays, 42,616 frames, port 1) this affects 1.88% of frames — modest but conceptually wrong, since isolated c-stick aerials and isolated shield/L-cancel frames are explicit actions in high-level play.

Recommended one-line addition to `compute_frame_weights/2`:
```elixir
cstick_moved = (action[:c_x] || 8) != 8 or (action[:c_y] || 8) != 8
shoulder_pressed = (action[:shoulder] || 0) != 0
if any_button or stick_moved or cstick_moved or shoulder_pressed, do: 1.0, else: neutral_w
```

### Premise check (1e) — surprising

Ran `scripts/audit_frame_weights.exs` on 5 huggingface high-tier replays (42,616 frames, port 1, 17-bucket discretization):

| Criterion | Action frames | Neutral frames |
|-----------|--------------:|--------------:|
| Current (buttons + main stick) | 33,730 (**79.15%**) | 8,886 (**20.85%**) |
| Proposed (+ c-stick + shoulder) | 34,530 (**81.03%**) | 8,086 (**18.97%**) |

The Solutions section above states "~70% of frames are neutral" as the root cause for mode collapse at 200 files. On the actual huggingface dataset that figure is **inverted** — only ~20% of frames are neutral. High-tier play involves near-constant input (dashes, wavedashes, shield-tilts), not 70% idle time.

**Implications:**
- The per-frame weighting fix is still *directionally* correct — neutral is still the largest single class — but its expected effect size is much smaller than the design assumed. Downweighting 20% of samples by 0.75 changes average loss weight from 1.0 to 0.85; the neutral class's gradient contribution drops from 20% to ~6%, not from 70% to ~22%.
- If 200-file training is genuinely collapsing to "predict neutral," the cause may not be neutral-frame dominance but rather: (a) higher dataset diversity making the cross-entropy basin around neutral more attractive even at 20%, (b) interaction between focal loss and the action-class long-tail, or (c) something orthogonal (LR schedule, data loader determinism, etc.).
- Confirming or refuting this matters before declaring the fix "doesn't scale" if 200-file results don't improve.

### Verdict

Implementation is correct end-to-end. Two items deserve consideration before re-running the 200-file experiment:
1. Tighten the criterion (c-stick + shoulder) — small but free correctness win.
2. Treat the 200-file experiment as a test of *both* the fix and the framing premise. If the fix's effect at 200 files is modest, that's evidence the root cause is not neutral dominance and we need to look elsewhere.
