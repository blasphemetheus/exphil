# Behavioral-Cloning Failure Modes: Dolphin Diagnosis Playbook

**Purpose:** When you watch your bot in Dolphin (Step 3 of the
[Mewtwo Ship Runbook](../planning/MEWTWO_TRAINING_PLAN.md)), you will *not* get a bot that plays
well on the first try. That's expected. The skill is reading *which* failure mode you have from a
few seconds of footage, then reaching for the **one** matching lever — not tuning five knobs at once.

Every remediation below is a real ExPhil flag/option. All training flags go to `scripts/train.exs`.

---

## How to use this

1. Watch ~30 seconds of the bot vs a level-9 CPU.
2. Match what you see to a **symptom** row below.
3. Apply the **first** listed fix, retrain (runbook Step 1), re-watch. One change per loop.
4. If a symptom isn't here, note it in `docs/reference/GOTCHAS.md` so the next person has it.

BC has a hard ceiling: it can only imitate states present in the data. Several symptoms below are
*not bugs* — they're the imitation ceiling, and the fix is more data or a move to RL, not a knob.
Those are marked **[ceiling]**.

---

## Symptom → cause → fix

### 1. Freezes, or holds a single input forever
**Cause:** Mode collapse. The policy collapsed to the most frequent action (usually "no buttons,
neutral stick"). Melee frames are dominated by no-input, so an under-regularized loss learns to
always predict it.

**Confirm:** Training already warns — `⚠ COLLAPSE WARNING: diversity=1, max button prob=0.98` from
the `Diagnostics` callback. The per-button predicted-vs-actual table shows predicted rates near 0.

**Fix, in order:**
- `--focal-loss --focal-gamma 2.0` (down-weights easy majority frames) — should already be on; if
  collapse persists, raise gamma to `3.0`.
- `--button-pos-weight auto` — upweights positive button presses against class imbalance. (Known
  footgun: this is ignored when combined with `--preset` — see TODO.md. Pass it *without* a preset.)
- `--entropy-weight 0.01` — penalizes over-confident, low-entropy output.
- At inference, sample instead of argmax: the agent supports `:temperature` and `:deterministic`
  (`lib/exphil/agents/agent.ex`). A collapsed *argmax* policy can still look alive with
  `temperature: 1.0`, `deterministic: false` — but that's masking, not fixing, collapse in the weights.

### 2. Twitchy / jittery — never commits to a direction
**Cause:** Per-frame independent sampling with no temporal smoothing, and/or coarse stick
discretization making the policy flip between adjacent buckets.

**Fix, in order:**
- Lower inference temperature toward `0.5` or set `deterministic: true` (agent opts) — the single
  biggest lever for twitch. Jitter is usually a *sampling* artifact, not a weights problem.
- `--kmeans-centers 21` — K-means stick clusters instead of the 17 uniform buckets; puts bucket
  boundaries where real inputs actually are (see `scripts/train_kmeans.exs`).
- Confirm you trained a *temporal* backbone (GRU/LSTM/Mamba have temporal on by default via
  `train.exs`). A non-temporal MLP has no notion of input persistence and will always twitch.

### 3. Plays neutral, but SDs / never recovers off-stage **[ceiling]**
**Cause:** High-tier replays contain few off-stage-recovery frames, and none where *this* bot got
knocked off. BC has never seen the recovery state distribution, so it acts randomly there.

**Fix:**
- Not a knob. This is the imitation ceiling. Options: (a) more replays that include recoveries,
  (b) `--online-robust` won't help here, (c) the real answer is **self-play / PPO** to explore
  recovery via reward — that's the post-ship RL phase (`scripts/train_ppo.exs`,
  `scripts/train_self_play.exs`), not a BC fix. Ship the neutral-game bot first; recovery is v2.

### 4. Reasonable neutral, whiffs punishes / inconsistent combos **[ceiling]**
**Cause:** Data volume. At 129 replays the model sees each punish situation a handful of times.

**Fix:** More replays (500+ noticeably helps — see runbook data table). Not a hyperparameter. Don't
burn loops tuning this; it's expected at 129 and it's a post-ship data lever.

### 5. Plays fine offline, falls apart online / with input lag
**Cause:** Netplay adds 2–4 frames of input delay the model never saw in clean replays.

**Fix:**
- `--online-robust` (bundles frame-delay augmentation) or `--frame-delay-augment` directly. Train a
  dedicated online build; keep the clean build for offline/CPU playtesting.

### 6. Loss looks great but Dolphin behavior is bad
**Cause:** Val loss on next-frame prediction does not measure *closed-loop* play. Small per-frame
errors compound over a match (covariate shift — the bot drives itself into states not in the data).
This is the canonical BC gap and it's why Step 3 exists.

**Fix:** Don't chase lower val loss past this point — it's the wrong metric now. The levers that
close the closed-loop gap are: data augmentation (`--augment --mirror-prob 0.5 --noise-prob 0.1`),
online-robustness, and ultimately RL fine-tuning. Trust the footage over the number.

### 7. NaN loss / training diverges immediately
**Cause:** Backbone-specific LR sensitivity (H3, TTT, Jamba, Zamba are known-fragile per
BENCHMARK_RESULTS.md), or an init issue.

**Fix:** `--lr 1e-5` and `--max-grad-norm 0.5` (clip). For GRU this shouldn't happen — if it does on
GRU, suspect a data/embedding problem, not the backbone. Check `docs/reference/GOTCHAS.md` for the
EXLA/BinaryBackend closure gotchas.

---

## Quick lever reference

| Lever | Flag | Fixes |
|---|---|---|
| Focal loss | `--focal-loss --focal-gamma N` | Collapse, rare buttons |
| Positive-class weight | `--button-pos-weight auto` | Collapse (no preset!) |
| Entropy bonus | `--entropy-weight N` | Over-confidence, collapse |
| Inference temperature | agent `:temperature` / `:deterministic` | Twitch, collapse masking |
| K-means sticks | `--kmeans-centers 21` | Twitch, coarse movement |
| Data augmentation | `--augment --mirror-prob --noise-prob` | Covariate shift, overfit |
| Netplay robustness | `--online-robust` | Online-only breakdown |
| Grad clip / low LR | `--max-grad-norm 0.5 --lr 1e-5` | NaN / divergence |
| EMA | `--ema --ema-decay 0.999` | Noisy late-training weights |

**The meta-rule:** if a symptom is marked **[ceiling]**, stop tuning and either add data or accept
it for v1. BC gives you a bot that imitates; it does not give you a bot that problem-solves in
states it never saw. That's what the RL phase is for — after the ship.
