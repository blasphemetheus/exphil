# Exposure bias in the drill policies — what we know and what to try

Tracking doc for the root cause found 2026-07-26 (GOTCHAS #81, "ROOT CAUSE
FOUND"). Task #8's phase-2 fix turned out to address the wrong problem; this is
the right one.

## The evidence

| finding | number |
|---|---|
| policy's own rollout that is OFF the training manifold | 74–79% (four runs) |
| frames stuck in grounded reflector (361), deterministic | 97.8% |
| `action_frame` seen in training for 361 | **1..2** |
| `action_frame` occupied live for 361 | **3..28** |
| expert corrections needed on visited frames (DAgger aggregation) | **89%** |
| offline teacher-forced agreement (same policy) | 99.9% |

The trap is absorbing: holding B keeps it in the reflector, `af` grows, and it
drifts further from anything it has seen. Nothing pulls it back.

## Ruled out, each with evidence

| hypothesis | verdict |
|---|---|
| undertrained | **no** — 99.9% offline, exact press-rate match, loss 0.0016 |
| missing prev-action channel | **no** — retrained with it, channel verified active (Agent logs "feeding own outputs back"), still froze |
| af convention shift (GOTCHAS #81) | **no** — moves 2/288 dims, no live effect; model later proven invalid outright |
| stochastic vs deterministic sampling | **inverted** — sampling HELPED |

That last row is the most informative result we have. The stochastic run's 66
shines came from random RELEASES accidentally knocking the policy back onto its
trajectory. It was never multishining; noise was doing the work. Deterministic
decoding removes the noise and it freezes solid.

## Caveat on our own metric

"Off-manifold" is pointwise `{action, action_frame}` membership, but the policy
is a GRU over 16 frames and `af` enters as a CONTINUOUS normalized value
(af/60 — so af 3 is 0.05 vs af 2 at 0.033, not categorically novel). The real
shift is in the 16-frame TRAJECTORY context; pointwise novelty is a proxy that
probably overstates it. Conclusions hold, the number is a proxy. See the header
of `scripts/eval_policy_on_rollout.exs` for the two confounds in its agreement
columns (use them as a between-policy delta on a fixed rollout, never as
absolute quality).

## 0. METHODOLOGY FIRST — the metric is noisier than the effects

Found 2026-07-26 while running item 3. Off-manifold agreement on a FIXED
rollout, across repeated trainings of IDENTICAL configs, ranges **44–77%**.
Three window-16 runs alone spread 43.1 / 51.7 / 48.5.

**Consequence: every n=1 before/after comparison in this thread is
uninterpretable, including ones already recorded.** The DAgger round-1 result
(probe 59.4% → dagger 37.4%) was reported as "backwards from DAgger's purpose";
that gap sits INSIDE this noise band and should not be read as a regression.
GOTCHAS #81 carries the same caveat.

**Protocol for anything below, from now on:**

1. Train **≥3 seeds** per configuration; report the spread, never a single run.
2. Compare distributions, not points. A change under ~30 points on this metric
   is not evidence of anything.
3. Keep the rollout fixed across the whole comparison — the numbers are
   rollout-specific as well as run-specific.
4. Prefer a behavioral measure with a tighter distribution where one exists:
   shine COUNT and max chain length from `ExPhil.Eval.ShineChain` are direct,
   and a 9-frame gap either happens or it does not.

This is the second time in one session that repeated-looking evidence turned
out to be one measurement in disguise (the other: the af table's three
"agreeing" recordings were three repetitive recordings). Default to suspicion
when a number is stable across runs that share structure.

## Things to try

Status: [ ] untried · [~] in progress · [x] done · [-] ruled out

- [~] **1. DAgger** — aggregate policy-visited states with expert labels.
  Aggregation demonstrably works (89% of frames corrected). Round 1 was
  INCONCLUSIVE: plateaued ~0.045 against a 2e-3 target and its off-manifold
  agreement got worse (59.4% → 37.4%), but that is an epoch-60-of-200 snapshot,
  not a converged policy. Note 2e-3 is a MEMORIZATION target inherited from the
  single-fixture trainer and is probably the wrong bar for a 10x more diverse
  aggregate. **Needs the GPU** (~2.6 min/epoch here). Resume from
  `checkpoints/multishine_dagger1_policy.bin.trainer.ckpt` with `--resume`.
  Rollouts re-recorded 2026-07-26 against a REAL CPU (the earlier ones were
  collected against an idle HUMAN port — GOTCHAS #57b).

- [~] **2. Synthesize the correction set without rollouts.** `MultishineExpert`
  ALREADY has recovery rules for exactly the off-table states the policy falls
  into; today they only run live. Enumerate off-manifold states — reflector at
  af 3..28 above all — label them with those rules, mix into training. DAgger's
  benefit without the rollout loop, aimed straight at the known trap. Laptop.

- [x] **3. Context-length ablation — NO EFFECT, but it found something worse.**
  Trained window 16/8/4/2 (all memorize equally: loss 0.00128–0.00182) and
  scored each on a fixed rollout. Off-manifold agreement, repeated runs of
  IDENTICAL configs:

  | window | runs |
  |---|---|
  | 16 | 43.1 · 51.7 · 48.5 |
  | 8 | **76.9** · 44.0 · 48.5 |
  | 4 | 46.4 |
  | 2 | **71.0** · 71.0 · 47.2 |

  The high scores did not replicate. There is no context-length effect visible
  through this much noise, so the hypothesis is neither confirmed nor refuted —
  see item 0, which is the real result.

- [ ] **4. Noise injection (DART-style).**
  `ExPhil.Training.Augmentation.add_noise/2` and `maybe_add_noise/2` already
  exist and are NOT used by the multishine trainer. Perturbing states during
  collection widens training into a tube around the trajectory instead of a
  line — the standard cheap alternative to iterative DAgger. Laptop.

- [ ] **5. Teacher-driven recovery data.** The closed-loop teacher holds **791
  unbroken cycles**. Start it from perturbed / off-trajectory states and record
  how it recovers: on-policy-adjacent data with a perfect labeller and no policy
  rollouts. Needs Dolphin, laptop-capable.

- [ ] **6. Scheduled sampling.** Feed the model its own predictions for a
  fraction of training frames. Nothing implements this today; textbook
  exposure-bias fix, fits the existing train loop. Laptop.

- [ ] **7. PPO fine-tuning from the BC policy.** Infrastructure exists.
  Optimizes the CLOSED-LOOP objective (shine count) rather than one-step
  imitation, which is the thing that actually diverges. GPU.

## How to judge any of them

`mix run scripts/eval_policy_on_rollout.exs --policy X --rollout R` on a FIXED
rollout, comparing off-manifold agreement before vs after. The absolute numbers
are confounded; the delta is not. Live confirmation needs
`--seconds N` so the replay finalizes, and `check_replay_ports.exs --expect-cpu 2`
so the opponent is real.
