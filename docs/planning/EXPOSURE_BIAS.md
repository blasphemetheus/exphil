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

- [x] **2. Synthesize the correction set without rollouts — WORKS.** Live,
  deterministic, vs a level-1 Fox CPU, 2 min each:

  | | self-initiated shines/min | max chain |
  |---|---|---|
  | baseline (3 seeds) | 11.4 · 3.0 · 5.2 | 1 · 1 · 1 |
  | **synth (3 seeds)** | **58.4 · 36.2 · 29.4** | **5 · 3 · 6** |

  Ranges 3.0–11.4 vs 29.4–58.4: no overlap, nearest points 2.6x apart, on
  n=3 both sides. Rate and chain length do NOT track each other (the weakest
  seed on rate, 29.4, is the best on chain, 6) — "shines often" and "sustains
  a cycle" look like partly independent capabilities, worth separating in any
  future scoring.

  No overlap, 3x gap between nearest points — legible where the agreement
  metric was pure noise (item 0). Every baseline shine was ISOLATED (max chain
  1); the synth policies chain, which is multishine behaviour appearing in a
  trained policy for the first time.

  Shines are split self-initiated vs hit-induced because Bradley observed the
  CPU jabbing the bot out of a held shine — so raw shine count is contaminated
  by how often the opponent hit it. Baselines: 15-18 hit-induced vs 6-20 self,
  i.e. roughly half their shines were the opponent's doing. Synth: 115 self vs
  21 hit-induced. The gain is genuine recovery, not more jabs.

  Cost: seconds of synthesis + ~6 min training, no GPU, no Dolphin, no rollout
  loop — against DAgger's ~3 GPU-less hours that did not converge.

  Caveats: n=2 synth (the third run's replay truncated), one drill only, and
  max chain 3-5 against the teacher's 791. A step, not a solution. The
  synthesis only reaches states reachable by EXTENDING segments the fixture
  already visits; getting hit, ledge and tech situations still need recordings.

  **Generalise this.** The recovery rules already existed and were only ever
  executed LIVE, never trained on. Every scripted expert has the same latent
  asset — MewtwoFairExpert, MewtwoTechChaseExpert, MewtwoPunishExpert,
  FoxRecoveryExpert. Free training signal across the whole drill suite.

- [ ] **2b. Apply the same synthesis to the other drills.** The recovery rules
  in MewtwoFairExpert, MewtwoTechChaseExpert, MewtwoPunishExpert and
  FoxRecoveryExpert are all trained-on-never, executed-live-only, exactly as
  the multishine ones were. `ExPhil.Data.RecoverySynth` takes an `:expert` and
  an `:actions` set, so pointing it at another drill is mostly picking which
  actions that drill gets stuck in — run the diagnostic first
  (`eval_policy_on_rollout.exs` names the most-visited unseen states).

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

## 0b. RESOLUTION FLOOR — we can no longer measure small effects

Synth's own spread is 29.4–58.4 self-shines/min: a 2x range across seeds.
Anything smaller than that is invisible at n=2-3 with 2-minute runs. Measured
2026-07-26:

| condition | self-initiated shines/min | max chain |
|---|---|---|
| baseline (n=3) | 11.4 · 3.0 · 5.2 | 1 · 1 · 1 |
| synth (n=3) | 58.4 · 36.2 · 29.4 | 5 · 3 · 6 |
| synth + prev-action (n=2) | 32.3 · 44.3 | 6 · 4 |
| synth + noise 0.02 (n=2) | 39.4 · 41.2 | 1 · 3 |

Both additions land INSIDE synth's range. "No detectable gain" here does NOT
mean "no gain" — it means this setup cannot tell. Distinguishing a 20-30%
improvement needs many more seeds or much longer runs, and the binding cost is
LIVE EVAL (~4 min/run), not training (~2 min).

**Consequence: stop tuning knobs.** Further micro-variations on synthesis are
unmeasurable until the measurement gets cheaper or tighter. Prefer the
remaining big swings (items 4/5 below), or first invest in the measurement:
longer runs, batched evaluation, or an offline proxy that correlates with live
chain length.

Also: **2 of 6 runs lost to truncated replays** (graceful SD failed, ~20% flake,
also hit ms_synth_c). Each failure costs a seed. Worth fixing before any
larger seed sweep.

- [-] **3b. prev-action x synthesis — NO DETECTABLE GAIN (n=2, below
  resolution).** Prediction was that Melee's press-EDGE requirement plus the
  recovery rules' alternation on previously-landed input meant synthesis was
  teaching a press probability the policy could not condition properly. Result
  32.3 / 44.3 vs synth's 29.4-58.4: inside the noise. The reasoning may still
  be right; the experiment cannot resolve it. Original note follows.

- [-] **3b-original. prev-action x synthesis — TRY THIS FIRST.** Highest expected value
  per minute of the untried items, and theoretically motivated rather than a
  guess. Melee registers shine on a press EDGE, and MultishineExpert's recovery
  rules alternate on the PREVIOUSLY-LANDED input ("press when the button was
  up, release when it was down"). Synthesis is currently teaching "tap X at
  reflector af 3+" to a policy that CANNOT SEE whether X was already held — so
  it can only learn a marginal press probability, never the alternation. That
  is a plausible reason max chain stalls at 3-6.

  It also reframes the earlier null result: `--prev-action` alone did nothing
  BECAUSE there was no recovery data to condition on. The two may only work
  together, which is exactly the interaction a single-variable ablation misses.
  One flag combination, ~6 min per seed. Laptop.

- [ ] **3c. Widen synthesis beyond the reflector.** `RecoverySynth` currently
  extends only actions 360-368, but the policy also drifts in jumpsquat (24),
  aerial jump (25) and landing states. Same machinery, different `:actions`
  set. Laptop, ~10 min.

- [ ] **3d. Sweep `--synth-ratio` / `--synth-max-af`.** 1.0 and 30 were picked
  without testing. Higher ratio = more recovery coverage but risks swamping the
  core loop. 3 seeds x 3 ratios is ~1 h of training plus live eval, and the
  live eval (~4 min/run) is the bottleneck. Laptop.

- [ ] **3e. Longer live evals.** Runs are 2 min, i.e. only ~90 shine
  opportunities. Max chain 3-6 vs the teacher's 791 may partly be a sampling
  limit rather than a capability limit. One 5-min run settles it. Laptop, cheap.

- [-] **4. Noise injection (DART-style) — WIRED, no detectable gain (n=2).**
  `--noise SCALE [--noise-prob P]` on train_multishine_policy.exs, using the
  existing `Augmentation.maybe_add_noise/2` (continuous fields only — never
  action state, buttons or flags, which would corrupt the label). Result at
  scale 0.02: 39.4 / 41.2, inside synth's 29.4-58.4 range. Only one scale was
  tried; like 3b this is below the resolution floor rather than disproven.

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
