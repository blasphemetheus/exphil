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

## 0a. THE VARIANCE IS IN THE RUNS, NOT THE SEEDS

Measured 2026-07-27. FOUR live runs of ONE fixed policy (`ms_synth_a`):

| run | self/min | max chain |
|---|---|---|
| first | 58.4 | 5 |
| v1 | 26.9 | 4 |
| v2 | 40.2 | 5 |
| v3 | 34.2 | 5 |

A **2.2x spread with the network held constant**. So the scatter previously
blamed on training seeds is mostly GAME-to-game randomness — CPU behaviour,
starting positions, luck.

**Every comparison in this doc before this point trained ~3 seeds and did ONE
run each**, which is n=1 on the dimension that actually varies. That is why
prev-action and noise (items 3b, 4) showed nothing: not because the effects
are absent, but because the design had no power.

**Corrected protocol — cheaper AND stronger than what it replaces:**

1. **≥3 live RUNS per policy.** This matters more than seed count.
2. Budget accordingly: a run is ~4 min, a training ~2 min. Three runs of one
   policy beats three policies of one run, at similar cost.
3. Report the mean and range over runs; a difference smaller than ~2x is not
   resolvable without many more runs.
4. Use `scripts/record_until_valid.sh` so a truncated replay costs a retry
   rather than a data point.

The synth-vs-baseline result survives this (mean ~39.9 vs ~6.5, no overlap
anywhere). Nothing smaller does.

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

## 0c. THE HARNESS ITSELF IS PART OF THE MEASUREMENT (2026-07-27)

Investigated whether the eval harness has distributional quirks of its own.
Three findings:

1. **Inference headroom is razor-thin.** AsyncRunner decouples the frame
   loop from inference: the frame loop always sends the LATEST completed
   action, so slow inference silently RE-SENDS the previous action. The four
   post-SD-fix idle runs of `ms_synth_a` ran 1.20–1.28 inferences/frame —
   only 20–30% margin over the 60 Hz cadence on an idle laptop. Any
   background load pushes some frames under 1.0 and each such frame is a
   stale send = a timing slip the policy did not cause. For a 9-frame cycle
   this is a direct, machine-induced chain-breaker, and a plausible
   contributor to the 2.2x run-to-run spread (item 0a).

2. **Under heavy load the harness doesn't degrade — it collapses.** With
   CPU training saturating cores, a live run advanced ~540 game frames in
   ~8 minutes (~1 fps) and never left STAGE_SELECT. Protocol rule: NEVER
   run live evals concurrently with training or any heavy job. (This also
   means a loaded-vs-idle A/B needs a CALIBRATED load — e.g. one busy core —
   not a saturating one.)

3. **Staleness is now measured per run.** AsyncRunner counts stale sends
   and the longest stale run; `play_dolphin_async.exs` prints
   `Staleness: N/frames (x%), longest stale run M` in its final stats.
   Runs with outlier staleness should be discarded as measurement failures,
   the same as truncated replays. CORRECTED after 18 instrumented runs: an
   early "~16% steady state" read did NOT hold — staleness ranged **1.6%
   to 30.8%** across the day, tracking ambient load (cleanest runs 1.6–3.8%
   at true idle; 25–31% in back-to-back chained eval blocks). It is a
   PER-RUN health stat: report it with every score, treat >10% as a
   degraded run, and leave cooldown between eval blocks. Longest stale run
   stayed 1–5 frames throughout — degradation is many single-frame slips,
   not stalls. (Also fixes the n: the four sdfix runs of
   ms_synth_a scored 39.7/53.6/52.5/54.6 self/min, chains 5/6/6/7 — item
   0a's four runs plus these = n=8 idle baseline, mean ~45, range
   26.9–58.4.) SECOND CORRECTION (post-reboot 2026-07-27): loadavg does
   NOT predict staleness either — after a reboot, 9 consecutive runs
   held 0.2–1.9% at loadavg up to 4.9, where the pre-reboot machine
   (15h uptime) gave 17–31% at similar loadavg. Accumulated machine
   state (thermal throttle / memory pressure / background churn) is the
   real driver. Practical rule stands: staleness is a PER-RUN stat,
   read it from the run itself, and a reboot is the cheapest way to a
   clean measurement block.

4. **Training/eval delay mismatch — RESOLVED by the delay campaign
   (2026-07-28..08-03).** The variable-delay harness was engineered away
   (blocking sync, measured intrinsic +2), and delay became a TRAINED
   property: multi-delay pools + delay-id + queue-as-input + SS-on-queue.
   Production: ms_g6_sp1 ({2,3}) covers d2-d4 (434/413/332 via id
   override). Jitter and rung-spacing hypotheses both refuted en route.
   Full record: LATENCY_ARCHITECTURE.md.

5. **The eval OPPONENT is part of the input distribution (2026-07-27).**
   `--dummy stand` (idle opponent) was supposed to be the clean capability
   control. Result: `ms_synth_a` AND `ms_pa_a` — both ~45 self/min vs the
   CPU — score **0/0/0** against it (`ms_synth_a` spends 3465/4429 frames
   CROUCHING; the PA runs were the day's cleanest at 1.6–3.8% staleness, so
   this is policy collapse, not harness noise). Opponent state is embedded;
   an idle Fox at spawn is off-manifold, and non-robust policies key their
   behavior to opponent context. Consequences: (a) eval-opponent choice is
   itself a distribution-shift experiment — a policy must be robust before
   the idle measurement means anything; (b) for policies that survive it,
   the idle eval is FAR less noisy: ms_synth_ss scored 90.6/90.1/92.5
   self/min — a 1.03x spread vs the CPU condition's 2.2x — and with zero
   hit-induced shines it is pure capability. See item 6 for that result.
   CORRECTED same day: the low-noise claim held only in the degraded-
   harness regime. Under a clean harness the same policy spread 7.3–56.1
   (7.7x) vs idle — WORSE than the CPU condition, because the idle eval
   removes the opponent's perturbations and exposes absorption-time
   luck (see 6-replication). Tightness vs idle was an artifact of
   staleness-noise ergodizing the loop, not a property of the eval.

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

- [x] **5. Teacher-driven recovery data — RUN AND MEASURED 2026-07-27:
  UNDERPERFORMS SYNTH ON CHAINS.** First harness run (`PERTURB_EVERY=180
  PERTURB_FRAMES=6` mode random, 120s, level-1 CPU): 7200 frames, ~40
  perturbations, 25.1% of rollout frames relabeled; trained to
  convergence (loss 0.00154, epoch 56). Protocol result (n=3 runs per
  policy, staleness-matched at ~16%):

  | condition | self/min | max chain |
  |---|---|---|
  | synth (n=11 runs) | ~45 (26.9–58.4) | **3–7** |
  | perturb, loss 0.0015 (n=3) | 32.4 · 44.1 · 38.8 | **2 · 2 · 1** |
  | perturb e40, loss 0.09 (n=3) | 33.9 · 44.9 · 47.7 | **1 · 1 · 2** |

  Rate equivalent, chains DON'T overlap — on the LOW side. Two collateral
  findings: (a) the moderate-loss "statue at the floor" lore does NOT
  reproduce here — 0.0015 and 0.09 behave identically (that lore came from
  the conversion metric on r-series drills; don't port it blindly);
  (b) chain_break_forensics on all 6 runs: ~half of breaks UNFORCED, bot
  pools in reflector states after (156–262 frames/run in the #81 trap at
  BOTH loss bands) — the absorbing trap survives perturbation training.

  Suspected causes, untested: fixture dilution (1679 of 8549 frames = 20%
  cycle data vs synth's targeted trap-state density), death/respawn frames
  in the corrections (random mode side-B'd Fox off stage ~3 times), and
  corrections spread thin (~40 recovery episodes). Next levers:
  synth+perturb COMBINED aggregate, `PERTURB_MODE=stick|release`
  re-recording (no deaths), scheduled sampling on top of synth (item 6,
  now implemented).

- [-] **6. Scheduled sampling — MEASURED 2026-07-27: initially read as the
  first clean win since synthesis; **SUPERSEDED same day by 6-replication
  below — does not survive seed replication or a clean harness.** Original
  result kept for the record. `ms_synth_ss` (synth + prev-action + `--scheduled-sampling
  0.5`, train_multishine_policy.exs, 4063 frames, loss 0.001 @ 29 epochs):

  | policy (all synth-trained) | vs CPU self/min · chain | vs IDLE self/min · chain |
  |---|---|---|
  | no PA (`ms_synth_a`) | ~45 (26.9–58.4) · 3–7, n=11 | **0·0·0 — crouches** |
  | PA teacher-forced (`ms_pa_a`) | 32.3 · 44.3 · chains 6·4, n=2 | **0·0·0** (cleanest runs of the day) |
  | PA + SS (`ms_synth_ss`) | 38.9–48.2 · 2·3·3 | **90.6 · 90.1 · 92.5 · chains 4·4·6, 0 hit-induced** |

  The 2x2 isolates it: prev-action ALONE does not survive the idle-opponent
  shift; SS does, at DOUBLE the CPU-condition rate — and it scored that
  under 17–29% staleness (degraded harness). Mechanism as predicted:
  teacher-forced, the PA channel is redundant with the state stream, so the
  model leans on opponent-context features that collapse off-manifold; SS
  decorrelates the channel (sometimes it contains the model's own decode),
  forcing a genuinely self-conditioned loop that transfers. Explains the
  3b null retroactively: PA without SS trains a channel the model ignores.

  Caveats: vs-CPU chains 2-3 measured at the day's WORST staleness (30%)
  — redo under clean conditions before reading that cell; single training
  run (no seed replication); rate not chain length is the robust gain.
  Original implementation note follows. BOTH caveats resolved same day —
  see 6-replication; both went against the result.

- [-] **6-replication. SS REPLICATION FAILED 2026-07-27 (same day, post-
  reboot, cleanest harness ever: 0.2–1.9% staleness all runs).** Two fresh
  training runs of the EXACT recipe (same command, fresh init seeds; both
  reproduced 4063 frames; losses 0.00164@27ep, 0.00187@20ep vs original
  0.001@29ep) plus a clean re-run of the original binary:

  | policy, vs IDLE | self/min | max chain | staleness |
  |---|---|---|---|
  | `ms_synth_ss` (orig, YESTERDAY) | 90.6 · 90.1 · 92.5 | 4 · 4 · 6 | 17–29% |
  | `ms_synth_ss` (orig, TODAY clean) | **7.3 · 18.0 · 56.1** | 2 · 2 · 2 | 0.5–1.2% |
  | `ms_synth_ss_b` | **0 · 0 · 0** (crouches) | 0 | 0.2–0.4% |
  | `ms_synth_ss_c` | **0.8 · 1.6 · 0.8** | 1 · 2 · 1 | 0.2–0.4% |

  Also the vs-CPU redo under clean staleness: orig scored 31.2 · 34.6 ·
  35.9, chains 4·3·2 — INSIDE plain synth's 26.9–58.4 spread. So: no SS
  gain vs CPU (confirmed), and the idle-opponent "win" is (a) seed-
  dependent — 2 of 3 seeds fall into the crouch absorber like the
  controls — and (b) not reproducible even for the winning seed under a
  clean harness. Yesterday's 90/min was measured ONLY in the degraded-
  harness regime.

  **Leading hypothesis — staleness-as-noise:** a stale send repeats the
  previous action for a frame; at 17–29% that is heavy action-repeat
  noise. A deterministic policy (argmax decode, conf 0.97) in the crouch
  fixed point stays there forever; noise makes the membrane leaky and
  keeps kicking the policy back onto the cycle. Supporting spread
  pattern: orig vs idle was eerily TIGHT dirty (1.03x) and WILD clean
  (7.7x) — noise ergodizes the closed loop; clean, absorption time is
  luck of the transient. Same logic explains why vs-CPU runs are tight
  (1.15x) at moderate rate: the OPPONENT is a perturbation source
  (approaches/hits knock the state around, preventing permanent
  absorption).

  **Discriminating experiment RUN (stress-ng, 3 workers): staleness does
  NOT resurrect an absorbed seed.** `ms_synth_ss_b` vs idle at 26.9 /
  30.2 / 29.6% staleness, 3-4-frame slips — a near-exact reproduction of
  yesterday's regime — scored **0/0/0**. (First attempt at full
  saturation hit 56% staleness with 21-28-frame freezes — that regime is
  uninterpretable, a 9-frame cycle can't execute at 4-frame action
  granularity; discard such runs.) This is theoretically clean: a stale
  send REPEATS the previous action, and repeat-of-crouch is crouch —
  action-repeat noise perturbs a policy in motion but is inert inside an
  absorber. Consequences: (a) yesterday's 90/min is still not fully
  explained — for the one marginally-capable seed, the degraded harness
  can only have been altering cycle-break/absorber-ENTRY dynamics, not
  rescuing it from crouch; (b) the three noise families are now
  empirically separated: training-time state noise (item 4, null),
  inference-time action-repeat (inert in absorbers, this experiment),
  inference-time SAMPLING (untested, the only one that generates new
  actions in a fixed point — item 9).

  **The crouch absorber, named:** distinct from the #81 reflector trap.
  Live sequence (observed): imperfect multishine → missed shine after
  jump → full shorthop (airborne ~20f with no shine in the 16-frame
  window = zero training support) → lands holding down (the marginal-
  mode action of multishine data) → Squat/SquatWait forever, at high
  confidence. Candidate coverage fix, untried: extend `RecoverySynth`
  to Squat/SquatWait/landing states labeled with down-B (same machinery
  that killed the reflector trap; item 3c already proposed widening —
  the crouch basin is now the highest-value target). Forensics gap:
  `chain_break_forensics.exs` counts #81-trap pooling but not crouch
  pooling; add Squat/SquatWait to make basin occupancy a tracked stat.

- [x] **6-implementation. Scheduled sampling — IMPLEMENTED 2026-07-27;
  VALIDATED AS THE RECIPE 2026-07-31..08-03.** SS-on-queue broke the d3
  rung (mdq_ss 380.5 c367 vs teacher-forced mdq's 73 c1), buys
  id-mismatch robustness, and is in every champion recipe since
  (ms_g6_sp1: d2 434.5 c434 / d3 413.4 c409). Rule: never combine with
  --shift-jitter (grind-3: inconsistent targets flatten every rung).
  `ExPhil.Training.ScheduledSampling`: with probability P per sample, the
  prev-action slice of the last window position is replaced by the model's
  OWN decoded prediction (decode pinned to the live path: logit>0 buttons,
  argmax/16 sticks rescaled (v-0.5)*2, argmax/4 shoulder). Depth-1,
  last-position-only — game-state dims need a simulator, which is what the
  perturbation harness covers; the two compose. Slice located empirically
  via `Attribution.prev_action_dim_range/1` (layout has scrambled silently
  before). One extra forward pass per step.

  Drill usage: `--scheduled-sampling 0.5 --ss-ramp 10` (linear 0→P over 10
  epochs). Requires `--prev-action`. NOTE: loss under scheduled sampling is
  a harder objective — do not compare loss curves across this flag, judge
  by live runs only (item 0a protocol: ≥3 runs).

  **SS-on-queue (2026-07-31).** With `--queue-depth K` the same flag now
  self-samples ALL K queue slots: slot k is filled with the model's decoded
  prediction on the window truncated by k frames. Alignment argument: queue
  slots are built from already-shifted frames (`shift_actions` relabels
  `:controller` in place), so slot k at position t IS the model's target at
  t-k — truncate-by-k is exact under any `--pipeline-offset`/
  `--shift-jitter`/multi-delay mix, no shift bookkeeping. All slots swap
  together under one per-sample mask (live, every slot is self-generated —
  a mixed queue would be a training-only artifact). Depth flows from the
  dataset's embed config automatically; needs `window > K`. Cost: one extra
  forward per slot (K=4 ≈ 2x step time). Motivation: queue1's teacher-forced
  channel amplified exposure bias (58-73/min vs control 239); jq's d3
  crouch-absorber collapse (0 shines, 4/4 runs, 97% Squat occupancy from
  frame 104) is the same signature.

- [ ] **7. PPO fine-tuning from the BC policy.** Infrastructure exists.
  Optimizes the CLOSED-LOOP objective (shine count) rather than one-step
  imitation, which is the thing that actually diverges. GPU.

- [~] **8. Crouch-basin coverage — IMPLEMENTED 2026-07-27, first policy
  training.** `RecoverySynth.build_crouch/2`: manufactures the crouch
  absorber (a state the teacher NEVER visits, so `build/2`'s segment
  extension can't reach it) by grafting synthetic `Squat(39) ->
  SquatWait(40) af 1..40` tails onto post-shine fixture frames. Tails run
  past the 16-frame window so some training windows are ENTIRELY crouch —
  covering the deep-basin state, not just the entrance. Labels come from
  `MultishineExpert`'s existing grounded fallback ("start a shine"), and
  — unlike `build/2`'s extend, which labels every frame with `prev=nil`
  (a held-button label sequence!) — the tail THREADS each label into the
  next frame's `prev`, so B alternates press/release exactly as the
  expert behaves live. REQUIRES `--prev-action` (the alternation is
  unlearnable without the channel — item 3b's lesson). Wired as
  `--synth-crouch [--crouch-max-af N] [--crouch-ratio R]` in
  train_multishine_policy.exs; 5 tests in
  test/exphil/data/recovery_synth_test.exs. First policy: `ms_crouch_a`
  (synth + PA + crouch, no SS — isolating the coverage variable).
  Forensics support: chain_break_forensics.exs now reports crouch
  occupancy, absorbed spells (>= --absorb frames, default 120) and
  run-length-compressed ENTRY ROUTES per spell.

  **FIRST EVAL (n=1 training run, replication in progress): the sustain
  ceiling broke.** vs IDLE, clean harness (0.7-1.0% stale),
  deterministic decode:

  | run | self/min | max chain |
  |---|---|---|
  | r1 | 129.1 | 22 |
  | r2 | 99.7 | 20 |
  | r3 | 103.1 | 19 |

  Previous best chain from ANY intervention: 7 (open problem #1 said
  "every intervention moves rate, barely moves sustain" — this moved
  BOTH, 2x rate and ~3x chain, zero hit-induced, 1.29x spread).
  Forensics on r1: crouch occupancy 2.4% vs the absorbed seed's 78.3%
  (one 3448-frame spell) — breaks still 100% unforced but now resolve
  through air-reflector/jump states back into the cycle instead of
  terminally pooling. Interpretation: covering the basin turns terminal
  breaks into instant re-entries, which is exactly what stitches chains.
  Loss converged slower (0.00172 @ 80 epochs vs ~27 for non-crouch
  recipes) — the alternating escape labels are a genuinely harder
  objective.

  **Next basin observed live (ledge valley):** with crouch covered, the
  flow finds the next-largest trap — aerial drift near the edge ->
  ledge-grab -> CliffWait hang. A time-sink so far, not an absorber.
  NOT synthesisable with the current expert: its airborne fallback
  (press B) is a NO-OP on the ledge — escape needs a real new rule
  (drop/climb first). chain_break_forensics.exs now counts ledge
  occupancy (252..263) so the valley is tracked before it is attacked.

  **REPLICATED (2 more seeds + vs-CPU, all clean 0.7-1.5% stale):**

  | seed | vs IDLE self/min | vs IDLE chain |
  |---|---|---|
  | a | 129.1 · 99.7 · 103.1 | 22 · 20 · 19 |
  | b | 67.9 · 68.8 · 68.2 | 2 · 2 · 2 |
  | c | 98.3 · 99.1 · 95.1 | 11 · 9 · 8 |

  Seed a vs CPU: 72.9 · 81.4 · 67.8, chains 7 · 9 · 7 — above plain
  synth's 26.9-58.4 / 3-7 in the harder condition too. Verdict, split by
  effect: (1) ABSORBER ESCAPE REPLICATES 3/3 — every seed >= 68/min vs
  idle where the same recipe minus crouch data scored 0/0/0; seed b at
  chains-of-2 shines/breaks/re-enters forever without ever absorbing.
  The coverage effect is a property of the METHOD. Contrast SS
  (6-replication): 0 / ~1 / 7-56 across seeds. (2) SUSTAIN is
  seed-variant (2 -> 22) but per-seed STABLE across runs (22/20/19,
  11/9/8, 2/2/2) — chain ability is a trained-policy property, not run
  luck; the basin fix unlocks the possibility, the init decides how much
  is realised. Next levers for sustain: longer runs (the 60s window may
  clip seed a), SS x crouch composition, seed selection at train time
  (cheap: eval 3+ seeds, keep the best — legitimate now that the
  variance is understood).

  **SEED FARM CORRECTION (2026-07-27 evening, 5 fresh seeds d-h, same
  command, all clean 0.3-0.7% stale, 3x60s vs idle): "escape replicates
  3/3" was itself small-n luck.**

  | seed | self/min | chains | verdict |
  |---|---|---|---|
  | d | 86.3 · 89.7 · 88.8 | 2 · 2 · 2 | escape (b-like) |
  | e | 1.6 · 0.8 · 6.5 | 1 | FAILED |
  | f | 7.4 · 3.3 · 18.7 | 3 · 2 · 3 | FAILED (marginal) |
  | g | 0.0 · 0.0 · 0.0 | 0 | FAILED (dead) |
  | h | 5.7 · 6.5 · 4.9 | 1 · 2 · 1 | FAILED |
  | i | 66.0 · 74.3 · 71.1 | 3 · 3 · 3 | escape |
  | j | 4.1 · 1.6 · 1.6 | 1 · 2 · 1 | FAILED (73.8% crouch) |
  | k | 68.8 · 81.0 · 70.9 | 3 · 4 · 4 | escape |
  | l | 13.9 · 24.4 · 13.0 | 2 · 3 · 3 | FAILED (oscillating) |

  Escape rate at n=12: **6/12 — exactly 50%** (a, b, c, d, i, k).
  Sustain distribution among escapees: chains 2-4 for four of six,
  8-11 for c, 19-22 only for a — the champion is ~1/12 rare, the
  functional-bot tier is a coin flip per seed. Operational recipe:
  train 2-3 seeds, keep any escapee; farm longer only when hunting a
  sustainer. Seed l is a NEW failure variant: six REPEATED absorbed
  spells of ~300-730f with brief escapes between, entering via a
  near-identical RLE route each time (40x9-10 > 360x3 > 361x6 > 24x3 >
  25x35 > 42x4) — a semi-permeable membrane, deterministic loop
  structure visible in the route repetition. Forensics on every failure:
  ALL crouch-absorbed despite training WITH the crouch data — occupancy
  57.5-78.3%, spells 2525-3459f; g absorbs at frame 104 and never
  shines at all (entry 324x20 > 29x10 > 42x30, identical both runs);
  f absorbs latest (frame 1042 in its best run) — a marginal policy's
  score is absorption-time luck, exactly the 0c-5 pattern. Coverage
  raises the odds the escape solution wins the init lottery (0/3
  without crouch data -> 4/8 with) but does NOT remove the basin from
  the loss landscape. Per-seed run stability HOLDS at n=8 (d spread
  86.3-89.7; g exactly 0/0/0 twice). Protocol lesson: for seed-variant
  properties, 3/3 carries little evidence (p=1/8 under a fair coin) —
  claims about METHOD effects need the failure MODE checked (forensics),
  not just the count. ms_crouch_a remains the champion; farm found no
  new sustain seed (best new chains: 3).

  **SUSTAIN CEILING IS REAL (queue-1 answer, 3x180s on seed a, clean
  0.4-1.1%):** 117.1 / 101.0 self/min, max chains 12 across 188 / 171
  chains per run (r1 lost to the ~20% SD flake). Longer windows did NOT
  reveal clipped chains — the 60s blocks' 19-22 were upper-tail draws,
  not truncations. Chain max is heavy-tailed run-to-run; the ceiling is
  intrinsic to the policy. All breaks remain 100% unforced, resolving
  through air-reflector/jump states (366/27/361/365). NEW at 180s: r3
  spent 647f (5.4%) at the LEDGE where every 60s replay showed zero —
  the ledge valley is reachable, just rare per minute. Strengthens the
  future ledge-drill direction (HANDOFF_2026-07-27c recorded decision)
  without changing the leave---synth-ledge-off verdict.

- [x] **9. Inference-time sampling — MEASURED 2026-07-27: RESURRECTS AN
  ABSORBED POLICY.** The live path already had `--temperature` (CLI
  float, agent config -> Policy sampling); eval_live_protocol.sh now
  takes `--temperature T` (replaces `--deterministic` for the block,
  recorded in protocol.txt). The mechanism-closing result, all vs idle,
  same policy (`ms_synth_ss_b`, the reference absorbed seed):

  | decode | self/min | max chain |
  |---|---|---|
  | deterministic, clean (0.2-0.4%) | 0 · 0 · 0 | 0 |
  | deterministic, repeat noise (27-30% stale) | 0 · 0 · 0 | 0 |
  | **sampled, T=0.5, clean (2.1-2.6%)** | **53.5 · 57.2 · 59.9** | **8 · 9 · 6** |

  A policy that sat 3448 consecutive frames in SquatWait under argmax
  plays at plain-synth level when decode samples — and chains 6-9 show
  T=0.5 costs no visible cycle precision. The absorbing-state model
  called all three cells: action-REPEAT noise is inert in a fixed point
  (repeat-of-crouch = crouch), SAMPLING makes the membrane leaky (at
  ~97% crouch confidence the button head still fires B a few % of
  frames — each one a shine that re-enters the cycle), COVERAGE (item
  8) removes the basin outright and remains the strongest and the only
  deterministic-decode fix.

  **T=0.5 on a crouch-covered policy (ms_crouch_b, the chains-of-2 seed):
  54.5 / 55.7 self/min, chains STILL 2 · 2** (third run lost to the ~20%
  SD flake). Sampling does NOT unstick a functioning policy's chain
  ceiling and costs ~20% rate in cycle slips. Sharpened conclusion:
  temperature is a RESCUE for absorbed/pathological policies (0 -> ~57),
  neutral-to-harmful for working ones (68 -> 55, chains unchanged) —
  seed b's chain-2 limit is structural to what it learned, not a stuck
  state. Untried: T sweep (0.3/1.0); `--deterministic-buttons` reverse
  ablation; adaptive T (sample only when the recent action distribution
  looks pooled — a cheap absorber detector at inference time).

  **RESCUE REPLICATED on a second absorbed policy (2026-07-27 evening,
  pre-registered):** ms_crouch_g — the seed farm's dead seed, 0/0/0
  deterministic, absorbed at frame 104 with 78.3% crouch occupancy —
  scores **78.4 · 72.8 · 61.5 self/min, chains 4 · 3 · 3** at T=0.5
  (clean 0.9-1.4% stale). Prediction made from the theory before the
  block ran. Sampling-rescue is now n=2 policies (ms_synth_ss_b
  53.5-59.9, ms_crouch_g 61.5-78.4), both to roughly plain-synth level,
  both with intact cycle precision (chains 3-9). The membrane leak is a
  property of absorbed policies as a class, not of one seed.

  **T SWEEP on ms_crouch_g (same evening): the rescue SATURATES by
  T=0.3 — the dose-response is FLAT, and the pre-registered
  "lower T -> weaker rescue" prediction was WRONG.**

  | T | self/min | chains |
  |---|---|---|
  | 0 (argmax) | 0 · 0 · 0 | 0 |
  | 0.3 | 80.9 · 77.4 · 70.3 | 4 · 5 · 3 |
  | 0.5 | 78.4 · 72.8 · 61.5 | 4 · 3 · 3 |
  | 1.0 | 63.9 · 77.8 · 76.0 | 3 · 3 · 3 |

  Reading: escape is not rate-limited by leak probability in this range
  — even T=0.3's occasional B-fire suffices, because once out of the
  basin the (deterministic-ish) cycle dynamics are intact and every
  escape immediately pays shines. The 0 -> full-rescue transition is a
  sharp threshold somewhere in (0, 0.3), untested. Chains sit at 3-5
  across ALL temperatures — sampling rescues RATE, never SUSTAIN,
  consistent with the ms_crouch_b result. Practical: if shipping a
  sampled decode as an absorber safety net, T=0.3 buys the rescue at
  the lowest cycle-slip cost measured.

  **MECHANISM CORRECTION (same night, interp session — see
  INIT_FORENSICS_OPTIONS.md findings): g's absorber is a HELD-B fixed
  point, and the flat curve is now quantitative, not mysterious.**
  Offline forward passes over g's real absorbed replay (93.9% parity
  with live) show B pressed on 100% of basin frames at logit ~+0.35 —
  the policy HOLDS B; Melee registers edges, so held B is a no-op. The
  rescue mechanism is random RELEASE: p(release) = 1 - sigmoid(logit/T)
  = 0.24 / 0.33 / 0.41 at T = 0.3 / 0.5 / 1.0 — all fast-escape, hence
  flat. Failure taxonomy across all six failed seeds: hold-B absorber
  (g, h, j), silent never-press absorber (e, f), no-fixed-point
  oscillator (l).

## How to judge any of them

`mix run scripts/eval_policy_on_rollout.exs --policy X --rollout R` on a FIXED
rollout, comparing off-manifold agreement before vs after. The absolute numbers
are confounded; the delta is not. Live confirmation needs
`--seconds N` so the replay finalizes, and `check_replay_ports.exs --expect-cpu 2`
so the opponent is real.
