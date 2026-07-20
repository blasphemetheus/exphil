# Brainstorm — 2026-07-18 (living document)

Format: interactive narrowing session (Bradley + Claude). Each round of
questions narrows from directions → concrete options; everything worth
keeping lands here, including roads not taken (marked ⏸).

Context snapshot: r14 best (9 conv headless, 6.0/9); DJ frequency solved,
instant-DJ decode escape + shield-lock tail remain; training = 92% of
round time at 54-game pool; GPU free today; fixtures fresh; edifice
adoption partial (audit P-list).

## Round 1: directions chosen (all in)

1. Training rig & velocity  2. Learning paradigm  3. Architectures &
policy  4. Interp & steering  5. Opponents & evaluation  6. Yeti/netplay
product  7. Character expansion  8. Infra subset: #27 NIF isolation +
ONNX/INT8 revival (cloud burst + multi-round automation ⏸ not selected)

## Round 2 decisions

**Training rig & velocity**: BUILD conversion-weighted pool sampling +
epoch-budget-by-gate-curve instrumentation. EVALUATE (not main strategy):
pool windowing, stride-2 sequences — run as measured experiments, keep
the numbers.

**Learning paradigm** (all four greenlit, sequencing TBD in synthesis):
conversion-weighted BC, disagreement mining, PPO fine-tune on r14 (after
the audit's jit fixes), offline-RL-lite (AWR on probe corpus).

**Architectures** (all four): ACT action-chunking head (also a candidate
structural fix for single-frame impulse pathologies), 2x GRU scale
screen, analog shoulder revival (gated on WS5b), frame-delay
conditioning (merges Yeti delay-robustness into the main line).

**Interp & steering** (all four, cheap-first): steering-vector A/B →
SAE port (edifice P4) + feature browse → LEACE α<1 (#7) →
probe-as-regularizer in training.

## Round 3 decisions

**Opponents & eval** (all four): attacking dummy, checkpoint ladder
(#19), self-play opponent pool, scenario expansion + distributional mode.
**Yeti** (all four, in dependency order): user_json wiring + Direct
smoke → rollback integration → delay-robustness A/B → match-flow polish
+ rehearsal.
**Characters**: transfer test FIRST, per-char gate calibration, G&W
fixtures next recording session. (Multi-char single policy ⏸ until the
transfer test says sharing is real.)
**Infra**: #27 the PROPER way (per-checkout exla cache, upstreamable),
ONNX/INT8 revival with an equivalence pin, watcher back online after.
(Build-copy hack ⏸.)

## SOLVED DURING SESSION: the instant-DJ escape path

Debounce trace (9,144 events): the cooldown DECREMENTS PER INFERENCE
CALL, not per game frame. Stateful-step runs ~2.2 inferences/frame →
"10-frame" window ≈ 4.5 game frames, expiring ~liftoff+1 after an SH
release; the policy presses at the first legal moment (all escapes show
cooldown=0->0, some 1->0). Windowed probes ran ~0.5 inf/frame (window ≈
20 frames) — why the band-aid worked r9-r12 and died when probes went
stateful. FIX (immediate): tick cooldown on game-frame advance
(game_state.frame delta), not per call. Unit-testable.

## Idea ledger

- Debounce-trace lesson: default CLI verbosity mutes Logger.info —
  instrument at warning level or stderr, or run --verbose.
- Wall-clock vs frame-domain: ANY frame-semantic decode logic (debounce,
  hysteresis?, action_repeat) must count GAME FRAMES, not calls — audit
  hysteresis and action_repeat for the same bug class.
- The debounce boundary-riding pattern (press at first legal frame) is
  evidence the DJ impulse is an attractor, not noise — supports the
  paradigm work (PPO/conversion-weighted BC) over pure decode patches.

- ONNX-in-Elixir (Bradley: Elixir-first requirement): serve via **Ortex**
  (elixir-nx/ortex 0.1.10, Rustler over ONNX Runtime, Nx in/out,
  CUDA/TensorRT EPs) inside the Agent; export via **axon_onnx** (0.4.0)
  with the priv/python exporter demoted to build-time fallback; INT8
  quantization = offline one-shot. Runtime loop stays Elixir.

## SYNTHESIS — the program (2026-07-18)

Tiered by dependency + GPU economics; every item traces to a Round
decision above.

**TIER 0 — today (GPU free, hours):**
1. Debounce frame-tick fix (cooldown counts GAME FRAMES) + unit test +
   validation probe. [solved root cause above]
2. WS5b mainline-headless drift test (waiting on Bradley's launcher
   netplay-beta click) → fork decision + upstream posting unblocked.
3. Shield-lock steering-vector A/B (contrast extraction from r14, ~hours).
4. Bake-off screens queued behind those: min_gru + mamba + 2x-GRU at
   30-epoch screening budget (NEWERA8_BACKBONE).

**TIER 1 — the r15 cycle (this week):**
- Conversion-weighted pool sampling + windowing/stride-2 EVALUATION runs.
- Epoch-budget instrumentation (post-hoc gate-curve probe sweep of
  r14's _latest history; watcher revival lands later with #27).
- r15 recipe: frame-tick debounce + conversion weighting + (if steering
  A/B positive) probe-as-regularizer OR LEACE α<1 pick.
- Checkpoint ladder v1 (round-robin ELO, headless-parallel).
- Attacking dummy (also unblocks getup scenarios + realism).

**TIER 2 — next two weeks:**
- Paradigm ladder in order: conversion-weighted BC → disagreement
  mining → PPO fine-tune (needs audit jit fixes) → AWR-lite.
- SAE port (edifice P4) + feature browse; probe-as-regularizer if
  steering says the direction is causal.
- ACT chunking head screen; frame-delay conditioning (feeds Yeti).
- #27 proper (per-checkout exla cache, upstream PR) → watcher online.
- ONNX/Ortex serving path + equivalence pin.
- Transfer test (r14 → G&W corpus) + per-char gate config; G&W fixtures
  when Bradley records; scenario expansion + distributional mode.

**YETI TRACK (interleaved, August deadline):**
user_json wiring + Direct smoke (next quiet evening, needs Bradley
present) → rollback integration → delay A/B (informs conditioning) →
match-flow polish + full rehearsal night.

## Steering A/B results (2026-07-19, Tier-0 #3 executed)

Vector: `checkpoints/r14_shield_steer.bin` — contrast of r14 trunk
features (GRU, window 16, hidden 256) over all six r14 probe replays:
mean(trunk | P1 action 178-182) - mean(trunk | P1 grounded non-shield),
4,428 shield vs 92,865 grounded frames, raw contrast norm 10.25,
normalized to |v|=1 (`scripts/extract_steer_vector.exs`). Hook:
`--steer-vector/--steer-alpha` projects alpha of the feature component
along v̂ out before the heads (`ExPhil.Interp.Steering`), live on both
the windowed (Axon.nx `steer_project`) and stateful step seams.

Probes: one headless ExiAI game per arm, r14 policy, --stateful-step
--jump-debounce 10 --deterministic, tech_random dummy, FD (same recipe
as newera8 fan-outs; ports 51541-43; probes/steer_ab/).

| arm      | frames | card | shield occ % | run p50/p95/max (f) | runs>=200 | breaks | OOS idle % | kd | conv (rate) |
|----------|--------|------|--------------|---------------------|-----------|--------|------------|----|-------------|
| baseline | 21,683 | 7/9  | 2.79         | 5 / 215 / 215       | 2         | 2      | 8.0        | 31 | 7 (22.6%)   |
| α=0.5    | 28,801 | 6/9  | 5.86         | 13 / 215 / 215      | 6         | 7      | 20.45      | 26 | 8 (30.8%)   |
| α=1.0    | 16,838 | 9/9  | 1.52         | 3 / 32 / 59         | 0         | 0      | 2.27       | 20 | 5 (25.0%)   |

**VERDICT: CAUSAL.** Full projection removal (α=1.0) takes the policy
off the 215 pin entirely (p95 215→32, max 59, zero runs ≥180, breaks
→0) and scores the first 9/9 card of the program — with shield
occupancy DOWN (2.79→1.52), OOS idle DOWN (8.0→2.27) and conversion
rate intact (25%). Half-strength (α=0.5) is NOT a milder win: it reads
worse than baseline (occupancy 5.86%, p50 run 13f, 7 breaks) — partial
attenuation seems to leave the attractor reachable while weakening the
release signal. The 178-182 trunk direction is causally responsible for
the shield-lock; the fix is all-or-nothing along this axis.

Side observation (α=1.0): double jumps vanished entirely (DJ per air
stint 0.0%, liftoff→DJ p50 nil; jumps/100 still 0.96, X/Y p50 3f — SH
capable). The shield direction and the DJ impulse apparently share
subspace — consistent with the prev-action copy-signal story. Watch DJ
recovery usage before shipping α=1.0 as a default (recovery DJs matter
off-stage even if neutral DJs were pathological).

**DJ-recovery check (2026-07-19 evening, scripts/dj_recovery_check.exs):
INCONCLUSIVE — zero offstage samples at α=1.0.** Offstage recovery
situations (airborne, |x|>88 or y<-6, ≥15f) across the available games:
- baseline (2 games): 6 situations, DJ used 2/6, 3 returned / 3 died
  (both DJ'd recoveries returned) — r14 does use its recovery DJ.
- α=0.5 (1 game): 1 situation, no DJ, returned.
- α=1.0 (2 games, incl. a dedicated 28,801f probe): 2 + 0 situations,
  no DJ used, both returned. The dedicated full-length α=1.0 game never
  went offstage at all.
Recovery-DJ capability at α=1.0 is therefore UNTESTED by live sampling
(the steered policy barely leaves the stage — itself weak positive
evidence on positioning, not proof of DJ capability). GATE for
shipping --steer-alpha 1.0 as a live default: run an offstage-forcing
check first (multiple games, or a scenario prefix that starts P1
launched offstage) — on the ledger as follow-up work, not tonight.

Per the Tier-0 decision rule (causal → r15 gets probe-as-regularizer or
LEACE α<1): pick up probe-as-regularizer for r15, and keep
`--steer-vector r14_shield_steer.bin --steer-alpha 1.0` as a live
mitigation usable TODAY (it stacks with --jump-debounce; zero training
cost). α<1 variants are dead per the α=0.5 arm.

## Bake-off screen results (2026-07-19, Tier-0 #4 executed)

Three 30-epoch screens on the 54-game pool (newera8, NEWERA8_TAG
isolation, 1 round each, dropout 0.6, 2x3-arm headless probe fan-outs,
scored x/9). Loss = the drill's best/exported loss at stop ("Converged"
line); epoch-30 sampled loss in parens (per-epoch loss is noisy at
dropout 0.6). Reference rows: the r-line GRU-256 at its OWN epoch 30
(r13, dropout 0.4) and full 100-epoch r13/r14 results.

| screen        | backbone     | hidden | loss @30ep (best / e30 sample) | card avg | per-game cards   | conv | train (s) | probe (s) | stateful step path |
|---------------|--------------|--------|-------------------------------|----------|------------------|------|-----------|-----------|--------------------|
| screen_min_gru| min_gru      | 256    | 0.447 / 0.937                 | 6.6/9    | 7,6,6,7,7,7      | 1    | 2,656     | 1,048     | UNSUPPORTED (agent gate gru/lstm only — fell back windowed despite PROBE_STATEFUL=1) |
| screen_mamba  | mamba        | 256    | 0.072 / 0.535                 | 8.1/9    | 8,8,8,8,9,8      | 5    | 8,717     | 1,037     | n/a (PROBE_STATEFUL=0, windowed) |
| screen_gru2x  | gru          | 512    | 0.285 / NaN(restored)         | 6.0/9    | 6,6,6,6,7,5      | 8    | 9,263     | 1,058     | ACTIVE (512-hidden step path worked) |
| r13 reference | gru          | 256    | (0.307 e30 sample; dropout .4)| 4.5/9    | (100ep result)   | 7    | ~17,400   | —         | ACTIVE |
| r14 reference | gru          | 256    | conv. 0.048 @100ep            | 6.0/9    | (100ep result)   | 9    | ~17,400   | —         | ACTIVE |

Notes:
- **mamba is the card winner by a wide margin** (8.1/9 incl. one 9/9;
  first mamba on edifice's real conv) and the CHEAPEST loss at 30ep
  (0.072) — but converts less (5 vs 8/9). Its shield gates pass across
  the board. Cost: 3.3x min_gru train time, and no O(1) step path
  (windowed probes only; a Stateful impl or the GatedSSM-style cache
  would be needed for stateful probing/netplay rollback).
- **gru2x converts best of the screens (8)** but cards worst (6.0/9)
  and hit a NaN detonation at epoch 30 (restore 1/5 caught it — best
  params 0.285 exported; GOTCHA #46 class at 512 hidden, bf16). A
  gru2x rerun wants f32 or a lower LR before it's judged fairly.
- **min_gru is the speed king** (2,656s train, 3.3x faster than mamba)
  but behaviorally weakest here: 1 conversion, mid cards, loss floor
  0.447 at 30ep. Also NOT stateful-capable in the agent today.
- Screens vs r-line: all three screens at 30 epochs card >= the r13
  100-epoch result (4.5/9); mamba at 30ep beats even r14's 6.0/9.
  30-epoch screening is a real signal at ~1/6 the train cost —
  supports the epoch-budget instrumentation item in Tier 1.
- Screen probe throughput note: min_gru/mamba probes ran windowed at
  59.9/60 fps with ~0.7-0.75 inferences/frame (window 16 is cheap
  enough at 3 arms); gru2x ran the step path. Step latency was not
  separately instrumented this run.
