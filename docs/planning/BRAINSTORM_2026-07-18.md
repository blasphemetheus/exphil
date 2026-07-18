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
