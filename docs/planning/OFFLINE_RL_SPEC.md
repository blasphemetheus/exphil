# F5 spec — advantage-weighted BC: the offline-RL entry rung

**Written 2026-08-04 (ML_FIELDS_ROADMAP F5). Implementation-ready spec;
gated on the human corpus growing and the cycle-4 arc concluding — this
is the bridge to RL staging (#29), buildable without an emulator.**

## Why this rung first

Plain BC's ceiling is the demonstrator: every frame is imitated equally,
including the mistakes. Online RL's floor is emulator throughput: days
per experiment on one machine (see the 08-04 RL assessment). Between
them sits offline RL: reweight the SAME imitation pipeline by observed
outcomes, at BC compute cost, no Dolphin in the loop. The replays
already contain the reward signal BC throws away (stocks, percent), and
`ExPhil.Rewards.Standard.compute(prev_state, curr_state)` already
implements it.

Adjacent precedent in-repo: conversion-weighting (#35) validated
outcome-weighted data at the replay level; this is the same idea at the
frame level with a principled form (AWR/AWAC-family).

## Design — AWBC v1

### Advantage signal (per decision frame)

Discounted return-to-go over a bounded horizon, from existing rewards:

```
R_t = sum_{k=0..H} gamma^k * r_{t+k}         H=300 (5s), gamma=0.997
r_t = Rewards.Standard.compute(state_t, state_{t+1})
      (stock +/-1 terms + damage-differential terms, existing weights)
A_t = R_t - V_bar(t)
```

`V_bar` v1 is the REPLAY-MEAN return (a constant baseline per replay) —
no learned critic yet. This is deliberately crude: the first question is
whether outcome weighting moves behavior at all, and a learned V is the
v2 upgrade only if v1 shows signal (IQL-style expectile V comes with
it; that critic is also what online RL will want).

### Weighting

```
w_t = clip(exp(A_t / beta), w_min, w_max)    beta tunes sharpness
loss = mean( w_t * per_frame_imitation_loss_t )
```

- beta from a percentile rule (set so p90(w)/p10(w) ≈ 5-10, not hand
  units); clip at [0.2, 5.0] to keep gradients sane.
- Weights precomputed at pool-build time (like embeddings/teacher
  logits) — per-frame scalars threaded as a batch field into
  `Loss.build_loss_fn` (same insertion point as the F3 distill mask;
  build both plumbing paths in one pass if implementing together).
- Composes with, does not replace, snippet dosing and (if adopted) the
  F3 anchor: weighting shapes WHICH demonstrated behavior is amplified;
  the anchor protects the core skill while it happens.

### What the signal means per data source

- Drill/rollout multishine data: advantage ≈ damage dealt per cycle —
  weighting should favor unbroken chains (mild effect expected; the
  data is already curated).
- HUMAN games (the real target): advantage marks the exchanges the bot
  WON — the frames where its behavior beat a human's response get
  amplified over the frames where it got hit. This is the first
  training signal in the whole program that can exceed the
  demonstrator distribution rather than match it.

### Frame-boundary hygiene

Return-to-go must not cross replay boundaries (frame-number
discontinuity check — same convention as the prev-controller queue) or
snippet-list boundaries (each snippet is its own list; returns computed
within-list only, or inherited from the SOURCE replay at mine time —
prefer the latter: the miner sees the full game and can stamp each
snippet frame with its true return before extraction).

## Pre-registered evaluation (run when triggered)

Arms at equal data and compute, gated per EVAL_PROTOCOL.md:
  B1 plain BC (current recipe)
  B2 AWBC, beta at the percentile rule
  B3 AWBC with weights SHUFFLED within-replay (the control that kills
     placebo: same weight distribution, no outcome information)
Gates: stand d3 deterministic single run (must hold within 10% of
baseline); YS collapse bucket + AbsorberEntry count; promote rung 0
(opponent-sensitivity must NOT rise — outcome weighting could plausibly
amplify dummy-exploit frames, this is the guardrail); B2 > B3 on any
claimed gain or the gain is not the advantage signal.

Success = B2 beats B1 on the deploy-rung/YS/human-facing metrics with
B3 flat. Failure with B3 flat is also clean: outcome weighting doesn't
transfer at this data scale — record and stop.

## Triggers

1. Cycle-4 arc concludes (don't confound two training-recipe changes).
2. Human corpus reaches ~30+ games (advantage estimates on 12 replays
   are noise; the miner stamps returns as games accumulate).
3. Implementation slot: with F3's batch-field plumbing (same loss-fn
   insertion point), one pass covers both specs.

## Explicit non-goals (v1)

No learned critic, no bootstrapped targets, no off-policy corrections,
no return conditioning (the DT backbone exists if that route ever
opens). Each is a v2+ decision gated on v1's B2-vs-B3 result.
