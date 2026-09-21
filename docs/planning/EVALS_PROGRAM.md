# Evals program — from gates to explanations

**Written 2026-09-21 (Bradley: "make evals better").** Living doc: the plan
in priority order, then a ledger. Everything here runs on the sim loop
(`SIM_INTEGRATION.md`) unless marked Dolphin. The thesis: the gates we have
(held-out CE, parity, fingerprint tells, drill conversion, live case)
answer *better or worse*; the evals below answer *where, against what, and
why*. In a deterministic sim with savestates an eval can be a controlled
experiment (same state, two policies; same policy, one input changed), so
most of these are causal, not statistical.

## Priority list (Bradley's order, 09-21)

1. **Regret maps** — per start, the oracle's best outcome minus the
   policy's own; binned by situation (`ExPhil.Situations` labels +
   geometry: distance, relative height, percent band, defender state).
   With policy-guided oracles the data already exists: `n_converted / 64`
   is the policy's own conversion probability from that start and
   `converted_any` is the oracle's. Output: a table per bin (n, policy
   rate, oracle rate, regret) and a **diff of two maps** on the same pool
   (what a training step changed, by situation). `scripts/regret_map.exs`.
2. **Bootstrap intervals on every existing gate** — drill rates (300
   starts ≈ ±0.05), fingerprint tells (10 games), oracle rates. Report the
   interval next to every number; iterations 2–3 were argued at the noise
   floor.
3. **Opponent axes / opponent panel** — every drill number is vs idle or
   vs self. Fixed panel: idle, epoch-3, mix2, a scripted dash-dancer, a
   scripted shield-grabber, later CPU levels. One number becomes a profile
   and "converts vs idle only" (the full-hop story) is caught by
   construction.
4. **Full-game outcomes in the sim: the league** — A vs B, ≥ 200 games,
   stocks + damage → win rate, Elo, matchup table. Never run yet. The
   26-character league on the checklist is the large version.
5. **Per-situation option divergence** — the policy's option distribution
   vs the corpus's in each situation (`ExPhil.Options`, situation_stats
   v2 exist). Human-likeness WITH a location: "full hops in neutral"
   would have been named directly.
6. **Execution evals with a range (D4)** — L-cancel offset, wavedash
   angle, short-hop rate, dash-dance cadence vs the per-player human
   spread; the sim supplies thousands of attempts per skill from seeded
   starts instead of whatever a 30-s game contains.
7. **Robustness sweeps** — reaction delay 0..4, input noise, seed variance,
   per policy; a robustness curve per gate.
8. **Checkpoint diffs + interp** — see the interp section; the direction
   Bradley wants most.
9. **Transfer evals** — the drill run in Dolphin from savestates
   (Improoover recipe) so sim conversion is checked against real
   conversion; R1 did this for behaviour, not outcomes.

## Interp × sim — the possibility space

What the sim adds to interpretability is *intervention with ground
truth*: any hypothesis about what the policy computes can be tested by
changing the state, the activations, or the policy and measuring the
outcome in a deterministic world. Existing tools: SAE features, steering
vectors (`ExPhil.Interp.Steering`), probes (early-reject probes
default-on), `Inspect.counterfactual`, the CycleSim offline simulator,
the identity channel.

- **Regret-conditioned probes.** Train probes on trunk features to
  predict "the oracle converts from here" and "the policy converts from
  here"; the difference is a readout of *known-but-unselected* behaviour.
  Where that probe fires but the policy does not act is the selection
  failure, localized in feature space.
- **Causal steering in the sim.** Steer along a feature (e.g. the
  "commit" direction found by SAE on converting vs non-converting
  samples) and measure conversion and fingerprint deltas on the drill
  pool. That is the first eval where an interp claim has a behavioural
  ground truth with a confidence interval.
- **Input attribution by intervention.** Perturb one input group at a
  time in the sim state (opponent action id, opponent percent, distance,
  own action frame) and measure the change in the policy's decision and
  in outcome. "What does the policy read" answered causally, per
  situation, not by gradient.
- **Checkpoint diffs.** mix2 − epoch-3 on the same 1,000 starts:
  activation deltas, per-situation option deltas, regret-map deltas,
  fingerprint deltas. A training step becomes a *mechanism* ("the
  short-hop feature lost weight in neutral") instead of a score.
- **Opponent-model readout.** Probe for opponent actionability /
  hitstun-remaining / DI direction in the trunk; the sim knows the true
  values every frame. Tests whether the policy tracks the defender state
  the scorer uses.
- **Identity geometry.** The 112-slot name embedding: which directions
  change which tells (jump button, c-stick use)? Steer between slots and
  measure the fingerprint. The "prior or costume" question from the
  checklist, answered by intervention.
- **Engine lens (coach C2).** Per state: the policy's option
  distribution, the oracle's best, the critic's value once it exists.
  Rendered in the rewind viewer.
- **Circuit-level, later.** Which GRU units carry "opponent actionable"
  and "I am in lag"; ablate them in the sim and watch the drill.

## Ledger

| Date | Item | State | Evidence |
| --- | --- | --- | --- |
| 2026-09-21 | doc | written | priority list from Bradley; regret maps first |
| 2026-09-21 | 1 regret maps | in progress | `scripts/regret_map.exs` (oracle results + pool states → situation bins) |
