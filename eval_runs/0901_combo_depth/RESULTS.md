# F4 combo-depth probe — RESULTS

Policy fox_gen_v1.3_ARrefit_policy.bin, 24 expert files, K=16
coherent samples at T=0.5, Leg S match rule, teacher-forced
expert states (no state-visitation confound). Depth = opponent-hitstun
rising edges since the conversion opened.

| stratum | rows | sampling pass@1 % | pass@16 % |
|---|---:|---:|---:|
| neutral | 2500 | 7.8 | 45.1 |
| punish hit 1 | 2500 | 4.5 | 30.3 |
| hit 2 | 2500 | 4.2 | 28.7 |
| hit 3 | 2500 | 3.9 | 26.8 |
| hit 4+ | 2500 | 4.0 | 27.4 |

Reading (pre-declared):
- flat across depth (and ~neutral level) -> continuations ARE learned;
  live 1-2-hit punishes are STATE-VISITATION (it never creates/holds the
  follow-up state) -> levers: hit-confirm savestate drills / DAgger,
  closed-loop work.
- collapses with depth -> BC never learned the deep layer (rare frames
  underweighted) -> levers: conversion-window curation / AWBC standard
  RTG (damage return-to-go upweights exactly these frames), deep-punish
  oversampling.

## VERDICT

**Depth-collapse REJECTED — the fork lands on state-visitation, with a
uniform punish-class deficit on the side.**

1. Within punish sequences, match is FLAT across depth (pass@16: 30.3 →
   28.7 → 26.8 → 27.4). The model's knowledge of hit-4 continuations
   equals its knowledge of hit-1 continuations. Since LIVE punishes cap
   at 1–2 hits while teacher-forced depth is flat, the live shallowness
   is STATE-VISITATION: it can act correctly from mid-combo states — it
   just never creates/holds them (the closed-loop drift family, same as
   dash-dance).
2. Secondary: ALL punish strata sit ~35–40% below neutral (pass@1 4.0–4.5
   vs 7.8) — a uniform conversion-class deficit, though this cross-
   stratum comparison carries a hardness confound (punish decisions are
   faster/more precise; match rules bite harder there).

**Lever ordering that follows:**
- PRIMARY (depth): closed-loop — hit-confirm savestate drills / DAgger
  from real .slp combo moments (improoover thread = the delivery
  mechanism); teach it to ARRIVE at hit 3, the actions are already there.
- SECONDARY (class deficit): AWBC standard-RTG / conversion-window
  oversampling — now port-correct — to lift the whole punish class.
