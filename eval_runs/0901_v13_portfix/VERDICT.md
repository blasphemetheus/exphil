# v1.3 portfix — CHAIN VERDICT (2026-09-01 22:00)

## PRE-REGISTERED PRIMARY: PASSED, decisively

approach_delta (position sweep on the SAME livelook states as v1.2's
probe, :neutral filter, both heads):

| d | v1.2-ARrefit | v1.3-ARrefit | v1.3-INDrefit |
|---:|---:|---:|---:|
| 15 | -0.072 | -0.011 | -0.012 |
| 25 | -0.127 | -0.016 | -0.020 |
| 40 | -0.199 | -0.008 | -0.012 |
| 60 | -0.247 | **+0.037** | **+0.043** |
| 90 | -0.262 | **+0.114** | **+0.135** |
| 130 | -0.243 | **+0.190** | **+0.220** |

The retreat curve did not shrink — it INVERTED into the expert's
signature (F3c: expert toward rises with distance). Close range ~neutral,
increasing approach with range. AR≈IND → trunk property, now the FIXED
trunk's property. P(z) baseline also eased (0.24-0.27 vs 0.28-0.33).

**Conclusion: the E1b corpus corruption (44% scrambled-slot files) WAS
the root cause of the positional-blindness/retreat cluster.**

## Guards

- **Recovery: AMBIGUOUS at n.** died% 44.4 (AR) / 43.8 (IND) vs v1.2's
  28.6 — ratio 1.55x, UNDER the pre-registered 2.5x deaths floor at
  ~36 episodes → formally unresolved; airdodge-first-route back up
  (30.6/46.9 vs v1.2's 18.4). Both still crush ep10 (60.7). Needs either
  a bigger-n score run or the live look to resolve.
- **L_cond: 1.49** on ARrefit (joint arm 1.09, IND —). Below the old
  ~2.5 reference, BUT this is the first run of the probe on CLEAN
  captures — the old 2.39/2.67 were measured with the E1c-diluted
  instrument, and its whole scale reads differently (R_state 1.31-1.40
  clean vs ~1.05-1.14 dirty). Within-table: the wire is restored
  directionally (1.49 vs 1.09), magnitude TBD against a clean 8a re-probe.
- Edgeguard first-option dash 7.9% (AR) vs expert 41.5 — approach is in
  the STICK signature but converting it into dash-dance/movement remains
  open (closed-loop drift family). Grab-first still 18.4% vs expert 0.7.

## Also in this chain

- Action-sensitivity rerun with FIXED embed-space actions: hard_right
  diverges 68.7% of drift (mislabeled first run said 33%) — dynamics
  model solidly action-conditional.
- v1.3 arms: AR val 5.3242 / IND 5.7643 (conditioning gap reproduces on
  the clean corpus).

## Next

1. **Bradley's live look at fox_gen_v1.3_ARrefit** (decides the survival
   ambiguity + feel; gates --head flip as ever).
2. Clean re-baselines: 8a/v1.2 coincidence re-probe on the fixed
   instrument, critic extract + dynamics retrain on clean embeds
   (INFRA_HARDENING §7).
3. If live look confirms: the complaint cluster re-derives — next lever
   likely dash-dance/movement (closed-loop) and finisher selection, with
   AWBC now port-correct and available.
