# g15r2 (champion recipe, NO awbc, fixed stack — replicate) — results

Run 2026-08-19 21:16–22:35. Prereg in `run_g15r2.sh` header.

## Headline: the no-awbc arm COLLAPSED on replicate

| gate | g15r (0814) | g15r2 exported (ep51) | g15r2 ep50 rescue |
|---|---|---|---|
| stand-fox d3 x3 | 116.8/min c23 | **12.0/min c5** (x3 identical) | **362.5/min c353** (x3 identical) |
| stand-mewtwo d3 | 27.0/min c2 | **0.0/min c0** | **0.0/min c0** |

(ep50 caveat: `_latest.bin` is a periodic mid-training save, not the
drill's normal export path — if export applies any selection/EMA the
comparison to other runs' exports carries that asterisk.)

Training anomaly: loss tracked the 0814 run's band (~0.007–0.024
through epoch 50), then dropped SIX ORDERS OF MAGNITUDE in one epoch
(0.0105 → 2.55e-8 at epoch 51), tripping the convergence exit; the
exported checkpoint is the post-drop epoch-51 weights and is
behaviorally dead (12.0 fox / 0.0 mewtwo). A loss of ~1e-8 on 380k
mixed human+rollout frames is not learnable signal — it is a
degenerate solution (or a loss-computation degeneracy) reached in a
single epoch.

`ms_g15r2_latest.bin` is the epoch-50 periodic save (pre-collapse,
loss 0.0105) — gated separately below as the honest replicate.

## Prereg reads

- **RC (determinism): ANSWERED — training is NOT deterministic.**
  Per-epoch losses diverge from the 0814 run. No --seed knob is
  REQUIRED for replicates (run-to-run noise exists); a seed flag is
  now a reproducibility nice-to-have, not a blocker. The planning-queue
  "--seed if deterministic" item is CLOSED (not needed).
- **RA/RB (variance sizing): superseded by the collapse.** The
  exported arm is not a variance sample; the ep-50 rescue gate is the
  usable replicate datapoint (see table).

## What the run means (new findings, bigger than the prereg)

1. **The no-awbc fixed-stack recipe has ENORMOUS run-to-run variance**:
   fox 116.8 (0814) vs 362.5 (this run, ep50) among alive states, PLUS
   a one-epoch collapse mode (ep51). Single-run reads on this stack
   are close to uninterpretable — every future arm comparison at this
   scale needs either replicates or a variance-aware bar.
2. **The AWBC attribution VERDICT R1 IS NOW PARTLY OVERTURNED.**
   The "+117% fox" claim is dead: this no-awbc run BEAT g16 (awbc) on
   fox by +43% (362.5 vs 253.6) and nearly closed the gap to old-grad
   g15 (430.4) — so the fixed stack CAN reach near-champion fox levels
   without AWBC, and g16's 253.6 may be a middling draw from a wide
   distribution. What SURVIVES for AWBC: (a) generalization — mewtwo
   109.8 (awbc) vs 27.0 / 0.0 / 0.0 (all three no-awbc states); this
   ep50 fox monster is a pure specialist with ZERO mewtwo transfer;
   (b) ~~training stability~~ **RETRACTED same night**: g18 (awbc,
   90ep arm) collapsed the same way at epoch 10 — the collapse mode is
   recipe-INDEPENDENT (2-of-3 runs on 08-19 evening, awbc on and off,
   all lr 2e-4; never seen before this night). See GOTCHA #99; a
   collapse guard now routes these epochs through the NaN-restore
   path. The 0813 arms result (B2 > B1 with placebo collapse) still
   stands — different pool, offline+deploy validated.
3. **The retune program's premise needs re-examination**: g16's G1
   "drop" vs g15 may be substantially VARIANCE, not (only) a
   grad-fix optimum shift — the fixed stack just produced 362.5/c353
   at lr 2e-4/60ep with no recipe change. Before spending more
   single-run knob arms (epochs, bptt), consider n>=3 replicates of
   the champion+awbc recipe to establish the actual distribution.
3. **Convergence-exit hardening (small, do while GPU idle):** the
   drill's converged check trusted a single-epoch loss. A guard —
   e.g. require loss < target for 2 consecutive epochs, or flag a
   >100x one-epoch drop as divergence-suspect and export best-epoch
   instead — would have exported epoch 50 and saved the arm. Candidate
   FIXES.md entry; also worth a GOTCHA.
4. Crime-scene capture per the dev practice: checkpoints (ep50 +
   ep51), full train.log, and the pool composition are all retained
   in this dir + checkpoints/; a fatal-batch style capture does not
   exist for epoch granularity — if collapse recurs, add per-epoch
   optimizer-state snapshots near the loss floor.

## Next

- GPU #3 (g18, 90-epoch @ 2e-4) launches next per the g17 negative
  read (`eval_runs/0819_g18_ep90/run_g18.sh`). NOTE for g18 scoring:
  (a) its convergence exit has the same single-epoch trust — check the
  loss trajectory before believing any "Converged" line; (b) finding
  #1 above caps what one g18 run can prove — a pass/fail vs the 304
  bar is a DRAW from a distribution, not a verdict on epochs. The
  awbc arms look lower-variance so far (never collapsed; g16/g17
  spread unknown), but Bradley should weigh replicates-first
  (finding #3) against more single-run knob arms when he's back.
- NO CROWN implications (stand numbers never crown).
