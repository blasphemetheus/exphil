# Peak science P1 + P2 — results (2026-08-21/22 overnight)

Preregs in run_p1.sh / run_p2.sh. Baselines: g19 (ramp-10 SS, ~12/60
epochs >=300), g20 (same recipe, ~24/60). Ceiling ~435-439 in every
arm ever swept.

## Card (peak-dwell = epochs gating >=300 of total)

| arm | mechanism tested | dwell | ceiling | notes |
|---|---|---|---|---|
| P1 anneal | lr 2e-5 from ms_g19_ep4 | 8/15 = 53% | 437.4 (ep2) | still hops at tiny lr; tail drifts basin-ward |
| P2 ss0 | NO scheduled sampling | 8/60 = 13% | 439.4 (ep18 — highest single gate ever) | still hops; late onset (ep14); worst dwell |
| P2 ssfull | SS 0.5 from epoch 1 | 18/60 = 30% | 434.4 (ep7) | ~baseline dwell; BIMODALITY SOFTENS — many 240-300/min, c140-300 epochs (every other arm is strictly ~90 or ~435) |
| (F3 A3, for comparison) | KL anchor to peak teacher + 2x dose | **46/60 = 77%** | 438.4 | + absorbed 2x human data |
| (F3 A2, completed 08-22) | KL anchor, 1x dose | 22/60 = 37% | 438.4 (ep13) | TEMPERS the anchor claim: anchor alone ~ baseline dwell; the 77% belongs to anchor+2x-dose (or variance, n=1) — A4 disambiguates |

## Verdicts vs prereg

- **P1: DRIFT/HOP — step size is not the lever.** Annealing raises
  dwell (53%) but cannot pin the peak; the attractor structure
  survives a 10x lr cut.
- **P2: NULL on "SS drives the hopping"** (ss0 still hops) — the
  two-attractor dynamic is intrinsic. SS dose-response on dwell:
  13% (off) -> 20-40% (ramped) -> 30% (full): SS helps REACH the
  peak, saturating around ramped levels. New observation: full SS
  softens the bimodality (intermediate chain-competence states exist
  under heavy self-conditioning — consistent with SS training partial
  recovery behaviors that other arms never learn).
- **Program conclusion: the F3 anchor is the only strong stabilizer
  found (77%), and it composes with data absorption.** Mechanism of
  the hopping itself remains OPEN (remaining suspects: AWBC weighting
  noise, per-epoch shuffle order, intrinsic GRU closed-loop
  sensitivity) — but with anchor+sweep in hand, the mechanism hunt is
  a curiosity, not a blocker.

## The standing recipe after this program

1. Train a peak-finder run (any of the family; gate-sweep it).
2. Anchor follow-on runs to the best peak checkpoint (distill w=0.5)
   — they stay on the peak 77% of epochs AND can absorb new data.
3. Select by sweep argmax with the multi-gate profile (fox + mewtwo +
   d4-id3). NO CROWN without the human rung (g6 rule).

Pending: A4 no-anchor dose control (attribution), A2 sweep completion
— both queued behind this program (run_postqueue.sh).
