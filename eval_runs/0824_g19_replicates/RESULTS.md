# RESULTS — champion-recipe replicates (n=3) vs PREREG.md

| run | argmax epoch | fox peak /min (chain) | confirm x3 | stand-mewtwo /min (chain) |
|---|---|---|---|---|
| g19 (original) | 4 | 437.4 (c438) | — | — |
| r1 | 16 | 438.4 (c439) | 438.4 flat | **24.0 (c5)** |
| r2 | 45 | 439.4 (c440) | 439.4 flat | **204.7 (c141)** |
| r3 | 4 | 439.4 (c440) | 439.4 flat | **255.6 (c218)** |

## R1 (spread): fox peak spread = 0.5% — but it's a CEILING, not a distribution

{437.4, 438.4, 439.4, 439.4}: essentially identical. Honest caveat:
chain ≈ 439 over 3,605 frames IS the stand-gate ceiling (one
unbroken multishine for the whole 60s; ~7.3 shines/s = the cycle
rate). The read is therefore "**the champion+awbc recipe reliably
REACHES the gate ceiling**" — single-run fox-peak reads are fine
(the prereg's tight branch), AND the fox stand gate is now
SATURATED as a discriminator: future arms cannot be separated by it.
Discrimination moves to transfer/YS/human rungs. (The 08-19 chaos
was the no-awbc stack's, as suspected.)

## R2 (peak structure): argmax epochs {4, 16, 45, 4} — SCATTERED

Peak HEIGHT is deterministic; peak LOCATION is a lottery.
**Gate-sweep is hereby MANDATORY for every arm** (the prereg's
scatter branch): a fixed-epoch export draws blind from {4..45}.
The champion's ep4 was not special — r1's ep4 presumably gated
lower (its peak was ep16).

## R3 (transfer): mewtwo {24, 205, 256} — 10x RANGE, a pure lottery

The most consequential read. While fox sits pinned at the ceiling,
mewtwo transfer at the fox-argmax epoch varies 10x across identical
runs. The g16-based "AWBC buys transfer (109.8)" claim dissolves
into "transfer is a high-variance DRAW; AWBC runs can draw 24 or
256". Two implications:
1. Any past or future single-run transfer claim is uninterpretable.
2. **Transfer is selectable**: r2/r3 drew 205/256 — if transfer
   matters, add a stand-mewtwo gate to the sweep and argmax jointly
   (we only measured mewtwo AT the fox argmax; a mewtwo-swept
   selection is one cheap protocol change and might buy 2x transfer
   for free from existing snapshots).

## Program notes

- r1/r2's first sweeps failed on the train_delays [0] metadata bug
  (fixed 635e08e) and then on the sync-runner missing the override
  bypass (fixed same night) — both now guarded/unit-covered; the
  final numbers above come from clean re-sweeps.
- NO crown implications (stand numbers never crown). ms_g19_ep4
  remains champion.
- Checkpoints kept: ms_g19r{1,2,3}.bin + per-epoch snapshots
  (r2_ep45/r3_ep4 argmaxes are candidates for any future
  transfer-selected line).

## Verdict for the next arm (RTG conditioning)

Green light on budget: fox-axis reads are cheap (n=1 suffices — but
the fox gate is saturated, so RTG must be judged on CHAIN structure
at harder rungs or dose-response, not stand fox/min). Any transfer
claim needs n>=3 or the joint-sweep protocol above.
