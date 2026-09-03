# v1.4-long morning readout (09-03) — the compute-only arm

One knob vs v1.3 AR: +4 epochs (2.3x total compute), same recipe/corpus.
Policy: fox_gen_v1.4_long_20260903_072152_best_policy.bin.

## 1. Underfit CONFIRMED
val_loss 5.324 -> 5.296 -> 5.274 -> 5.256 -> 5.241: monotone, ~-0.02/epoch,
no plateau at 2.3x compute. More training keeps buying loss at this data
scale — the slippi-ai parity audit's "days not hours" read holds.

## 2. T=1.0 health bar: SURVIVES (was a death sentence)
| arm | mean len | to cap | frozen | d_up/min | loops/min |
|---|---:|---:|---:|---:|---:|
| T=1.0 | 115.5s | 2/4 | 0.00 | 410 | 0.16 |
| T=0.5 | 105.8s | 3/4 | 0.01 | ~101 | 0.49 |
Full-temperature play is now as survivable as T=0.5 — but the buttons
head sprays at T=1.0 (d_up 6.8 presses/SEC, mostly no-ops in game):
the distribution survives full temperature without yet being SHARP at it.

## 3. Drill metric: compute alone does NOT move it (the 2x2 closes)
| arm | drill 0-19: hits / >=3% / dmg |
|---|---|
| v1.3 baseline (538 eps) | 2.9 / 30 / 16.4 |
| v1.4-long (compute only, n=39) | **1.9 / 18 / 12.5** |
| v1.3-AWBCdrill (drill mix) | 3.7 / 41 / 17.5 |
| expert | 3.9 / 87 / 27.0 |
More BC compute alone left the hit-confirm skill flat-to-worse; the
drill mix moved it. TWO INDEPENDENT LEVERS confirmed: scale buys general
sharpness (loss, T=1.0 survival), drills buy specific skills — they
compose, which is exactly the v2 recipe + drill-curriculum design.
(Caveat: n=39 in the 0-19 cell for v1.4; direction clear, magnitude noisy.)

Owed: Bradley live looks (AWBCdrill transfer check; optionally v1.4 at
T=0.75-1.0 to eyeball the noisy-buttons tradeoff).
