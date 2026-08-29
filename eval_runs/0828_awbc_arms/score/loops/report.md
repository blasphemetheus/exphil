# Loop / Taunt Report

Bot port 1, 24 replays, 3 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| B1 | 8/8 | 2.04 [1.47-3.43] | 114.63 [101.79-131.18] | 0.00 [0.00-0.00] | 0.25 [0.15-0.30] | 1.49 [0.49-3.43] | 8.38 [3-14] |
| B2 | 8/8 | 1.87 [0.49-2.94] | 126.85 [122.38-132.54] | 0.00 [0.00-0.01] | 0.27 [0.20-0.33] | 1.80 [1.47-2.45] | 8.38 [7-11] |
| B3 | 8/8 | 1.90 [0.98-3.92] | 119.00 [112.04-126.25] | 0.00 [0.00-0.01] | 0.25 [0.20-0.39] | 1.35 [0.98-2.45] | 5.38 [3-9] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| B1 | r1.slp | 2.04 | 1.96 | 108.26 | 0.00 | 0.30 | 1.96 | 11 |
| B1 | r2.slp | 1.85 | 1.62 | 128.03 | 0.00 | 0.29 | 2.16 | 11 |
| B1 | r3.slp | 2.04 | 1.47 | 110.17 | 0.00 | 0.15 | 1.47 | 9 |
| B1 | r4.slp | 2.04 | 1.96 | 131.18 | 0.00 | 0.25 | 0.49 | 3 |
| B1 | r5.slp | 2.04 | 1.47 | 101.79 | 0.00 | 0.30 | 3.43 | 14 |
| B1 | r6.slp | 2.04 | 3.43 | 112.10 | 0.00 | 0.26 | 0.49 | 6 |
| B1 | r7.slp | 2.05 | 2.93 | 110.48 | 0.00 | 0.20 | 0.49 | 3 |
| B1 | r8.slp | 2.04 | 1.47 | 115.01 | 0.00 | 0.29 | 1.47 | 10 |
| B2 | r1.slp | 2.04 | 0.49 | 129.99 | 0.00 | 0.31 | 1.47 | 7 |
| B2 | r2.slp | 2.04 | 0.98 | 129.25 | 0.00 | 0.24 | 1.96 | 7 |
| B2 | r3.slp | 2.04 | 2.45 | 122.91 | 0.00 | 0.33 | 2.45 | 8 |
| B2 | r4.slp | 2.04 | 1.96 | 126.30 | 0.00 | 0.29 | 1.96 | 7 |
| B2 | r5.slp | 1.78 | 2.24 | 126.23 | 0.00 | 0.21 | 1.68 | 10 |
| B2 | r6.slp | 2.04 | 2.94 | 125.23 | 0.00 | 0.32 | 1.47 | 7 |
| B2 | r7.slp | 2.04 | 2.93 | 132.54 | 0.00 | 0.26 | 1.47 | 10 |
| B2 | r8.slp | 2.04 | 0.98 | 122.38 | 0.01 | 0.20 | 1.96 | 11 |
| B3 | r1.slp | 2.04 | 0.98 | 126.25 | 0.00 | 0.28 | 2.45 | 4 |
| B3 | r2.slp | 2.04 | 0.98 | 121.05 | 0.00 | 0.24 | 0.98 | 5 |
| B3 | r3.slp | 2.04 | 0.98 | 112.07 | 0.00 | 0.25 | 1.47 | 4 |
| B3 | r4.slp | 2.04 | 3.92 | 119.58 | 0.00 | 0.39 | 1.47 | 6 |
| B3 | r5.slp | 2.04 | 1.47 | 112.04 | 0.00 | 0.20 | 0.98 | 3 |
| B3 | r6.slp | 2.04 | 2.45 | 120.39 | 0.00 | 0.21 | 0.98 | 8 |
| B3 | r7.slp | 2.03 | 3.45 | 123.09 | 0.01 | 0.24 | 1.48 | 9 |
| B3 | r8.slp | 2.04 | 0.98 | 117.54 | 0.00 | 0.22 | 0.98 | 4 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 57 |
| `GRAB_PUMMEL>GRAB_WAIT` | 11 |
| `DAMAGE_HIGH_2>LANDING` | 2 |
| `STANDING>GRAB` | 1 |
| `CROUCHING>DOWNTILT` | 1 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `SHIELD_REFLECT>SPOTDODGE` | 1 |
| `DOWN_B_GROUND>KNEE_BEND>DOWN_B_STUN>DOWN_B_AIR` | 1 |
