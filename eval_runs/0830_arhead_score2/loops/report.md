# Loop / Taunt Report

Bot port 1, 16 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| AR | 8/8 | 1.63 [0.49-2.94] | 94.36 [67.17-111.67] | 0.00 [0.00-0.00] | 0.31 [0.24-0.40] | 0.18 [0.00-0.49] | 1.13 [0-3] |
| IND | 8/8 | 0.46 [0.00-0.87] | 78.92 [70.54-89.02] | 0.00 [0.00-0.00] | 0.37 [0.20-0.48] | 0.34 [0.00-1.75] | 1.13 [0-3] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| AR | r1.slp | 1.08 | 2.78 | 109.29 | 0.00 | 0.28 | 0.00 | 0 |
| AR | r2.slp | 2.04 | 1.96 | 92.26 | 0.00 | 0.29 | 0.49 | 3 |
| AR | r3.slp | 2.04 | 0.49 | 88.70 | 0.00 | 0.28 | 0.00 | 0 |
| AR | r4.slp | 2.04 | 1.96 | 108.84 | 0.00 | 0.32 | 0.00 | 0 |
| AR | r5.slp | 2.04 | 0.98 | 67.17 | 0.00 | 0.40 | 0.00 | 0 |
| AR | r6.slp | 2.04 | 0.49 | 85.29 | 0.00 | 0.24 | 0.49 | 3 |
| AR | r7.slp | 2.04 | 1.47 | 91.67 | 0.00 | 0.29 | 0.49 | 3 |
| AR | r8.slp | 0.68 | 2.94 | 111.67 | 0.00 | 0.34 | 0.00 | 0 |
| IND | r1.slp | 2.04 | 0.49 | 77.38 | 0.00 | 0.44 | 0.00 | 0 |
| IND | r2.slp | 2.04 | 0.00 | 87.60 | 0.00 | 0.48 | 0.00 | 0 |
| IND | r3.slp | 2.04 | 0.49 | 79.27 | 0.00 | 0.33 | 0.49 | 3 |
| IND | r4.slp | 2.04 | 0.49 | 77.38 | 0.00 | 0.32 | 0.49 | 3 |
| IND | r5.slp | 1.15 | 0.87 | 89.02 | 0.00 | 0.20 | 1.75 | 3 |
| IND | r6.slp | 2.04 | 0.00 | 70.98 | 0.00 | 0.45 | 0.00 | 0 |
| IND | r7.slp | 1.19 | 0.84 | 70.54 | 0.00 | 0.40 | 0.00 | 0 |
| IND | r8.slp | 2.05 | 0.49 | 79.17 | 0.00 | 0.37 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 4 |
| `SHIELD_START>SPOTDODGE` | 1 |
| `SHIELD_REFLECT>SPOTDODGE` | 1 |
| `MARTH_COUNTER>DOWN_B_AIR` | 1 |
