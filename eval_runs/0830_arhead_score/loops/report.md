# Loop / Taunt Report

Bot port 1, 16 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| AR | 8/8 | 0.00 [0.00-0.00] | 0.00 [0.00-0.00] | 0.49 [0.39-0.62] | 0.31 [0.26-0.43] | 6.02 [2.21-10.19] | 6.88 [3-12] |
| IND | 8/8 | 0.00 [0.00-0.00] | 0.00 [0.00-0.00] | 0.37 [0.25-0.43] | 0.23 [0.16-0.30] | 0.57 [0.00-1.61] | 1.75 [0-6] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| AR | r1.slp | 0.61 | 0.00 | 0.00 | 0.56 | 0.31 | 3.28 | 11 |
| AR | r2.slp | 0.45 | 0.00 | 0.00 | 0.62 | 0.43 | 2.21 | 3 |
| AR | r3.slp | 0.50 | 0.00 | 0.00 | 0.51 | 0.34 | 6.00 | 3 |
| AR | r4.slp | 0.78 | 0.00 | 0.00 | 0.50 | 0.26 | 10.19 | 5 |
| AR | r5.slp | 0.67 | 0.00 | 0.00 | 0.43 | 0.27 | 8.91 | 7 |
| AR | r6.slp | 0.48 | 0.00 | 0.00 | 0.54 | 0.35 | 4.13 | 4 |
| AR | r7.slp | 0.79 | 0.00 | 0.00 | 0.40 | 0.28 | 7.56 | 10 |
| AR | r8.slp | 0.51 | 0.00 | 0.00 | 0.39 | 0.26 | 5.89 | 12 |
| IND | r1.slp | 0.38 | 0.00 | 0.00 | 0.38 | 0.26 | 0.00 | 0 |
| IND | r2.slp | 0.64 | 0.00 | 0.00 | 0.43 | 0.30 | 1.57 | 5 |
| IND | r3.slp | 0.62 | 0.00 | 0.00 | 0.35 | 0.16 | 1.61 | 6 |
| IND | r4.slp | 0.42 | 0.00 | 0.00 | 0.37 | 0.24 | 0.00 | 0 |
| IND | r5.slp | 0.36 | 0.00 | 0.00 | 0.41 | 0.28 | 0.00 | 0 |
| IND | r6.slp | 0.47 | 0.00 | 0.00 | 0.43 | 0.25 | 0.00 | 0 |
| IND | r7.slp | 0.71 | 0.00 | 0.00 | 0.25 | 0.16 | 0.00 | 0 |
| IND | r8.slp | 0.72 | 0.00 | 0.00 | 0.33 | 0.16 | 1.39 | 3 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `TURNING>DASHING` | 20 |
| `DASHING>TURNING` | 10 |
| `STANDING>WALK_SLOW` | 1 |
| `CROUCH_START>DAMAGE_NEUTRAL_1` | 1 |
| `LANDING>DAMAGE_HIGH_2` | 1 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
