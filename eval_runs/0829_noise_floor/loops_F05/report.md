# Loop / Taunt Report

Bot port 1, 20 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| base | 8/8 | 1.29 [0.49-1.96] | 111.63 [99.86-120.83] | 0.00 [0.00-0.01] | 0.27 [0.17-0.36] | 1.53 [0.49-2.94] | 7.38 [5-11] |
| btn0.5 | 9/12 | 1.96 [0.49-2.95] | 111.45 [95.39-119.49] | 0.00 [0.00-0.01] | 0.26 [0.16-0.34] | 1.52 [0.49-2.94] | 8.56 [3-14] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| base | r1.slp | 2.04 | 0.98 | 105.23 | 0.00 | 0.17 | 1.47 | 6 |
| base | r2.slp | 2.04 | 1.96 | 117.01 | 0.01 | 0.28 | 1.96 | 6 |
| base | r3.slp | 2.04 | 1.47 | 111.63 | 0.00 | 0.23 | 1.47 | 11 |
| base | r4.slp | 2.05 | 0.98 | 119.28 | 0.00 | 0.20 | 0.98 | 11 |
| base | r5.slp | 2.04 | 0.98 | 105.38 | 0.00 | 0.32 | 1.47 | 5 |
| base | r6.slp | 2.04 | 1.47 | 99.86 | 0.00 | 0.29 | 2.94 | 8 |
| base | r7.slp | 2.04 | 1.96 | 120.83 | 0.00 | 0.29 | 1.47 | 7 |
| base | r8.slp | 2.03 | 0.49 | 113.78 | 0.00 | 0.36 | 0.49 | 5 |
| btn0.5 | r1.slp | 2.04 | 1.47 | 108.88 | 0.00 | 0.34 | 2.94 | 9 |
| btn0.5 | r2.slp | 2.04 | 0.49 | 105.74 | 0.01 | 0.22 | 0.98 | 4 |
| btn0.5 | r3.slp | 2.05 | 1.47 | 112.44 | 0.01 | 0.16 | 0.49 | 3 |
| btn0.5 | r8.slp | 2.03 | 2.95 | 114.65 | 0.00 | 0.27 | 1.48 | 7 |
| btn0.5 | r1.slp | 2.04 | 1.96 | 111.70 | 0.00 | 0.24 | 0.98 | 3 |
| btn0.5 | r2.slp | 2.04 | 2.45 | 119.49 | 0.01 | 0.20 | 2.45 | 10 |
| btn0.5 | r3.slp | 2.04 | 2.45 | 117.41 | 0.01 | 0.30 | 0.98 | 14 |
| btn0.5 | r4.slp | 2.04 | 1.96 | 95.39 | 0.00 | 0.28 | 2.45 | 13 |
| btn0.5 | r5.slp | 2.05 | 2.44 | 117.34 | 0.00 | 0.28 | 0.98 | 14 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 32 |
| `GRAB_PUMMEL>GRAB_WAIT` | 17 |
| `STANDING>WALK_SLOW` | 1 |
| `WALK_SLOW>STANDING` | 1 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `SHIELD_REFLECT>SPOTDODGE` | 1 |
