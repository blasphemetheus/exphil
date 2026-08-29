# Loop / Taunt Report

Bot port 1, 16 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| base | 8/8 | 1.29 [0.49-1.96] | 111.63 [99.86-120.83] | 0.00 [0.00-0.01] | 0.27 [0.17-0.36] | 1.53 [0.49-2.94] | 7.38 [5-11] |
| mode16 | 8/8 | 0.22 [0.00-1.78] | 1.12 [0.00-3.57] | 0.74 [0.63-0.85] | 0.51 [0.43-0.60] | 0.51 [0.00-1.78] | 1.50 [0-5] |

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
| mode16 | r1.slp | 0.30 | 0.00 | 0.00 | 0.85 | 0.52 | 0.00 | 0 |
| mode16 | r2.slp | 0.29 | 0.00 | 0.00 | 0.84 | 0.45 | 0.00 | 0 |
| mode16 | r3.slp | 1.23 | 0.00 | 1.62 | 0.64 | 0.47 | 0.00 | 0 |
| mode16 | r4.slp | 0.56 | 1.78 | 3.57 | 0.66 | 0.48 | 1.78 | 4 |
| mode16 | r5.slp | 0.47 | 0.00 | 2.13 | 0.63 | 0.43 | 0.00 | 0 |
| mode16 | r6.slp | 1.79 | 0.00 | 1.67 | 0.74 | 0.59 | 1.12 | 5 |
| mode16 | r7.slp | 0.97 | 0.00 | 0.00 | 0.75 | 0.54 | 0.00 | 0 |
| mode16 | r8.slp | 0.85 | 0.00 | 0.00 | 0.80 | 0.60 | 1.18 | 3 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 15 |
| `GRAB_PUMMEL>GRAB_WAIT` | 7 |
| `STANDING>WALK_SLOW` | 2 |
| `WALK_SLOW>STANDING` | 1 |
| `LANDING>CROUCHING>DAMAGE_NEUTRAL_2` | 1 |
| `DAMAGE_NEUTRAL_1>CROUCH_START>CROUCHING` | 1 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `DOWN_B_GROUND_START>DOWN_B_GROUND>SWORD_DANCE_3_LOW_AIR>CROUCH_START` | 1 |
