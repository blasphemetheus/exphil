# Loop / Taunt Report

Bot port 1, 16 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| BASE | 8/8 | 0.91~0.98 [0.00-1.86] | 89.66~89.71 [73.68-102.09] | 0.00~0.00 [0.00-0.00] | 0.33~0.35 [0.25-0.39] | 1.35~1.47 [0.00-2.94] | 4.00~6 [0-7] |
| CRITIC | 8/8 | 0.30~0.50 [0.00-0.80] | 3.44~3.48 [1.20-6.41] | 0.48~0.51 [0.34-0.54] | 0.32~0.32 [0.26-0.39] | 0.96~1.06 [0.00-2.29] | 3.50~4 [0-8] |

mean~median [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all. A mean far from its median means
one degenerate game is dragging the group — trust the median for session dirs.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| BASE | r1.slp | 2.04 | 0.00 | 80.44 | 0.00 | 0.25 | 2.94 | 7 |
| BASE | r2.slp | 2.04 | 1.47 | 84.47 | 0.00 | 0.35 | 1.47 | 6 |
| BASE | r3.slp | 1.61 | 1.86 | 100.52 | 0.00 | 0.36 | 0.00 | 0 |
| BASE | r4.slp | 2.04 | 0.49 | 89.33 | 0.00 | 0.29 | 0.49 | 6 |
| BASE | r5.slp | 2.04 | 0.00 | 73.68 | 0.00 | 0.29 | 2.46 | 6 |
| BASE | r6.slp | 2.04 | 0.98 | 102.09 | 0.00 | 0.39 | 1.47 | 3 |
| BASE | r7.slp | 2.04 | 0.98 | 89.71 | 0.00 | 0.34 | 0.00 | 0 |
| BASE | r8.slp | 2.04 | 1.47 | 97.02 | 0.00 | 0.38 | 1.96 | 4 |
| CRITIC | r1.slp | 1.25 | 0.00 | 2.40 | 0.54 | 0.37 | 0.00 | 0 |
| CRITIC | r2.slp | 0.83 | 0.00 | 1.20 | 0.54 | 0.38 | 1.20 | 6 |
| CRITIC | r3.slp | 2.01 | 0.50 | 3.48 | 0.50 | 0.39 | 0.00 | 0 |
| CRITIC | r4.slp | 0.94 | 0.00 | 2.12 | 0.51 | 0.27 | 1.06 | 8 |
| CRITIC | r5.slp | 2.01 | 0.50 | 5.48 | 0.40 | 0.32 | 1.49 | 4 |
| CRITIC | r6.slp | 1.25 | 0.80 | 6.41 | 0.34 | 0.26 | 0.80 | 4 |
| CRITIC | r7.slp | 1.75 | 0.57 | 4.01 | 0.45 | 0.28 | 2.29 | 3 |
| CRITIC | r8.slp | 1.22 | 0.00 | 2.45 | 0.52 | 0.27 | 0.82 | 3 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 16 |
| `STANDING>WALK_SLOW` | 6 |
| `GRAB_PUMMEL>GRAB_WAIT` | 5 |
| `DAMAGE_NEUTRAL_2>LANDING` | 2 |
| `STANDING>SHIELD_REFLECT>GRAB` | 1 |
| `LANDING>DAMAGE_HIGH_2` | 1 |
| `LANDING>DAMAGE_NEUTRAL_2` | 1 |
| `DAMAGE_HIGH_2>LANDING` | 1 |
