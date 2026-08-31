# Loop / Taunt Report

Bot port 1, 16 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| AR | 8/8 | 2.23~2.45 [1.47-3.43] | 99.11~98.53 [88.69-111.75] | 0.00~0.00 [0.00-0.01] | 0.20~0.20 [0.18-0.23] | 0.61~0.49 [0.00-1.47] | 2.88~4 [0-7] |
| IND | 8/8 | 2.01~1.94 [0.49-3.91] | 112.79~114.79 [105.00-120.91] | 0.00~0.00 [0.00-0.00] | 0.24~0.23 [0.20-0.32] | 2.08~2.44 [0.49-3.42] | 6.50~7 [3-13] |

mean~median [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all. A mean far from its median means
one degenerate game is dragging the group — trust the median for session dirs.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| AR | r1.slp | 2.04 | 2.94 | 102.77 | 0.01 | 0.23 | 0.00 | 0 |
| AR | r2.slp | 2.04 | 3.43 | 98.53 | 0.00 | 0.19 | 0.49 | 4 |
| AR | r3.slp | 2.04 | 2.45 | 95.99 | 0.01 | 0.20 | 0.49 | 7 |
| AR | r4.slp | 2.04 | 1.47 | 96.40 | 0.00 | 0.22 | 0.00 | 0 |
| AR | r5.slp | 1.78 | 1.68 | 102.68 | 0.00 | 0.18 | 0.00 | 0 |
| AR | r6.slp | 2.04 | 1.47 | 96.04 | 0.00 | 0.19 | 1.47 | 4 |
| AR | r7.slp | 2.04 | 2.94 | 111.75 | 0.00 | 0.22 | 0.98 | 4 |
| AR | r8.slp | 2.04 | 1.47 | 88.69 | 0.00 | 0.20 | 1.47 | 4 |
| IND | r1.slp | 1.03 | 1.94 | 105.00 | 0.00 | 0.23 | 0.97 | 3 |
| IND | r2.slp | 2.05 | 3.91 | 114.79 | 0.00 | 0.23 | 2.44 | 13 |
| IND | r3.slp | 2.04 | 1.47 | 120.91 | 0.00 | 0.32 | 2.45 | 6 |
| IND | r4.slp | 2.05 | 3.91 | 117.30 | 0.00 | 0.22 | 2.44 | 7 |
| IND | r5.slp | 2.05 | 0.49 | 109.98 | 0.00 | 0.20 | 1.47 | 6 |
| IND | r6.slp | 2.05 | 1.47 | 107.02 | 0.00 | 0.24 | 2.93 | 7 |
| IND | r7.slp | 2.04 | 1.96 | 108.07 | 0.00 | 0.22 | 0.49 | 3 |
| IND | r8.slp | 2.05 | 0.98 | 119.25 | 0.00 | 0.25 | 3.42 | 7 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 21 |
| `GRAB_PUMMEL>GRAB_WAIT` | 16 |
| `SHIELD_START>SPOTDODGE` | 2 |
| `LANDING>DAMAGE_HIGH_2` | 1 |
| `DAMAGE_HIGH_2>LANDING` | 1 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `SHIELD_START>ROLL_BACKWARD` | 1 |
