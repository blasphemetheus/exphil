# Loop / Taunt Report

Bot port 1, 16 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| AR | 8/8 | 0.92~0.52 [0.49-1.96] | 86.59~85.88 [81.38-97.12] | 0.00~0.00 [0.00-0.01] | 0.32~0.32 [0.23-0.40] | 1.55~1.47 [0.49-2.59] | 6.38~6 [3-10] |
| IND | 8/8 | 1.30~1.47 [0.00-2.95] | 88.25~89.53 [65.77-108.02] | 0.00~0.00 [0.00-0.01] | 0.35~0.42 [0.17-0.46] | 1.97~1.96 [0.98-3.92] | 7.50~6 [4-13] |

mean~median [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all. A mean far from its median means
one degenerate game is dragging the group — trust the median for session dirs.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| AR | r1.slp | 2.04 | 0.49 | 86.36 | 0.01 | 0.40 | 1.47 | 4 |
| AR | r2.slp | 2.04 | 1.47 | 97.12 | 0.00 | 0.27 | 1.96 | 6 |
| AR | r3.slp | 2.04 | 1.96 | 81.38 | 0.00 | 0.32 | 2.45 | 9 |
| AR | r4.slp | 1.93 | 0.52 | 83.49 | 0.00 | 0.27 | 2.59 | 6 |
| AR | r5.slp | 2.04 | 0.49 | 82.91 | 0.00 | 0.23 | 1.47 | 10 |
| AR | r6.slp | 2.04 | 0.49 | 85.88 | 0.00 | 0.37 | 1.47 | 5 |
| AR | r7.slp | 2.04 | 1.47 | 83.24 | 0.00 | 0.37 | 0.49 | 8 |
| AR | r8.slp | 2.04 | 0.49 | 92.30 | 0.00 | 0.31 | 0.49 | 3 |
| IND | r1.slp | 2.04 | 1.47 | 106.76 | 0.00 | 0.33 | 1.96 | 4 |
| IND | r2.slp | 2.04 | 1.47 | 100.44 | 0.00 | 0.34 | 3.92 | 9 |
| IND | r3.slp | 2.04 | 0.00 | 69.12 | 0.01 | 0.43 | 2.45 | 10 |
| IND | r4.slp | 2.04 | 0.98 | 81.17 | 0.01 | 0.42 | 1.47 | 6 |
| IND | r5.slp | 2.04 | 0.49 | 65.77 | 0.00 | 0.17 | 2.45 | 13 |
| IND | r6.slp | 1.90 | 1.58 | 89.53 | 0.00 | 0.27 | 1.58 | 6 |
| IND | r7.slp | 2.04 | 2.95 | 108.02 | 0.00 | 0.46 | 0.98 | 6 |
| IND | r8.slp | 2.04 | 1.47 | 85.22 | 0.00 | 0.42 | 0.98 | 6 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 37 |
| `GRAB_PUMMEL>GRAB_WAIT` | 15 |
| `DAMAGE_NEUTRAL_2>LANDING` | 3 |
| `KNEE_BEND>GRAB` | 1 |
| `SHIELD_START>ROLL_BACKWARD` | 1 |
