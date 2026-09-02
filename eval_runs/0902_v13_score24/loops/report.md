# Loop / Taunt Report

Bot port 1, 48 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| AR | 24/24 | 1.35~1.47 [0.00-2.94] | 92.43~91.20 [64.23-122.99] | 0.00~0.00 [0.00-0.01] | 0.33~0.32 [0.24-0.41] | 1.00~0.98 [0.00-3.43] | 4.50~4 [0-9] |
| IND | 24/24 | 1.33~0.98 [0.49-3.92] | 86.35~86.20 [59.18-114.08] | 0.00~0.00 [0.00-0.01] | 0.33~0.33 [0.21-0.45] | 2.39~2.45 [0.49-4.41] | 8.92~10 [3-16] |

mean~median [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all. A mean far from its median means
one degenerate game is dragging the group — trust the median for session dirs.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| AR | r1.slp | 2.04 | 1.47 | 93.06 | 0.00 | 0.33 | 0.49 | 4 |
| AR | r10.slp | 2.04 | 1.47 | 88.67 | 0.00 | 0.38 | 0.49 | 5 |
| AR | r11.slp | 2.04 | 0.00 | 74.47 | 0.00 | 0.31 | 0.98 | 5 |
| AR | r12.slp | 2.04 | 0.98 | 75.47 | 0.00 | 0.35 | 0.49 | 4 |
| AR | r13.slp | 2.04 | 1.47 | 114.32 | 0.00 | 0.25 | 0.98 | 4 |
| AR | r14.slp | 2.04 | 2.94 | 95.18 | 0.00 | 0.37 | 1.47 | 6 |
| AR | r15.slp | 2.04 | 0.49 | 91.64 | 0.01 | 0.33 | 0.49 | 4 |
| AR | r16.slp | 2.04 | 0.98 | 64.23 | 0.00 | 0.31 | 3.43 | 7 |
| AR | r17.slp | 2.04 | 0.98 | 83.36 | 0.00 | 0.30 | 0.49 | 9 |
| AR | r18.slp | 2.04 | 0.98 | 95.55 | 0.00 | 0.36 | 0.98 | 5 |
| AR | r19.slp | 2.04 | 1.47 | 89.19 | 0.00 | 0.37 | 0.98 | 3 |
| AR | r2.slp | 2.04 | 2.45 | 122.99 | 0.00 | 0.40 | 0.00 | 0 |
| AR | r20.slp | 2.04 | 2.45 | 91.20 | 0.00 | 0.35 | 0.98 | 5 |
| AR | r21.slp | 2.04 | 2.45 | 101.02 | 0.00 | 0.31 | 0.00 | 0 |
| AR | r22.slp | 2.04 | 0.00 | 91.15 | 0.00 | 0.30 | 0.98 | 4 |
| AR | r23.slp | 2.04 | 0.98 | 106.36 | 0.00 | 0.41 | 0.00 | 0 |
| AR | r24.slp | 2.04 | 0.98 | 89.33 | 0.00 | 0.24 | 1.47 | 4 |
| AR | r3.slp | 2.04 | 2.45 | 115.70 | 0.00 | 0.31 | 0.98 | 4 |
| AR | r4.slp | 2.04 | 0.49 | 85.83 | 0.00 | 0.30 | 1.47 | 7 |
| AR | r5.slp | 2.04 | 1.47 | 100.01 | 0.00 | 0.37 | 0.98 | 4 |
| AR | r6.slp | 2.04 | 0.98 | 84.35 | 0.00 | 0.32 | 0.98 | 4 |
| AR | r7.slp | 2.04 | 0.00 | 75.06 | 0.00 | 0.31 | 1.47 | 9 |
| AR | r8.slp | 2.04 | 2.45 | 89.25 | 0.00 | 0.32 | 0.98 | 4 |
| AR | r9.slp | 2.04 | 1.96 | 100.95 | 0.00 | 0.30 | 2.45 | 7 |
| IND | r1.slp | 2.04 | 0.98 | 85.15 | 0.00 | 0.36 | 1.96 | 10 |
| IND | r10.slp | 2.04 | 0.98 | 92.08 | 0.01 | 0.26 | 1.96 | 6 |
| IND | r11.slp | 2.04 | 1.47 | 73.49 | 0.00 | 0.30 | 2.45 | 13 |
| IND | r12.slp | 2.04 | 0.98 | 77.78 | 0.00 | 0.39 | 2.45 | 10 |
| IND | r13.slp | 2.04 | 1.47 | 80.77 | 0.00 | 0.33 | 2.45 | 6 |
| IND | r14.slp | 2.05 | 0.49 | 70.40 | 0.00 | 0.31 | 4.40 | 16 |
| IND | r15.slp | 2.04 | 0.49 | 82.29 | 0.00 | 0.34 | 2.94 | 12 |
| IND | r16.slp | 2.04 | 0.98 | 88.65 | 0.00 | 0.41 | 1.47 | 5 |
| IND | r17.slp | 2.04 | 0.49 | 59.18 | 0.00 | 0.21 | 4.40 | 12 |
| IND | r18.slp | 2.04 | 1.96 | 101.43 | 0.00 | 0.45 | 0.98 | 11 |
| IND | r19.slp | 2.04 | 1.47 | 86.20 | 0.00 | 0.29 | 0.98 | 10 |
| IND | r2.slp | 2.04 | 0.98 | 85.11 | 0.00 | 0.25 | 3.91 | 7 |
| IND | r20.slp | 2.04 | 1.47 | 108.13 | 0.00 | 0.33 | 0.49 | 3 |
| IND | r21.slp | 2.04 | 2.94 | 95.46 | 0.00 | 0.40 | 0.49 | 3 |
| IND | r22.slp | 2.04 | 3.92 | 114.08 | 0.00 | 0.27 | 3.92 | 13 |
| IND | r23.slp | 2.04 | 0.98 | 69.53 | 0.00 | 0.35 | 2.94 | 10 |
| IND | r24.slp | 2.04 | 0.49 | 70.43 | 0.00 | 0.39 | 1.96 | 9 |
| IND | r3.slp | 2.04 | 0.49 | 87.08 | 0.00 | 0.25 | 1.47 | 10 |
| IND | r4.slp | 2.04 | 0.98 | 94.88 | 0.00 | 0.34 | 2.45 | 8 |
| IND | r5.slp | 2.02 | 1.98 | 86.09 | 0.01 | 0.28 | 2.47 | 11 |
| IND | r6.slp | 2.04 | 0.98 | 84.69 | 0.00 | 0.32 | 4.41 | 7 |
| IND | r7.slp | 2.04 | 1.96 | 91.03 | 0.01 | 0.31 | 3.43 | 9 |
| IND | r8.slp | 2.04 | 1.47 | 90.10 | 0.00 | 0.28 | 1.47 | 8 |
| IND | r9.slp | 2.04 | 1.47 | 98.40 | 0.00 | 0.43 | 1.47 | 5 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 113 |
| `GRAB_PUMMEL>GRAB_WAIT` | 44 |
| `DAMAGE_NEUTRAL_2>LANDING` | 4 |
| `STANDING>LASER_GUN_PULL>NEUTRAL_B_CHARGING>NEUTRAL_B_ATTACKING` | 1 |
| `NEUTRAL_ATTACK_1>NEUTRAL_ATTACK_2>LOOPING_ATTACK_START>LOOPING_ATTACK_MIDDLE>LOOPING_ATTACK_END` | 1 |
| `SHIELD_START>ROLL_BACKWARD` | 1 |
| `SHIELD_REFLECT>SPOTDODGE>CROUCH_START` | 1 |
| `ROLL_BACKWARD>SHIELD_START` | 1 |
