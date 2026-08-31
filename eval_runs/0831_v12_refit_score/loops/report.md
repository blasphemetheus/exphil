# Loop / Taunt Report

Bot port 1, 16 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| AR | 8/8 | 0.80~0.98 [0.00-1.47] | 112.00~115.12 [97.58-127.68] | 0.00~0.00 [0.00-0.00] | 0.37~0.37 [0.23-0.55] | 0.25~0.49 [0.00-0.49] | 1.75~3 [0-4] |
| IND | 8/8 | 1.00~0.99 [0.00-2.95] | 112.34~117.79 [82.86-125.75] | 0.00~0.00 [0.00-0.00] | 0.37~0.37 [0.29-0.49] | 0.44~0.00 [0.00-1.76] | 1.38~0 [0-5] |

mean~median [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all. A mean far from its median means
one degenerate game is dragging the group — trust the median for session dirs.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| AR | r1.slp | 2.04 | 0.49 | 97.58 | 0.00 | 0.32 | 0.49 | 3 |
| AR | r2.slp | 2.04 | 0.49 | 107.35 | 0.00 | 0.44 | 0.49 | 4 |
| AR | r3.slp | 2.04 | 1.47 | 116.07 | 0.00 | 0.37 | 0.00 | 0 |
| AR | r4.slp | 2.04 | 0.98 | 119.54 | 0.00 | 0.38 | 0.00 | 0 |
| AR | r5.slp | 2.04 | 0.98 | 115.12 | 0.00 | 0.31 | 0.49 | 4 |
| AR | r6.slp | 1.72 | 0.00 | 99.45 | 0.00 | 0.33 | 0.00 | 0 |
| AR | r7.slp | 2.04 | 0.49 | 127.68 | 0.00 | 0.55 | 0.00 | 0 |
| AR | r8.slp | 2.04 | 1.47 | 113.19 | 0.00 | 0.23 | 0.49 | 3 |
| IND | r1.slp | 0.81 | 0.00 | 82.86 | 0.00 | 0.34 | 1.24 | 3 |
| IND | r2.slp | 2.02 | 0.99 | 102.69 | 0.00 | 0.30 | 0.50 | 3 |
| IND | r3.slp | 2.05 | 0.00 | 113.40 | 0.00 | 0.49 | 0.00 | 0 |
| IND | r4.slp | 2.04 | 1.47 | 123.80 | 0.00 | 0.45 | 0.00 | 0 |
| IND | r5.slp | 1.87 | 0.54 | 117.79 | 0.00 | 0.32 | 0.00 | 0 |
| IND | r6.slp | 1.70 | 1.17 | 125.09 | 0.00 | 0.29 | 1.76 | 5 |
| IND | r7.slp | 1.15 | 0.87 | 107.32 | 0.00 | 0.42 | 0.00 | 0 |
| IND | r8.slp | 2.04 | 2.95 | 125.75 | 0.00 | 0.37 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 4 |
| `DAMAGE_NEUTRAL_2>LANDING` | 2 |
| `SHIELD_REFLECT>ROLL_BACKWARD` | 1 |
| `GRAB>DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `GRAB_PUMMEL>GRAB_WAIT` | 1 |
