# Loop / Taunt Report

Bot port 1, 8 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| B2 | 8/8 | 1.87 [0.49-2.94] | 126.85 [122.38-132.54] | 0.00 [0.00-0.01] | 0.27 [0.20-0.33] | 1.80 [1.47-2.45] | 8.38 [7-11] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| B2 | r1.slp | 2.04 | 0.49 | 129.99 | 0.00 | 0.31 | 1.47 | 7 |
| B2 | r2.slp | 2.04 | 0.98 | 129.25 | 0.00 | 0.24 | 1.96 | 7 |
| B2 | r3.slp | 2.04 | 2.45 | 122.91 | 0.00 | 0.33 | 2.45 | 8 |
| B2 | r4.slp | 2.04 | 1.96 | 126.30 | 0.00 | 0.29 | 1.96 | 7 |
| B2 | r5.slp | 1.78 | 2.24 | 126.23 | 0.00 | 0.21 | 1.68 | 10 |
| B2 | r6.slp | 2.04 | 2.94 | 125.23 | 0.00 | 0.32 | 1.47 | 7 |
| B2 | r7.slp | 2.04 | 2.93 | 132.54 | 0.00 | 0.26 | 1.47 | 10 |
| B2 | r8.slp | 2.04 | 0.98 | 122.38 | 0.01 | 0.20 | 1.96 | 11 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 25 |
| `GRAB_PUMMEL>GRAB_WAIT` | 2 |
| `CROUCHING>DOWNTILT` | 1 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
