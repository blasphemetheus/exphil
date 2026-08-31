# Loop / Taunt Report

Bot port 1, 8 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| B3 | 8/8 | 1.90 [0.98-3.92] | 119.00 [112.04-126.25] | 0.00 [0.00-0.01] | 0.25 [0.20-0.39] | 1.35 [0.98-2.45] | 5.38 [3-9] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| B3 | r1.slp | 2.04 | 0.98 | 126.25 | 0.00 | 0.28 | 2.45 | 4 |
| B3 | r2.slp | 2.04 | 0.98 | 121.05 | 0.00 | 0.24 | 0.98 | 5 |
| B3 | r3.slp | 2.04 | 0.98 | 112.07 | 0.00 | 0.25 | 1.47 | 4 |
| B3 | r4.slp | 2.04 | 3.92 | 119.58 | 0.00 | 0.39 | 1.47 | 6 |
| B3 | r5.slp | 2.04 | 1.47 | 112.04 | 0.00 | 0.20 | 0.98 | 3 |
| B3 | r6.slp | 2.04 | 2.45 | 120.39 | 0.00 | 0.21 | 0.98 | 8 |
| B3 | r7.slp | 2.03 | 3.45 | 123.09 | 0.01 | 0.24 | 1.48 | 9 |
| B3 | r8.slp | 2.04 | 0.98 | 117.54 | 0.00 | 0.22 | 0.98 | 4 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 15 |
| `GRAB_PUMMEL>GRAB_WAIT` | 3 |
| `DAMAGE_HIGH_2>LANDING` | 2 |
| `STANDING>GRAB` | 1 |
| `SHIELD_REFLECT>SPOTDODGE` | 1 |
