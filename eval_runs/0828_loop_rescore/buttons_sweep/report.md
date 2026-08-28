# Loop / Taunt Report

Bot port 1, 20 replays, 4 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | n | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| btn0.5 | 5 | 2.25 [1.96-2.45] | 112.27 [95.39-119.49] | 0.00 [0.00-0.01] | 0.26 [0.20-0.30] | 1.57 [0.98-2.45] | 10.80 [3-14] |
| btn0.6 | 5 | 1.17 [0.49-1.96] | 170.54 [157.05-179.12] | 0.00 [0.00-0.00] | 0.24 [0.19-0.30] | 1.96 [0.98-3.91] | 10.40 [6-16] |
| btn0.7 | 5 | 2.94 [0.98-4.89] | 247.85 [236.51-259.38] | 0.00 [0.00-0.00] | 0.30 [0.27-0.34] | 1.96 [0.98-3.43] | 12.00 [5-19] |
| btn1.0 | 5 | 2.15 [1.47-2.94] | 430.16 [421.80-450.31] | 0.00 [0.00-0.00] | 0.34 [0.27-0.40] | 2.35 [1.47-2.94] | 13.40 [9-17] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| btn0.5 | r1.slp | 2.04 | 1.96 | 111.70 | 0.00 | 0.24 | 0.98 | 3 |
| btn0.5 | r2.slp | 2.04 | 2.45 | 119.49 | 0.01 | 0.20 | 2.45 | 10 |
| btn0.5 | r3.slp | 2.04 | 2.45 | 117.41 | 0.01 | 0.30 | 0.98 | 14 |
| btn0.5 | r4.slp | 2.04 | 1.96 | 95.39 | 0.00 | 0.28 | 2.45 | 13 |
| btn0.5 | r5.slp | 2.05 | 2.44 | 117.34 | 0.00 | 0.28 | 0.98 | 14 |
| btn0.6 | r1.slp | 2.04 | 0.49 | 157.05 | 0.00 | 0.19 | 3.91 | 16 |
| btn0.6 | r2.slp | 2.04 | 1.96 | 179.12 | 0.00 | 0.23 | 1.47 | 6 |
| btn0.6 | r3.slp | 2.04 | 1.96 | 172.43 | 0.00 | 0.25 | 1.47 | 6 |
| btn0.6 | r4.slp | 2.04 | 0.49 | 169.35 | 0.00 | 0.25 | 0.98 | 11 |
| btn0.6 | r5.slp | 2.04 | 0.98 | 174.76 | 0.00 | 0.30 | 1.96 | 13 |
| btn0.7 | r1.slp | 2.04 | 2.45 | 241.73 | 0.00 | 0.27 | 1.96 | 14 |
| btn0.7 | r2.slp | 2.04 | 3.92 | 253.50 | 0.00 | 0.34 | 1.96 | 19 |
| btn0.7 | r3.slp | 2.04 | 2.45 | 248.16 | 0.00 | 0.27 | 1.47 | 14 |
| btn0.7 | r4.slp | 2.04 | 4.89 | 259.38 | 0.00 | 0.29 | 0.98 | 5 |
| btn0.7 | r5.slp | 2.04 | 0.98 | 236.51 | 0.00 | 0.33 | 3.43 | 8 |
| btn1.0 | r1.slp | 2.04 | 2.45 | 450.31 | 0.00 | 0.40 | 1.47 | 9 |
| btn1.0 | r2.slp | 2.04 | 1.96 | 424.51 | 0.00 | 0.27 | 2.93 | 14 |
| btn1.0 | r3.slp | 2.04 | 2.94 | 427.91 | 0.00 | 0.34 | 2.94 | 14 |
| btn1.0 | r4.slp | 2.04 | 1.47 | 426.26 | 0.00 | 0.37 | 1.47 | 17 |
| btn1.0 | r5.slp | 2.04 | 1.96 | 421.80 | 0.00 | 0.32 | 2.94 | 13 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 58 |
| `GRAB_PUMMEL>GRAB_WAIT` | 20 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `SHIELD_REFLECT>SPOTDODGE` | 1 |
