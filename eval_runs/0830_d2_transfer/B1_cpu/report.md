# Loop / Taunt Report

Bot port 1, 8 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| B1 | 8/8 | 2.04 [1.47-3.43] | 114.63 [101.79-131.18] | 0.00 [0.00-0.00] | 0.25 [0.15-0.30] | 1.49 [0.49-3.43] | 8.38 [3-14] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| B1 | r1.slp | 2.04 | 1.96 | 108.26 | 0.00 | 0.30 | 1.96 | 11 |
| B1 | r2.slp | 1.85 | 1.62 | 128.03 | 0.00 | 0.29 | 2.16 | 11 |
| B1 | r3.slp | 2.04 | 1.47 | 110.17 | 0.00 | 0.15 | 1.47 | 9 |
| B1 | r4.slp | 2.04 | 1.96 | 131.18 | 0.00 | 0.25 | 0.49 | 3 |
| B1 | r5.slp | 2.04 | 1.47 | 101.79 | 0.00 | 0.30 | 3.43 | 14 |
| B1 | r6.slp | 2.04 | 3.43 | 112.10 | 0.00 | 0.26 | 0.49 | 6 |
| B1 | r7.slp | 2.05 | 2.93 | 110.48 | 0.00 | 0.20 | 0.49 | 3 |
| B1 | r8.slp | 2.04 | 1.47 | 115.01 | 0.00 | 0.29 | 1.47 | 10 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 17 |
| `GRAB_PUMMEL>GRAB_WAIT` | 6 |
| `DOWN_B_GROUND>KNEE_BEND>DOWN_B_STUN>DOWN_B_AIR` | 1 |
