# Loop / Taunt Report

Bot port 1, 21 replays, 4 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| base | 8/8 | 2.02 [0.98-4.41] | 431.89 [415.84-454.10] | 0.00 [0.00-0.00] | 0.36 [0.27-0.39] | 2.26 [0.98-2.94] | 10.75 [4-17] |
| btn1.0 | 5/5 | 2.15 [1.47-2.94] | 430.16 [421.80-450.31] | 0.00 [0.00-0.00] | 0.34 [0.27-0.40] | 2.35 [1.47-2.94] | 13.40 [9-17] |
| live_ep10 | 3/3 | 2.17 [1.13-2.93] | 438.44 [425.79-451.65] | 0.00 [0.00-0.00] | 0.38 [0.34-0.40] | 1.22 [0.49-1.69] | 5.67 [3-9] |
| scalar_05 | 5/5 | 1.96 [1.47-2.94] | 438.64 [423.24-452.14] | 0.00 [0.00-0.00] | 0.35 [0.26-0.40] | 2.06 [0.49-2.94] | 8.60 [5-12] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| base | r1.slp | 2.04 | 1.47 | 429.75 | 0.00 | 0.34 | 2.94 | 8 |
| base | r2.slp | 2.04 | 4.41 | 454.10 | 0.00 | 0.37 | 1.47 | 4 |
| base | r3.slp | 2.04 | 1.47 | 429.87 | 0.00 | 0.37 | 2.94 | 17 |
| base | r4.slp | 2.04 | 1.96 | 437.64 | 0.00 | 0.38 | 0.98 | 9 |
| base | r5.slp | 2.04 | 3.43 | 431.59 | 0.00 | 0.37 | 1.96 | 12 |
| base | r6.slp | 2.04 | 0.98 | 415.84 | 0.00 | 0.27 | 2.94 | 13 |
| base | r7.slp | 2.04 | 0.98 | 425.37 | 0.00 | 0.39 | 2.94 | 17 |
| base | r8.slp | 2.04 | 1.47 | 430.96 | 0.00 | 0.37 | 1.96 | 6 |
| btn1.0 | r1.slp | 2.04 | 2.45 | 450.31 | 0.00 | 0.40 | 1.47 | 9 |
| btn1.0 | r2.slp | 2.04 | 1.96 | 424.51 | 0.00 | 0.27 | 2.93 | 14 |
| btn1.0 | r3.slp | 2.04 | 2.94 | 427.91 | 0.00 | 0.34 | 2.94 | 14 |
| btn1.0 | r4.slp | 2.04 | 1.47 | 426.26 | 0.00 | 0.37 | 1.47 | 17 |
| btn1.0 | r5.slp | 2.04 | 1.96 | 421.80 | 0.00 | 0.32 | 2.94 | 13 |
| live_ep10 | r1.slp | 1.77 | 1.13 | 425.79 | 0.00 | 0.40 | 1.69 | 9 |
| live_ep10 | r2.slp | 2.05 | 2.93 | 451.65 | 0.00 | 0.40 | 0.49 | 3 |
| live_ep10 | r3.slp | 2.04 | 2.45 | 437.89 | 0.00 | 0.34 | 1.47 | 5 |
| scalar_05 | r1.slp | 2.05 | 1.96 | 452.14 | 0.00 | 0.39 | 0.49 | 5 |
| scalar_05 | r2.slp | 2.04 | 1.47 | 423.24 | 0.00 | 0.33 | 2.94 | 10 |
| scalar_05 | r3.slp | 2.04 | 1.47 | 437.15 | 0.00 | 0.40 | 1.96 | 12 |
| scalar_05 | r4.slp | 2.04 | 1.96 | 443.27 | 0.00 | 0.39 | 2.94 | 7 |
| scalar_05 | r5.slp | 2.04 | 2.94 | 437.37 | 0.00 | 0.26 | 1.96 | 9 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 62 |
| `GRAB_PUMMEL>GRAB_WAIT` | 25 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `GRAB>SHIELD_START` | 1 |
