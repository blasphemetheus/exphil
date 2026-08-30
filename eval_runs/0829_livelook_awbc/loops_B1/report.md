# Loop / Taunt Report

Bot port 1, 7 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | 7/7 | 1.41 [0.84-2.53] | 92.53 [78.29-99.09] | 0.01 [0.00-0.01] | 0.24 [0.15-0.28] | 0.22 [0.00-0.90] | 1.00 [0-4] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | Game_20260829T215746.slp | 1.58 | 1.89 | 99.09 | 0.01 | 0.27 | 0.63 | 3 |
| 2026-08-Mainline | Game_20260829T215927.slp | 1.11 | 0.90 | 90.72 | 0.01 | 0.23 | 0.90 | 4 |
| 2026-08-Mainline | Game_20260829T220041.slp | 0.79 | 2.53 | 78.29 | 0.01 | 0.15 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T220134.slp | 0.69 | 1.45 | 97.34 | 0.00 | 0.25 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T220222.slp | 0.79 | 1.26 | 98.63 | 0.01 | 0.27 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T220316.slp | 1.18 | 0.84 | 95.43 | 0.01 | 0.28 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T220433.slp | 0.98 | 1.03 | 88.21 | 0.01 | 0.24 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 2 |
