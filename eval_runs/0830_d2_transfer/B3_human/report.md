# Loop / Taunt Report

Bot port 1, 5 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | 5/5 | 1.59 [0.00-3.02] | 102.11 [96.25-113.88] | 0.00 [0.00-0.01] | 0.21 [0.04-0.36] | 0.81 [0.00-2.01] | 2.20 [0-4] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | Game_20260829T221232.slp | 1.46 | 1.37 | 102.27 | 0.01 | 0.36 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T221405.slp | 1.97 | 1.53 | 113.88 | 0.00 | 0.24 | 1.02 | 3 |
| 2026-08-Mainline | Game_20260829T221610.slp | 0.99 | 3.02 | 99.72 | 0.01 | 0.19 | 1.01 | 4 |
| 2026-08-Mainline | Game_20260829T221716.slp | 1.99 | 2.01 | 96.25 | 0.00 | 0.21 | 2.01 | 4 |
| 2026-08-Mainline | Game_20260829T221922.slp | 0.41 | 0.00 | 98.43 | 0.00 | 0.04 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 5 |
| `GRAB_PUMMEL>GRAB_WAIT` | 2 |
