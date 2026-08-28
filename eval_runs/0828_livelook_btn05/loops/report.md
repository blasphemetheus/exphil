# Loop / Taunt Report

Bot port 1, 11 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | 11/11 | 331.65 [0.00-3600.00] | 93.19 [0.00-320.79] | 0.00 [0.00-0.01] | 0.16 [0.00-0.28] | 0.74 [0.00-1.96] | 3.36 [0-9] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | Game_20260828T165031.slp | 2.02 | 0.49 | 99.90 | 0.00 | 0.26 | 1.48 | 4 |
| 2026-08-Mainline | Game_20260828T165239.slp | 0.00 | 3600.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260828T165244.slp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260828T165248.slp | 0.03 | 35.64 | 320.79 | 0.00 | 0.00 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260828T165324.slp | 1.58 | 2.53 | 100.05 | 0.01 | 0.20 | 0.63 | 3 |
| 2026-08-Mainline | Game_20260828T165505.slp | 2.77 | 1.80 | 90.13 | 0.00 | 0.21 | 0.72 | 9 |
| 2026-08-Mainline | Game_20260828T165758.slp | 2.24 | 2.23 | 104.90 | 0.00 | 0.28 | 1.79 | 7 |
| 2026-08-Mainline | Game_20260828T170020.slp | 2.39 | 2.09 | 87.01 | 0.01 | 0.28 | 0.84 | 4 |
| 2026-08-Mainline | Game_20260828T170250.slp | 2.88 | 1.04 | 97.28 | 0.01 | 0.26 | 0.69 | 4 |
| 2026-08-Mainline | Game_20260828T170549.slp | 2.55 | 2.35 | 125.02 | 0.01 | 0.28 | 1.96 | 6 |
| 2026-08-Mainline | Game_20260828T170829.slp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 10 |
| `GRAB_PUMMEL>GRAB_WAIT` | 3 |
| `LANDING>DAMAGE_NEUTRAL_2` | 2 |
| `STANDING>WALK_SLOW` | 1 |
| `DAMAGE_HIGH_3>LANDING` | 1 |
| `DAMAGE_NEUTRAL_3>LANDING` | 1 |
| `DAMAGE_FLY_TOP>TECH_MISS_UP>GROUND_ATTACK_UP` | 1 |
