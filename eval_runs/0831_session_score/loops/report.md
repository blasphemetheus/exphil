# Loop / Taunt Report

Bot port 1, 19 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | 19/19 | 1.47~1.66 [0.00-2.66] | 83.59~83.26 [68.33-97.58] | 0.00~0.00 [0.00-0.01] | 0.24~0.24 [0.19-0.29] | 0.31~0.00 [0.00-2.17] | 1.11~0 [0-4] |

mean~median [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all. A mean far from its median means
one degenerate game is dragging the group — trust the median for session dirs.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | Game_20260831T133730.slp | 1.51 | 2.65 | 93.45 | 0.00 | 0.24 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T133906.slp | 1.92 | 1.56 | 76.16 | 0.00 | 0.23 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T134108.slp | 1.13 | 2.66 | 71.79 | 0.00 | 0.27 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T134222.slp | 1.32 | 2.27 | 83.26 | 0.01 | 0.22 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T134347.slp | 1.20 | 1.66 | 89.81 | 0.00 | 0.27 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T134506.slp | 1.51 | 0.66 | 68.33 | 0.00 | 0.23 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T134642.slp | 1.87 | 1.07 | 85.18 | 0.00 | 0.24 | 0.54 | 3 |
| 2026-08-Mainline | Game_20260831T134841.slp | 1.15 | 1.73 | 81.46 | 0.00 | 0.21 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T134956.slp | 2.07 | 2.41 | 95.58 | 0.00 | 0.29 | 0.48 | 3 |
| 2026-08-Mainline | Game_20260831T135207.slp | 2.24 | 1.34 | 78.66 | 0.01 | 0.22 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T135427.slp | 1.68 | 1.19 | 78.71 | 0.00 | 0.20 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T135614.slp | 2.01 | 0.50 | 78.79 | 0.00 | 0.28 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T135821.slp | 1.20 | 2.49 | 83.08 | 0.00 | 0.29 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T140334.slp | 1.28 | 0.00 | 91.11 | 0.00 | 0.20 | 0.78 | 4 |
| 2026-08-Mainline | Game_20260831T140457.slp | 0.94 | 2.13 | 91.65 | 0.01 | 0.23 | 1.07 | 3 |
| 2026-08-Mainline | Game_20260831T140600.slp | 1.08 | 0.93 | 76.91 | 0.01 | 0.26 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T140711.slp | 1.84 | 0.00 | 83.09 | 0.00 | 0.19 | 2.17 | 4 |
| 2026-08-Mainline | Game_20260831T140908.slp | 0.94 | 2.12 | 97.58 | 0.01 | 0.24 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260831T141011.slp | 2.22 | 0.45 | 83.62 | 0.00 | 0.19 | 0.90 | 4 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 4 |
| `GRAB_PUMMEL>GRAB_WAIT` | 3 |
| `DAMAGE_FLY_TOP>TECH_MISS_UP>GROUND_ATTACK_UP` | 1 |
| `SHIELD_REFLECT>SPOTDODGE` | 1 |
| `THROWN_UP>DAMAGE_FLY_TOP>TECH_MISS_UP>GROUND_ATTACK_UP>GRAB_PULL>GRABBED` | 1 |
