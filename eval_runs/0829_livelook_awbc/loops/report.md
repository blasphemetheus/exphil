# Loop / Taunt Report

Bot port 1, 22 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| 2026-08-Mainline | 22/22 | 1.54 [0.00-3.43] | 102.55 [78.29-129.44] | 0.01 [0.00-0.01] | 0.24 [0.04-0.39] | 0.68 [0.00-2.14] | 2.27 [0-7] |

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
| 2026-08-Mainline | Game_20260829T214052.slp | 2.22 | 1.80 | 126.76 | 0.00 | 0.21 | 0.90 | 3 |
| 2026-08-Mainline | Game_20260829T214312.slp | 1.03 | 0.97 | 103.41 | 0.01 | 0.28 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T214420.slp | 1.95 | 2.05 | 105.93 | 0.01 | 0.32 | 0.51 | 7 |
| 2026-08-Mainline | Game_20260829T214624.slp | 0.97 | 0.00 | 117.66 | 0.01 | 0.39 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T214728.slp | 0.74 | 1.35 | 129.44 | 0.00 | 0.25 | 1.35 | 4 |
| 2026-08-Mainline | Game_20260829T214819.slp | 0.93 | 3.23 | 113.17 | 0.00 | 0.15 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T214921.slp | 1.75 | 3.43 | 106.24 | 0.00 | 0.19 | 1.71 | 4 |
| 2026-08-Mainline | Game_20260829T220638.slp | 1.77 | 1.70 | 106.26 | 0.00 | 0.23 | 1.13 | 3 |
| 2026-08-Mainline | Game_20260829T220831.slp | 1.40 | 0.71 | 87.94 | 0.00 | 0.22 | 2.14 | 4 |
| 2026-08-Mainline | Game_20260829T221001.slp | 1.25 | 0.80 | 101.00 | 0.01 | 0.30 | 1.60 | 7 |
| 2026-08-Mainline | Game_20260829T221232.slp | 1.46 | 1.37 | 102.27 | 0.01 | 0.36 | 0.00 | 0 |
| 2026-08-Mainline | Game_20260829T221405.slp | 1.97 | 1.53 | 113.88 | 0.00 | 0.24 | 1.02 | 3 |
| 2026-08-Mainline | Game_20260829T221610.slp | 0.99 | 3.02 | 99.72 | 0.01 | 0.19 | 1.01 | 4 |
| 2026-08-Mainline | Game_20260829T221716.slp | 1.99 | 2.01 | 96.25 | 0.00 | 0.21 | 2.01 | 4 |
| 2026-08-Mainline | Game_20260829T221922.slp | 0.41 | 0.00 | 98.43 | 0.00 | 0.04 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 15 |
| `GRAB_PUMMEL>GRAB_WAIT` | 6 |
| `DAMAGE_FLY_TOP>TECH_MISS_UP>GROUND_ATTACK_UP` | 1 |
| `DOWN_B_GROUND>KNEE_BEND>DOWN_B_STUN>DOWN_B_AIR` | 1 |
