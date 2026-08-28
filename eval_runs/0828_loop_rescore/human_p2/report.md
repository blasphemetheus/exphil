# Loop / Taunt Report

Bot port 2, 12 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | n | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| 0828_session | 12 | 0.00 [0.00-0.00] | 0.00 [0.00-0.00] | 0.45 [0.34-1.00] | 0.12 [0.00-1.00] | 9.64 [0.00-12.44] | 21.00 [0-57] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| 0828_session | Game_20260828T053254.slp | 1.91 | 0.00 | 0.00 | 0.36 | 0.00 | 11.49 | 18 |
| 0828_session | Game_20260828T053456.slp | 1.22 | 0.00 | 0.00 | 0.42 | 0.01 | 9.85 | 16 |
| 0828_session | Game_20260828T053615.slp | 2.23 | 0.00 | 0.00 | 0.39 | 0.00 | 9.42 | 18 |
| 0828_session | Game_20260828T053835.slp | 1.21 | 0.00 | 0.00 | 0.40 | 0.05 | 12.44 | 24 |
| 0828_session | Game_20260828T053954.slp | 1.15 | 0.00 | 0.00 | 0.34 | 0.03 | 9.53 | 13 |
| 0828_session | Game_20260828T054110.slp | 1.84 | 0.00 | 0.00 | 0.47 | 0.10 | 10.87 | 22 |
| 0828_session | Game_20260828T054307.slp | 1.82 | 0.00 | 0.00 | 0.38 | 0.01 | 9.32 | 57 |
| 0828_session | Game_20260828T054502.slp | 0.94 | 0.00 | 0.00 | 0.46 | 0.02 | 11.67 | 11 |
| 0828_session | Game_20260828T054605.slp | 2.67 | 0.00 | 0.00 | 0.39 | 0.07 | 12.35 | 26 |
| 0828_session | Game_20260828T054853.slp | 1.56 | 0.00 | 0.00 | 0.40 | 0.07 | 8.32 | 23 |
| 0828_session | Game_20260828T055033.slp | 1.92 | 0.00 | 0.00 | 0.40 | 0.04 | 10.41 | 24 |
| 0828_session | Game_20260828T055235.slp | 0.09 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `TURNING>DASHING` | 100 |
| `DASHING>TURNING` | 70 |
| `GRABBED>GRAB_PUMMELED` | 21 |
| `LANDING>DAMAGE_HIGH_2` | 1 |
| `DAMAGE_HIGH_2>LANDING>CROUCHING` | 1 |
| `SHIELD_STUN>SHIELD` | 1 |
| `GRABBED_WAIT_HIGH>PUMMELED_HIGH` | 1 |
