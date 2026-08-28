# Loop / Taunt Report

Bot port 1, 12 replays, 1 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | n | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| 0828_session | 12 | 1.17 [0.00-3.13] | 455.54 [398.36-646.44] | 0.00 [0.00-0.00] | 0.34 [0.21-0.74] | 0.71 [0.00-2.25] | 2.92 [0-9] |

Mean [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| 0828_session | Game_20260828T053254.slp | 1.91 | 3.13 | 440.34 | 0.00 | 0.34 | 0.00 | 0 |
| 0828_session | Game_20260828T053456.slp | 1.22 | 1.64 | 439.32 | 0.00 | 0.28 | 0.00 | 0 |
| 0828_session | Game_20260828T053615.slp | 2.23 | 1.35 | 423.27 | 0.00 | 0.21 | 1.79 | 9 |
| 0828_session | Game_20260828T053835.slp | 1.21 | 0.83 | 448.55 | 0.00 | 0.31 | 0.83 | 3 |
| 0828_session | Game_20260828T053954.slp | 1.15 | 0.00 | 398.36 | 0.00 | 0.28 | 0.87 | 5 |
| 0828_session | Game_20260828T054110.slp | 1.84 | 2.17 | 435.33 | 0.00 | 0.32 | 0.00 | 0 |
| 0828_session | Game_20260828T054307.slp | 1.82 | 0.55 | 449.93 | 0.00 | 0.34 | 0.55 | 3 |
| 0828_session | Game_20260828T054502.slp | 0.94 | 0.00 | 471.09 | 0.00 | 0.29 | 0.00 | 0 |
| 0828_session | Game_20260828T054605.slp | 2.67 | 1.50 | 449.95 | 0.00 | 0.27 | 2.25 | 8 |
| 0828_session | Game_20260828T054853.slp | 1.56 | 1.28 | 446.00 | 0.00 | 0.34 | 0.64 | 3 |
| 0828_session | Game_20260828T055033.slp | 1.92 | 1.56 | 417.93 | 0.00 | 0.32 | 1.56 | 4 |
| 0828_session | Game_20260828T055235.slp | 0.09 | 0.00 | 646.44 | 0.00 | 0.74 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 12 |
| `GRAB_PUMMEL>GRAB_WAIT` | 4 |
| `SHIELD_START>GRAB` | 1 |
