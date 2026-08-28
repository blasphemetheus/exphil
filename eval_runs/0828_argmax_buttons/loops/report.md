# Loop / Taunt Report

Bot port 1, 40 replays, 4 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| base | 8/8 | 2.02 [0.98-4.41] | 431.89 [415.84-454.10] | 0.00 [0.00-0.00] | 0.36 [0.27-0.39] | 2.26 [0.98-2.94] | 10.75 [4-17] |
| btn0.5 | 4/8 | 1.59 [0.49-2.95] | 110.43 [105.74-114.65] | 0.00 [0.00-0.01] | 0.25 [0.16-0.34] | 1.47 [0.49-2.94] | 5.75 [3-9] |
| btn0.6 | 8/8 | 2.45 [1.47-4.90] | 183.39 [163.50-196.66] | 0.00 [0.00-0.00] | 0.30 [0.28-0.35] | 1.53 [0.49-1.96] | 8.50 [5-13] |
| detbtn | 1/8 | 0.00 [0.00-0.00] | 0.00 [0.00-0.00] | 0.86 [0.86-0.86] | 0.60 [0.60-0.60] | 3.43 [3.43-3.43] | 6.00 [6-6] |

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
| btn0.5 | r1.slp | 2.04 | 1.47 | 108.88 | 0.00 | 0.34 | 2.94 | 9 |
| btn0.5 | r2.slp | 2.04 | 0.49 | 105.74 | 0.01 | 0.22 | 0.98 | 4 |
| btn0.5 | r3.slp | 2.05 | 1.47 | 112.44 | 0.01 | 0.16 | 0.49 | 3 |
| btn0.5 | r8.slp | 2.03 | 2.95 | 114.65 | 0.00 | 0.27 | 1.48 | 7 |
| btn0.6 | r1.slp | 2.04 | 2.94 | 187.36 | 0.00 | 0.32 | 1.47 | 7 |
| btn0.6 | r2.slp | 2.04 | 1.96 | 196.66 | 0.00 | 0.30 | 1.96 | 13 |
| btn0.6 | r3.slp | 2.04 | 1.96 | 182.67 | 0.00 | 0.28 | 1.96 | 8 |
| btn0.6 | r4.slp | 2.04 | 1.47 | 194.53 | 0.00 | 0.35 | 0.98 | 7 |
| btn0.6 | r5.slp | 2.04 | 1.96 | 163.50 | 0.00 | 0.28 | 1.96 | 11 |
| btn0.6 | r6.slp | 2.04 | 4.90 | 176.37 | 0.00 | 0.29 | 1.47 | 5 |
| btn0.6 | r7.slp | 2.04 | 2.45 | 187.90 | 0.00 | 0.30 | 0.49 | 8 |
| btn0.6 | r8.slp | 2.04 | 1.96 | 178.14 | 0.00 | 0.28 | 1.96 | 9 |
| detbtn | r5.slp | 1.17 | 0.00 | 0.00 | 0.86 | 0.60 | 3.43 | 6 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 47 |
| `GRAB_PUMMEL>GRAB_WAIT` | 22 |
| `DAMAGE_NEUTRAL_2>LANDING` | 2 |
| `CROUCHING>DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `LANDING>CROUCHING>DAMAGE_HIGH_2` | 1 |
| `LANDING>CROUCHING>DAMAGE_NEUTRAL_1` | 1 |
| `LANDING>CROUCHING>DAMAGE_NEUTRAL_2` | 1 |
| `SHIELD_REFLECT>GRAB` | 1 |
| `SHIELD_REFLECT>SPOTDODGE` | 1 |
| `SHIELD_REFLECT>SPOTDODGE>CROUCH_START` | 1 |
