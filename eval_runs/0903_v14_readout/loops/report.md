# Loop / Taunt Report

Bot port 1, 8 replays, 2 groups.

Metrics that score the human-flagged pathologies (taunts, loops,
dithering) — none of which `coach_report` measures.

## By group

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| T05 | 4/4 | 2.03~1.97 [0.98-3.93] | 101.16~104.71 [93.69-112.39] | 0.00~0.01 [0.00-0.01] | 0.23~0.24 [0.15-0.29] | 0.49~0.49 [0.00-1.47] | 2.25~4 [0-5] |
| T10 | 4/4 | 2.05~2.46 [0.62-2.65] | 410.05~407.45 [399.45-427.14] | 0.00~0.00 [0.00-0.00] | 0.31~0.35 [0.20-0.41] | 0.16~0.00 [0.00-0.62] | 0.75~0 [0-3] |

mean~median [min-max]. Differences under 2x are unresolved; a range spanning
another arm's mean is no difference at all. A mean far from its median means
one degenerate game is dragging the group — trust the median for session dirs.

## Per game

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| T05 | r1.slp | 2.04 | 0.98 | 93.69 | 0.01 | 0.15 | 0.49 | 4 |
| T05 | r2.slp | 2.04 | 3.93 | 112.39 | 0.01 | 0.23 | 0.00 | 0 |
| T05 | r3.slp | 2.03 | 1.97 | 93.86 | 0.00 | 0.24 | 1.47 | 5 |
| T05 | r4.slp | 0.80 | 1.25 | 104.71 | 0.00 | 0.29 | 0.00 | 0 |
| T10 | r1.slp | 2.03 | 2.46 | 427.14 | 0.00 | 0.27 | 0.00 | 0 |
| T10 | r2.slp | 2.04 | 2.46 | 399.45 | 0.00 | 0.41 | 0.00 | 0 |
| T10 | r3.slp | 1.61 | 0.62 | 407.45 | 0.00 | 0.35 | 0.62 | 3 |
| T10 | r4.slp | 1.88 | 2.65 | 406.13 | 0.00 | 0.20 | 0.00 | 0 |

## Most repeated action cycles

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 2 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `SHIELD_START>SPOTDODGE` | 1 |
| `GRAB_PUMMEL>GRAB_WAIT` | 1 |
