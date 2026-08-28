# buttons-temperature sweep — 2026-08-28 (COMPLETE, null)

Sweep button temperature (1.0/0.7/0.6/0.5, sticks at scalar 0.5) on
fox_gen_v1 ep10, 4 arms x 5 games vs CPU, delay 0. All 20 runs valid
(staleness <=0.7%). Run after the human session flagged taunts + laser/grab
dithering at buttons=1.0.

## Scores (n=5/arm)

| arm (buttons T) | armed/min [range] | passive | deaths | conv (agg) |
|---|---|---|---|---|
| 1.0 (baseline) | 0.10 [0–0.49] | 3.8 | 1.6 | 3/9 (33%) |
| 0.7 | 0.39 [0–0.98] | 1.8 | 1.8 | 5/16 (31%) |
| 0.6 | 0.29 [0–0.98] | 3.0 | 1.4 | 4/20 (20%) |
| 0.5 | 0.59 [0–1.47] | 2.2 | 2.0 | 6/22 (27%) |

## Verdict: the instrument cannot resolve buttons temperature

**The baseline is not reproducible across days.** buttons=1.0 scored armed/min
0.10 here vs 0.69 in the 0827 bracket (same config, ~7x apart; 4/5 zero-armed
games here vs 3/5 >0.9 there). Armed approaches are bursty, so a 5-game mean
is one or two lucky games wearing the whole number. Within-arm ranges span the
full 0–1.47, so no between-arm difference clears the <2x law.

**The sweep also measures the wrong thing.** The human complaint was taunts +
same-action loops; `coach_report` scores armed/min / passive / deaths, none of
which count taunts or streaks directly.

## Reading

Buttons-temperature tuning is a human-in-the-loop question, not a CPU-sweep
question at n=5. Options: (a) live play-vs with `--buttons-temperature 0.6`;
(b) a dedicated loop/taunt metric (count taunts + same-action streaks from
.slp) if this is to stay CPU-scored.
