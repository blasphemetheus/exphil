# D1 — noise floors: same checkpoint, same decode, different days

Script `scripts/noise_floor.sh`; raw in `raw.txt`. fox_gen_v1 ep10 vs CPU
dummy, FD, delay 0, 120 s, scored with `loop_report --bot-port 1` and
`coach_report --char fox`. Two identical-condition families:

- **F05** scalar T=0.5 + buttons T=0.5 — three batches: 08-28 sweep (5),
  08-28 bracket (4 readable of 8), 08-29 base (8). 17 games, 2 days.
- **F10** scalar T=0.5, buttons raw 1.0 — four batches: 08-26 live_ep10 (3),
  08-26 scalar_05 (5), 08-28 bracket base (8), 08-28 sweep btn1.0 (5).
  21 games, 3 days.

(`loop_report` merged the two `btn0.5` batches by basename; they are split
below from the per-game rows.)

## Batch means

| metric | F05 · 08-28 sweep (5) | F05 · 08-28 bracket (4) | F05 · 08-29 base (8) | F10 · 08-26 live (3) | F10 · 08-26 scalar (5) | F10 · 08-28 base (8) | F10 · 08-28 btn1.0 (5) |
|---|---|---|---|---|---|---|---|
| d_up press/min | 112.3 | 110.4 | 111.6 | 438.4 | 438.6 | 431.9 | 430.2 |
| taunts/min | 2.25 | 1.60 | 1.29 | 2.17 | 1.96 | 2.02 | 2.15 |
| held-action frac | 0.26 | 0.25 | 0.27 | 0.38 | 0.35 | 0.36 | 0.34 |
| loops/min | 1.57 | 1.47 | 1.53 | 1.22 | 2.06 | 2.26 | 2.35 |
| max loop reps | 10.8 | 5.8 | 7.4 | 5.7 | 8.6 | 10.8 | 13.4 |
| deaths/game | 2.00 | 2.00 | 2.25 | 3.00 | 2.00 | 1.12 | 1.60 |
| conversion % | 27 | 40 | 36 | 70 | 35 | 20 | 33 |
| dropped/game | 5.4 | 5.0 | 4.6 | 4.3 | 3.0 | 3.6 | 2.4 |
| neutral loss/game | 7.2 | 9.8 | 7.6 | 9.0 | 7.8 | 9.1 | 6.8 |
| armed/min | 0.59 | 0.73 | 0.24 | 0.51 | 0.69 | 0.37 | 0.10 |

## Noise floor = max/min ratio between batches of the SAME condition

| metric | F05 (3 batches) | F10 (4 batches) | **floor to use** | verdict |
|---|---|---|---|---|
| d_up press/min | 1.02× | 1.02× | **1.1×** | resolves 10% differences at n=5–8 |
| held-action frac | 1.08× | 1.12× | 1.2× | good |
| taunts/min | 1.74× | 1.11× | 1.8× | needs ≥2× |
| loops/min | 1.07× | 1.93× | **2.0×** | the 2× law is exactly right for this one |
| max loop reps | 1.86× | 2.35× | 2.5× | weak |
| neutral loss/game | 1.36× | 1.34× | 1.5× | needs ≥2× |
| dropped/game | 1.17× | 1.80× | 2.0× | needs ≥2× |
| deaths/game | 1.13× | 2.68× | **2.5–3×** | weak at n≤8; the "within 1.5×" gate is inside noise |
| conversion % | 1.48× | 3.50× | 3.5× | unusable at n≤8 |
| armed/min | 3.0× | 6.9× | 7× | unusable (as already known) |

## Re-reading this week's calls against the floors

- **AWBC arms NULL** — max ratio 1.5× on any metric: inside every floor except d_up (where B2 was *worse*). NULL stands; the 2× rule was, if anything, generous.
- **Live look B1 vs B2** — loops/min 0.22 vs 0.94 = 4.3× against a 2.0× floor: **resolved** (B1 loops less). deaths 3.29 vs 3.80 = 1.15×: unresolved. Bradley's "B1 probably better" has one resolved number behind it.
- **mode-of-16** — d_up 1.1 vs 111.6 (100×), frozen-input 0.74 vs 0.00: resolved a hundred times over; deaths 3.75 vs 2.25 = 1.67× is *inside* the deaths floor — the collapse verdict rests on frozen-input and durations, correctly, not on deaths.
- **buttons T dose-response** (0.5→112, 0.6→171–183, 0.7→248, 1.0→430): every step ≥1.5× against a 1.1× floor: resolved, as claimed.
- **btn0.6 vs btn0.5 deaths** (1.6 vs 2.0 on 08-28): 1.25× — unresolved; the "deaths climb as buttons cool" claim is only supported by the T=1.0 endpoint (1.1) vs 0.5 (2.0), 1.8×, still inside the F10 deaths floor. **Retract as unresolved.**

## Rules that follow

1. d_up/min and held-action are the only per-game metrics that resolve
   under 2× at n=8; use them for decode comparisons.
2. loops/min keeps the 2× law. deaths need ≥3× or n≥16; conversion % and
   armed/min are not comparison metrics at this n.
3. The "deaths within 1.5×" competence gate in the brackets is inside
   noise; replace with durations-to-cap (categorical, replication 8/8 vs
   0/8) plus frozen-input, which is what actually decided every collapse.
4. B2's TV distance needs its own floor: run situation_hist on these same
   batches (todo, D1b).
