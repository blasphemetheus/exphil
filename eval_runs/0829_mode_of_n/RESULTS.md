# RESULTS — mode-of-N live bracket (fox_gen_v1 ep10, 2026-08-29 14:28 → 15:02)

Pre-registered read: `scripts/mode_of_n_bracket.sh` header. base = scalar
T=0.5, buttons T=0.5, delay 0, CPU dummy, 8 × 120 s; `mode16` = same +
`--mode-of-n 16`. Quiet machine (nothing else on the GPU). base's "KNOB
ASSERTION FAILED" line is a banner-quoting mismatch (`Mode-of-N: "off"`),
verified false; 8/8 games each arm, both scored 8/8.

## Axis 3 — duration (from run logs) and machine health

| arm | mean game | range | reaching 120 s cap | staleness | inferences/frame |
|---|---|---|---|---|---|
| base | 124.5 s | 123.9–124.8 | **8/8** | 0.1–0.8 %, longest run 1 | 13.0 |
| mode16 | **50.5 s** | **19.5–109.6** | **0/8** | 0.3–1.4 %, longest run 2 | 9.6 |

Staleness stayed low — the machine kept up. The games ended because the
bot lost all its stocks.

## Axis 1 — loop_report

| arm | d_up press/min | frozen-input frac | held-action frac | loops/min | taunts/min |
|---|---|---|---|---|---|
| base | 111.6 [99.9–120.8] | 0.00 [0.00–0.01] | 0.27 | 1.53 | 1.29 |
| mode16 | 1.1 [0.0–3.6] | **0.74 [0.63–0.85]** | 0.51 | 0.51 | 0.22 |

## Axis 2 — coach_report

| arm | deaths/game [range] | conversions | armed/min |
|---|---|---|---|
| base | 2.25 [1–4] | 14/39 | 0.24 |
| mode16 | **3.75 [3–4]** | 1/21 | **0.00** |

## Verdict

**DISQUALIFIED — argmax lock.** The 0828 bracket's guard (frozen-input
> 0.20 disqualifies) fires at 0.74, in the same regime as
`--deterministic-buttons` (0.86) and full argmax (0.98). Every mode16 game
loses 3–4 stocks in under a minute with zero armed approaches. The
"pathology" metrics collapsing (d_up 112 → 1/min) is the bot standing
still, not the bot choosing better.

The deaths gate (within 1.5× of base) also fails: 3.75 vs 2.25 = 1.67×,
ranges [3–4] vs [1–4].

## Mechanism, and what it says about the offline instrument

Majority vote over 16 draws at T=0.5 returns the modal joint action
almost every frame — it IS argmax with extra steps. Offline this scored
+8 pts pass@1 because **pass@1 / master-match is a mode-seeking metric**:
it rewards any decode that concentrates on the most likely action, and
argmax-like decodes always win it. Live play needs the tail — sampling
is the only noise that escapes an absorber (EXPOSURE_BIAS, 2026-07-27),
and a frozen policy has no tail.

So the Leg S "selection headroom" (+29 pts) is real as a *statement
about the distribution* (the master's action is in the top-16 43% of the
time) but **cannot be cashed by any decode that picks the mode**, and
pass@1-style offline scores cannot rank decodes for live play on their
own. A follow-up run (Leg S at T→0 on the same frames) is queued to
make this precise: if argmax scores even higher pass@1 than mode-of-16,
the offline metric is formally rejected as a decode ranker.

**Law (new): an offline decode metric must be paired with a live
frozen-input / duration check before any decode is ranked by it.** The
0828 bracket (argmax buttons) and this one (mode-of-N) both failed the
same way through different doors.

## What survives

- `--mode-of-n` stays in the CLI as a knob (default off). Small N (2–4)
  is untested and would be a much weaker mode pull; not queued — the
  metric that motivated it is the problem.
- The critic direction is not dead, but a learned selector must be
  scored *live* (frozen-input, durations, deaths) before any offline
  gap-recovered number is believed; the design doc's STRONG rule gets a
  live gate added.
- Default v1 live decode remains buttons T=0.5 (base here replicated:
  d_up 111.6 vs 110.4 on 08-28, 8/8 to the cap).

## Follow-up (15:30) — the offline metric is formally rejected as a decode ranker

Leg S at T=0.05 (≈ argmax) on the same frames: joint pass@1 **24.4%**,
above mode-of-16's 22.9 and the deploy decode's 14.5. The decode that
dies fastest live scores highest offline. `eval_runs/0828_legS/RESULTS.md`
§Near-argmax control.
