# per_head_temp bracket — 2026-08-27 (COMPLETE)

3-arm decode bracket on fox_gen_v1 ep10: scalar 0.5 (baseline) vs
buttons 0.3 vs full per-head schedule (buttons 0.3 / main 0.5 / c 0.7 /
shoulder 0.5). 5 games/arm x 120s vs CPU, delay 0. Machine quiet:
staleness 0.0–0.3% across all 15 runs (longest stale run 2 frames) — all
runs valid.

## Scores (mean; range) — n=5/arm

| arm | armed/min | conv (agg) | deaths | passive |
|---|---|---|---|---|
| scalar_05 | 0.69 [0.0–1.47] | 7/20 (35%) | 2.0 | 2.2 [1–4] |
| buttons_03 | 0.20 [0.0–0.51] | 9/26 (35%) | 3.0 | 0.8 [0–3] |
| full_sched | 0.50 [0.0–1.47] | 5/23 (22%) | 3.6 | 0.0 [0–0] |

## Verdict

**No arm separates on the core metrics** (armed/min, conversion) — within-arm
spread spans 0→1.5 and equals between-arm spread, n=5 (the <2x law). The one
consistent signal is passivity: the per-head arms are less idle (scalar 2.2 →
buttons_03 0.8 → full_sched 0.0), but full_sched's zero-passivity comes with
the HIGHEST death rate (3.6/run) — trading idleness for aggression, not
competence. buttons_03's low armed/min (median 0.0 vs scalar_05's 0.98) is
suggestive but within noise.

**Reading:** per-head temperature is not a free win. It moves behavior
(idleness ↓ monotonically with sharper buttons) without moving conversions
(35% flat), and the sharper the buttons the less it commits — consistent with
G1's finding that buttons sit at 69% of uniform entropy (cooling them over-
sharpens the exploratory approaches that lead to conversions). To crown or
reject a per-head schedule: n≥8 run-level buckets (the standing bar) or a
human session (the g6 lesson: CPU rankings can invert).
