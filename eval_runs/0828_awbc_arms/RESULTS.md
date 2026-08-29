# RESULTS — AWBC arms B1 / B2 / B3 (fox_gen_v1, 3-epoch resumes from ep10)

Read `PREREG.md` first. The rule below was fixed before any arm trained
and is applied here mechanically. Scored 2026-08-29 11:00 → 12:03 by
`scripts/awbc_score.sh` on a quiet machine (chain ended 10:59, nothing
else on the GPU). Raw: `score/score_table.txt`, `score/loops/report.md`,
`score/coach/`, per-run logs + replays `score/B{1,2,3}/`.

Protocol as pre-registered: 8 runs × 120 s, CPU dummy, delay 0, headless,
scalar T=0.5, buttons T=0.5. Every arm's decode banner reads
`buttons: 0.5` in all 8 run logs (the table's three "KNOB ASSERTION
FAILED" lines are a scorer bug — GOTCHA #104, `sed | grep -q` under
`pipefail` — verified false for all three arms; replays and logs intact).

## Arms as trained (from each checkpoint's `_config.json`)

| arm | awbc | reward | shuffle | val loss ep1 → ep3 |
|---|---|---|---|---|
| B1 | false | – | – | 5.7688 → 5.7328 |
| B2 | true | standard | false | 5.7152 → 5.6747 |
| B3 | true | standard | true | 5.8061 → 5.7694 |

No arm diverged; none disqualified. Val loss is recorded, not the verdict.

## Game duration (from run logs — truncation-proof)

| arm | n | mean | range | reaching 120 s cap |
|---|---|---|---|---|
| B1 | 8 | 123.2 s | 113.1–124.8 | 7/8 |
| B2 | 8 | 122.7 s | 109.0–124.7 | 7/8 |
| B3 | 8 | 124.5 s | 123.9–124.7 | 8/8 |

All arms play whole games. No collapse anywhere. scored/played 8/8 for
every arm on both scorers.

## Axis 1 — pathology (loop_report, mean [min–max])

| arm | d_up press/min | taunts/min | loops/min | max loop reps | held-action | pummel-loop episodes (per game) |
|---|---|---|---|---|---|---|
| B1 | 114.6 [101.8–131.2] | 2.04 [1.47–3.43] | 1.49 [0.49–3.43] | 8.4 [3–14] | 0.25 | 23 (2.9) |
| B2 | 126.9 [122.4–132.5] | 1.87 [0.49–2.94] | 1.80 [1.47–2.45] | 8.4 [7–11] | 0.27 | **27 (3.4)** |
| B3 | 119.0 [112.0–126.3] | 1.90 [0.98–3.92] | 1.35 [0.98–2.45] | 5.4 [3–9] | 0.25 | 18 (2.25) |

Pummel episodes = `GRAB_WAIT>GRAB_PUMMEL` + `GRAB_PUMMEL>GRAB_WAIT`
cycle episodes from a per-arm `loop_report` run (the pooled report does
not split the cycle table by arm).

Largest ratio on any pathology metric, any pair: **1.5×** (pummel
B2/B3, in the WRONG direction for B2). d_up: B2/B1 = 1.11×, ranges
overlap. loops/min: every range spans every other arm's mean.

## Axis 2 — competence (coach_report, per game)

| arm | deaths/game [range] | conversions | dropped/game | armed/min (read-only) |
|---|---|---|---|---|
| B1 | 2.38 [1–4] | 8/32 (25%) | 4.62 | 0.49 |
| B2 | 2.00 [1–3] | 9/42 (21%) | 5.25 | 0.61 |
| B3 | 1.50 [1–4] | 12/47 (26%) | 4.88 | 0.92 |

Deaths: B2 is within 1.5× of B1 (gate would pass) but the ranges are
identical. B3 has the fewest deaths (1.6× vs B1, overlapping ranges) —
that is the "path effect" read, and it is under 2×.

## Verdict by the pre-registered rule

- **SIGNAL** requires B2 to beat BOTH B1 and B3 by ≥2× with disjoint
  ranges on a pathology metric. B2 beats neither on anything; on d_up and
  pummel episodes it is the worst of the three. **Not SIGNAL.**
- **WEIGHT-DISTRIBUTION ARTIFACT** requires B2 and B3 to move together
  away from B1. They do not (B2 up on d_up/pummel, B3 down on
  deaths/pummel — both <2×, overlapping). **Not an artifact either.**
- **Confounded** check: B3-vs-B1 (path effect) is ≤1.6× everywhere and
  never disjoint. Nothing to confound.

**→ NULL.** Outcome-weighting at this strength (percentile-rule β,
p90/p10 weight ratio 7, clip [0.2, 5], standard reward, γ 0.997, H 300)
changes nothing observable in play after 3 epochs from ep10. Val loss
moved (B2 lowest) and play did not — the same decoupling seen across
ep1..ep10.

## What this does and does not say

- It does NOT say a value signal cannot help: Leg S measured +29 pts of
  selection headroom on the same checkpoint, and AWBC only *reweights*
  frames the master already chose — it cannot change which of the
  policy's own samples gets played. The headroom is in decode-time
  selection, which is what the D2 critic/selector targets.
- It does say the cheap retrain-side entry to offline RL is dry at
  default sharpness. PREREG's NULL branch allows one `--awbc-beta`
  (sharper) arm before abandoning; that arm is DEFERRED behind the
  critic shakedown, because the critic path is retrain-free (Bradley's
  standing direction: exhaust v1 retrain-free first) and Leg S already
  says selection is the bigger number.
- Human look: not requested — no arm won, so nothing gates a recipe
  change. fox_gen_v1 ep10 at buttons T=0.5 remains the default v1 decode.

## Caveats (declared in PREREG, still standing)

3 epochs is a budget; one 8-game batch per arm; CPU dummy measures
pathology, not fight-state; AWBC weights encode "what the master did
while winning", not "what beats a human".
