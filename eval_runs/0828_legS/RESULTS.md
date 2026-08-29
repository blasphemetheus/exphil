# 0828_legS — HEADROOM TRIAD Leg S, first real run

`scripts/interp_passk.exs`, fox_gen_v1 ep10, `replays/erickfm_ranked/FOX/extracted`
(20 files, port-normalized, pinned `--port 1`; this is the TRAINING corpus —
the config had `val_split 0.1` with no seed, so held-out files are
unrecoverable). 199,255 frames -> 45,007 decision frames (22.6%) -> 2,000
scored, n=16, temperature 0.5 (scalar; the agent's deploy decode), stick
tolerance 1/16.

## Run 1 was an instrument bug (kept: `ep10_BUGGED_run1.{log,md}`)

pass@k IDENTICAL to pass@1 at every k for every head (joint 0.4 flat).
Two defects in how the policy was queried:

1. `Agent.get_controller/3` enforces one decision per game frame (the
   08-01 netplay fix — a repeated frame re-sends the cached action), so 16
   calls on the same frame returned ONE sample and 15 copies.
2. The policy is a 60-frame-window GRU and was fed isolated frames strided
   across the corpus: its window held frames from unrelated games.

Fix (`scripts/interp_passk.exs`): per decision frame, reset the agent,
replay the preceding 60 frames OF THE SAME GAME through the normal path,
one forward via `get_action_with_confidence` (raw head logits), draw n
samples from those logits in-script with the agent's own Bernoulli /
Gumbel-max decode; `--seed`. Added a degenerate-samples self-check that
withholds the verdict when every frame scores all-or-nothing. Run 2
crashed on a key name (`ep10_crash_run2.log`); run 3 is the result below.
~28 min for 122k inferences.

## Result (run 3)

| head | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | headroom |
|---|---|---|---|---|---|---|
| joint | 14.5 | 21.6 | 29.3 | 36.9 | 43.7 | **+29.2** |
| buttons | 42.0 | 59.2 | 74.7 | 86.4 | 93.1 | **+51.1** |
| main | 38.0 | 44.8 | 50.6 | 56.1 | 61.7 | **+23.6** |
| c | 91.5 | 92.7 | 93.4 | 93.9 | 94.3 | +2.7 |
| shoulder | 86.0 | 89.7 | 92.1 | 93.9 | 95.6 | +9.6 |

Script verdict: **SELECTION headroom is large (+29.2 pts joint).**

Reading:
- The curve is still climbing at k=16 on joint, buttons and main stick —
  the distribution is broad and a selector has a lot to work with (the
  doc's "climbing at 16" case).
- Buttons carry the most: one sample matches the master's pressed set 42%
  of the time, 16 samples contain it 93% of the time. This is the
  quantitative form of "the model contains the behavior and lacks a
  selection rule", and of Bradley's live impression (the good option
  shows up sometimes; the punishable one is thrown out more often).
- C-stick and shoulder are near-saturated at k=1 (mostly "not touched" on
  decision frames) — they are not where the headroom is.

## What this does NOT yet establish

- **Calibration not done.** The doc's mandatory check — pass@1 across
  ep1..ep10 must be roughly flat, since those epochs demonstrably played
  the same — has not run (the GPU went to the AWBC arms immediately after
  this run). Until it does, treat +29 as "large and real-looking", not as
  a ranked number. Run on ep1, ep5, ep10 first.
- **Second corpus not done.** `replays/fox_il_v1` (auto-port) must agree.
- Scored on the TRAINING corpus (unavoidable, see above): pass@1 may be
  inflated by memorization; the GAP is the quantity of interest and
  memorization would shrink it, not grow it.
- Upper bound: assumes a perfect selector; open-loop; the master's exact
  press is not the only correct one.

## Consequence for the fork

If calibration holds, the ceiling is SELECTION, not capacity or data:
the critic / Best-of-N (direction B) is the highest-return investment
and fox_gen_v2 (direction D) is not yet justified. The AWBC arms
(`eval_runs/0828_awbc_arms`) running now are the cheap first test of
whether outcome information can supply that selection rule.

## Calibration (2026-08-29, unit `legS-critic`, seed 829, same 20 files / 2000 decision frames / n=16 / T=0.5)

| epoch | joint pass@1 | pass@16 | headroom | buttons @1 → @16 | main @1 → @16 |
|---|---|---|---|---|---|
| 1 | 10.4 | 34.3 | +23.8 | 38.5 → 92.5 | 30.2 → 51.9 |
| 5 | 13.3 | 41.0 | +27.7 | 41.3 → 93.5 | 36.0 → 59.5 |
| 10 | 14.5 | 43.6 | +29.1 | 42.4 → 93.7 | 38.2 → 60.6 |
| 10 (run 3, other seed) | 14.5 | 43.7 | +29.2 | 42.0 → 93.1 | 38.0 → 61.7 |

**Verdict: the instrument passes.**
- Not flat-by-construction: pass@1 rises monotonically with training
  (10.4 → 13.3 → 14.5), so the probe does see the policy improve.
- The selection headroom is present and stable at every epoch (24–29
  pts) and is ~2× everything training bought between ep1 and ep10 (4.1
  pts). Selection, not knowledge, remains the biggest number.
- Buttons pass@16 is pinned at 92.5–93.7 from epoch 1 onward: the button
  head already CONTAINS the master's press; training mostly moves `main`
  stick pass@1 (30 → 38).
- Seed replication: ep10 at seed 829 vs run 3's seed agree to 0.1 pt on
  every head — the +29 is not a sampling-seed artifact.

Files: `ep1.md`, `ep5.md`, `ep10.md`; logs `logs/legS_cal_ep*.log`.
Second corpus (`fox_il_v1`, auto-port) → `ep10_fox_il_v1.md` (next stage).
