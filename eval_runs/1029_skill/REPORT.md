# Corpus skill scoring — master-like Fox selection (2026-10-29)

Why: the master-only line (7,853 master-master games) plateaued at CHANGE fam ~53 by epoch 3 and width did
not move it (eval_runs/1028_w768: "width is not the lever"), so the open lever is more master-grade data.
Only the master-master files carry a tier; the other 56k (altf4 public dataset, 2018–2020 tournament/
netplay names) are anonymous. So: score every Fox game from PLAY (scripts/corpus_skill_score.exs ->
scores.jsonl, 29,436 Fox-game rows from 64,117 files in 451 s, 10 workers) and use the master games as
labels (scripts/skill_select.js).

## Master vs untiered, per feature (Fox games >= 1.5 min; 8,676 master rows, 19,843 untiered)

| feature | master mean | untiered mean | AUC master>untiered |
|---|---|---|---|
| l_cancel_rate | 0.911 | 0.849 | **0.679** |
| dashes_per_min | 49.7 | 40.0 | **0.665** |
| apm (presses+dashes+jumps+WDs) | 228.9 | 214.0 | 0.609 |
| short_hop_share | 0.408 | 0.381 | 0.550 |
| tech_rate | 0.488 | 0.464 | 0.524 |
| offstage_return_rate | 0.875 | 0.869 | 0.523 |
| damage_dealt_per_min | 134.3 | 133.2 | 0.510 |
| deaths_per_min | 1.19 | 1.19 | 0.499 |
| sd_per_min | 0.457 | 0.432 | 0.481 |
| wavedashes_per_min | 4.71 | 4.98 | 0.465 |
| kills_per_min | 1.19 | 1.25 | 0.466 |
| neutral_input_share | 0.248 | 0.271 | **0.384** (masters idle less) |

Reading: what separates the tiered masters from the public corpus is EXECUTION density (L-cancel rate,
dash density, APM, less idle), not outcomes — damage, kills, deaths and SDs do not separate at all.
The public corpus is mostly also strong players (tournament sets), so this is a weak-label problem.

Combined score = z-scores weighted by (AUC − 0.5): AUC 0.736 master vs untiered.
Threshold = master median (0.210) -> **3,986 of 19,843 untiered Fox games (20.1 %) score like masters**.
Their profile: L-cancel 0.928 (master 0.911), APM 254 (229), dashes 54/min (50), damage 147/min (134),
SD 0.34/min (master 0.46), idle share 0.246 (0.249). +10,652 minutes on the master corpus's 24,505 (+43 %).

## The arm (unit exphil-mlike-1029, scripts/session_1029_masterlike.sh)

replays/erickfm_masterlike = 7,837 master games (the master run's 16 held-out val games EXCLUDED) +
3,986 master-like untiered = 11,823 files. Recipe eval_runs/1029_masterlike/recipe.sh = the master recipe
(512,512,256 Mamba, seed 905, LR 2e-4/2e-4/1e-4), 3 epochs (~3.5–4 h each at +43 % data), light readout.
Comparable metric = agreement on the MASTER run's own 16 held-out games (--split master_ep1/split.json,
8 games) — the same frames as every master/w768 readout. The run's own val_loss is on mixed games: not comparable.

PASS: ep3 CHANGE fam > 53.4 / joint > 25.5 and DECISION fam > 64.3 by more than the ~2-point seed floor,
league Elo near/above master_ep7_plain. FAIL = execution-stat selection does not buy decision quality;
close "data selection by play features" and the remaining data lever is Bradley's own / newer master data.
