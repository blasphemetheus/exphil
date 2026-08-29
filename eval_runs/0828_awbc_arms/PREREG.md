# PREREG — AWBC arms B1 / B2 / B3 (value-model entry rung)

Written 2026-08-28 evening, BEFORE any arm ran. Launcher: `scripts/awbc_arms.sh`
(systemd unit `exphil-awbc-arms`, sequential, waits for Leg S to finish).

## Question

Does re-weighting BC frames by actual game outcome (advantage-weighted BC,
`Rewards.Standard` = stock + 0.01·damage + 5·win, within-replay
return-to-go) change how fox_gen_v1 plays — enough to justify building the
learned critic (D2 step 2)?

## Arms

All three: resume `fox_gen_v1_20260825_210355_epoch10.axon`, the v1 recipe
(gru, 512/512/256, window 60, batch 256, lr 1e-4 constant, dropout 0.1,
stream chunk 100, `replays/erickfm_ranked/FOX/extracted`, val 0.1),
`--seed 828` (identical val split), `--epochs 3`.

| arm | flags | role |
|---|---|---|
| B1 | (none) | plain continuation — controls for "3 more epochs" |
| B2 | `--awbc --awbc-reward standard` | the treatment |
| B3 | `--awbc --awbc-reward standard --awbc-shuffle` | same weight distribution, permuted across frames — controls for the weights' effect on the effective batch |

Sequential on one 5090. Epoch budget 3 (v1 ran ~40 min/epoch from cache;
AWBC forces the non-pipelined streaming path, expected slower). Checkpoints
`checkpoints/fox_gen_v1_B{1,2,3}_*`.

## Scoring (decided now)

Score each arm's FINAL checkpoint with the same live protocol, same day:
`eval_live_protocol.sh --runs 8 --seconds 120 --dummy cpu --temperature 0.5
-- --frame-delay 0 --headless --buttons-temperature 0.5`, then

1. `loop_report.exs --bot-port 1` — d_up press/min (dense, reproducible),
   `GRAB_WAIT>GRAB_PUMMEL` episodes/game, held-action frac.
2. `coach_report.exs --char fox --bot-port 1` — deaths/game, conversions,
   dropped/game. Read armed/min but never decide on it (7x day-to-day drift).
3. Game DURATION from the run logs (`Final stats: N frames`) and
   scored/played per arm — GOTCHA #102 cross-check.

Val loss is recorded but is NOT the verdict (it tracked nothing across
ep1..ep10 while play was flat).

## Decision rule

- **SIGNAL**: B2 beats BOTH B1 and B3 by >=2x on at least one pathology
  metric (pummel-loop episodes/game or d_up/min) with non-overlapping
  ranges, AND deaths/game within 1.5x of B1. -> build the critic.
- **WEIGHT-DISTRIBUTION ARTIFACT**: B2 and B3 both differ from B1 in the
  same direction. -> the effect is from non-uniform weights, not outcome
  information; do not build the critic on this evidence.
- **NULL**: no >=2x separation with disjoint ranges. -> outcome weighting
  at this strength does nothing observable; try `--awbc-beta` (stronger
  weights) once before abandoning, and re-check Leg S first (if selection
  headroom is ~0, no re-weighting can help either).
- Any arm whose val loss diverges (>1.5x its start) is DISQUALIFIED, not
  compared.

Bradley's live impression of the winning arm is recorded, and gates a
"default recipe" change (g6 lesson) but not the SIGNAL/NULL verdict.

## Known confounds, declared

- 3 epochs is a budget, not a convergence claim.
- One 8-game headless batch per arm; under-2x differences are unresolved.
- AWBC uses the corpus's own outcomes: the weights say "what the master did
  in games they were winning", not "what beats a human".

## Amendment 2026-08-28 21:55 (B1 done, B2 just started, no B2/B3 results seen)

- **Timing, corrected.** B1: epoch 1 = 2h06 (embedding + writing a NEW
  102 GB cache — `--seed 828` re-partitions files into chunks, so the v1
  cache keys did not match), epochs 2-3 = 37 min each from cache. B1 val
  loss 5.7688 -> 5.7502 -> 5.7328.
- **B2/B3 take the non-pipelined path with NO embedding cache** (log shows
  neither "Saving embeddings" nor "Cache hit"; it parses and embeds every
  chunk every epoch). Expect ~2 h/epoch, ~6 h/arm. Disk-safe (no cache
  writes; root is at 12 GB free).
- **Confound, declared before results:** the non-pipelined path reports
  322,407 batches/epoch vs B1's 286,622 on the same seed — the two paths
  do not construct identical batch sets. So B1 vs B2 is NOT a pure
  loss-weighting contrast. The B2 vs B3 contrast IS pure (same path, same
  weights, permuted). Decision rule stands as written (SIGNAL requires B2
  to beat BOTH B1 and B3); additionally, **B3 vs B1 is now read as the
  path effect**, and if B3 differs from B1 by more than B2 differs from
  B3, the path — not outcome weighting — dominates and the verdict is
  "confounded", not SIGNAL. Cheap follow-up if it matters: a B1' on the
  non-pipelined path (`--awbc` with all weights forced to 1) to remove
  the path difference.
