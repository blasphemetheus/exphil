# EVAL_DIRECTIONS — getting more out of the games we already have

**Started 2026-08-29 (late).** Tracking document. Everything here runs on
replays already on disk — no Dolphin, no GPU — and is scored against the
expert corpus as the baseline. Update the status column as instruments
land; link each one to its script and its first RESULTS.

## Why this exists

Today (08-29) three selectors failed in an informative way, and the
reason was the *instrument*: the offline metric (pass@1 / master-match)
is mode-seeking and the live metrics are per-game averages. Bradley's
complaints are all **conditional** ("when it grabs…", "when I'm on the
ledge…") and none of the current numbers are. The one number that moved
with the 3-epoch continuations (dropped conversions 2.6 → 0.7–1.3) was
the only situational one we had. See `eval_runs/0829_mode_of_n/RESULTS.md`
and `eval_runs/0829_livelook_awbc/IMPRESSIONS.md`.

## The bank

| set | games | what it is |
|---|---|---|
| `replays/erickfm_ranked/FOX/extracted` | 7,911 | v1's training corpus. **CORRECTED 08-30 (E1):** port 1 is NOT always the fox — fox is p1-only 44.2%, p2-only 43.3%, ditto 12.5%; v1 imitated port 1 regardless, so ~43% of its demonstrations are the non-fox opponent (`eval_runs/0830_corpus_mix/RESULTS.md`) |
| `replays/fox_il_v1` | 4,465 | older Fox corpus, off-distribution for v1 |
| `eval_runs/0828_*`, `0829_*` brackets | ~120 | bot vs CPU, every decode/checkpoint tagged |
| `eval_runs/0828_session`, `0828_livelook_btn05`, `0829_livelook_awbc_B{1,2,3}` | ~40 | bot vs Bradley |

## Instruments already in hand

| instrument | measures | limits |
|---|---|---|
| `loop_report` / LoopStats | d_up presses, taunts, frozen/held input, repeated action cycles | per-game averages; loops/min is not a pathology score by itself |
| `coach_report` / FailureScan | armed approaches, conversions, dropped punishes, deaths, passivity, neutral losses | armed/min drifts 7× day to day |
| `interp_passk` (Leg S) | pass@k − pass@1 on decision frames | mode-seeking; never rank decodes on it (L9) |
| `sd_scan` | self-destructs by side, walk-offs | detector's 90 f window counts late recovery deaths as SDs (symmetric) |
| `critic_*` / `interp_bestofn` | offline Best-of-N vs sampling / mode-of-N / oracle | same mode-seeking metric |
| run logs | game duration, inferences/frame, staleness | the truncation-proof cross-check |

## The directions

Status: `todo` · `building` · `ran` (has a RESULTS) · `adopted` (on the standard rung) · `dropped`.

### A. Situational play — "what does it do when X?"

| id | instrument | question it answers | status | script / results |
|---|---|---|---|---|
| **A1** | **Situation → next-action histograms, bot vs expert** | In each named situation (opp shielding in range, opp offstage, bot offstage, bot >100%, opp on ledge, fresh respawn, opp in hitstun low/mid/high), what does the bot do next vs what the expert does? P(grab \| opp shield in range) etc. | **ran** | `scripts/situation_hist.exs` → `eval_runs/0829_situation_hist/README.md` (dash 31.6% vs ~1%; throw 86% vs 9–36%; off-stage airdodge) |
| A2 | Edgeguard / recovery scorecards | Opp offstage: go out / ledge / shine / laser / wait, and conversion of each. Bot offstage: route chosen, success rate, how it got there. | **ran** | `scripts/edge_scorecard.exs` → `eval_runs/0829_edge_scorecard/RESULTS.md`; summary in `eval_runs/0829_evaldir2/README.md` |
| A3 | Grab follow-ups | After a grab: throw direction, pummels before throw, regrab, release — vs expert. Decides mask-vs-data for the pummel loop. | answered by A1 §2 (bot never throws) | |

### B. Expert comparison, situation-matched

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| B1 | Action-family match | Did the bot pick the same *category* (aerial/grab/shield/movement/special) as the expert in that state? Softer pass@1. | building | `scripts/action_family_match.exs` (written 08-30, first run pending) |
| **B2** | **Distribution distance per situation** | KL / total-variation between the bot's next-action histogram and the expert's, per situation. Flags narrowing decodes (argmax, mode-of-N) offline — the pass@1 replacement as a decode ranker. | **ran — adopt** | same script; mean TV orders B1/B2/B3 (0.46–0.47) < ep10 (0.53–0.56) < mode-16 (0.72), matching the human read and the live rung |
| B3 | Entropy per situation | Policy-output entropy by situation from captured logits; low entropy where the expert is diverse = a loop waiting to happen. | **ran** | `scripts/interp_entropy_by_situation.exs` → `eval_runs/0829_entropy/RESULTS.md` (grab: 4.1 bits, expert decisive) |

### C. Temporal structure — the "scrappy / harder to hit" axis

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| C1 | Neutral-exchange outcomes | Segment games into exchanges; who won each and how (first hit / trade / whiff punish). Dense "harder to hit" number. | **ran** | `scripts/neutral_exchange.exs` → `eval_runs/0829_neutral_exchange/RESULTS.md` (B1 wins 20% of exchanges vs Bradley; needs its floor) |
| C2 | Punish quality | Damage per opening, combo length, % openings ending in a kill, vs expert. | building | `scripts/punish_quality.exs` (written 08-30, first run pending) |
| C3 | Reaction latency | Time-to-action after opponent lands / grabs ledge / bot lands, vs expert. Dithering. | building | `scripts/reaction_latency.exs` (written 08-30, first run pending) |
| C4 | Stock-loss forensics | Every death classified: unforced walk-off / failed recovery / edgeguarded / combo'd / neutral kill. Extends `sd_scan`. | **ran** | `scripts/death_classifier.exs` → `eval_runs/0829_death_classifier/RESULTS.md` (bot dies at 52–71% vs expert 109%) |

### D. Reliability of the instruments

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| **D1** | **Noise floors** | Same checkpoint, different days/batches: the natural spread of every metric. Which numbers can ever resolve a real difference? | **ran** | `scripts/noise_floor.sh` → `eval_runs/0829_noise_floor/RESULTS.md`: d_up 1.1×, held 1.2×, loops 2.0×, deaths 2.5–3×, conv% 3.5×, armed 7×. D1b ran: TV floor 0.05 on the mean, 0.1 per situation (`eval_runs/0829_noise_floor/tv_floor.md`) |
| D2 | Human-vs-CPU transfer table | For each metric, CPU-bracket value vs Bradley-session value for the same checkpoint. Which metrics the CPU rung can stand in for. | building | `scripts/d2_transfer.sh` (runner; pair map needs the real CPU dirs checked) |
| D3 | Decode-vs-model sensitivity | From banked temperature sweeps: which metrics are purely decode-driven (press rates) vs model-driven. | building | `scripts/decode_sensitivity.exs` over `eval_runs/0828_loop_rescore/*/report.json` |

### E. Data-side audits

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| E1 | Corpus mix by frame | Characters / stages / players by frame; concentration (one player's style dominating the mode). | **ran (by file)** | `eval_runs/0830_corpus_mix/RESULTS.md` — v1 imitated port 1 blindly; 56.7% fox, 43% opponent characters. By-frame + player-concentration refinement still open |
| E2 | Rare-event coverage | How many expert examples of the exact situations the bot fails in. Hundreds = selection problem; dozens = data. | **ran — not data quantity** | `scripts/rare_event_coverage.exs` → `eval_runs/0829_rare_events/RESULTS.md` (thousands of labels/epoch for every missing behaviour) |
| E3 | Expert pathology baselines | Pummels, taunts, standing lasers per game in *expert* play, so "too much" has a denominator. | building | `scripts/expert_pathology.exs` (written 08-30, first run pending) |
| E4 | Left walk-off poison | Is the corpus poisoned with repeated one-sided SDs? | **ran — falsified** | `scripts/sd_scan.exs`, `eval_runs/0829_sd_scan/RESULTS.md` |

## Order of work

1. **A1** — the instrument that turns Bradley's sentences into rows; gives B and C their denominators.
2. **B2** — cheap once A1 exists; would have rejected mode-of-N without a live run.
3. **D1** — from batches already on disk; tells us which of this week's "<2× unresolved" reads were resolvable.
Then re-read B1-the-model, ep10, B2 through A1/B2/C.

## Standing rules for anything added here

- Expert corpus is the denominator; report the bot and the expert side by side, same detector.
- Report scored/played (L2) and cross-check durations from run logs (L3).
- Never rank a decode on a mode-seeking metric alone (L9); pair with frozen-input + duration + deaths.
- **Decode rules are instruments, not fixes** (Bradley, 08-29): a behaviour gap found here becomes a TRAINING question (is it in the corpus / does the recipe lose it / does the decode drop it) — never a hand-written mask.
- Floors (D1): d_up/min 1.1×, held-action 1.2×, loops/min 2×, dropped 2×, deaths 2.5–3× at n≤8, conversion % and armed/min unusable. The "deaths within 1.5×" gate is inside noise — use durations-to-cap + frozen-input for collapse.

## Log

- 2026-08-29 22:55 — document created; E4 recorded as ran/falsified; A1 started.
- 2026-08-29 23:15 — A1 + B2 ran (`eval_runs/0829_situation_hist/README.md`).
  Findings: the bot has NO ground movement (dash 0.1% of frames vs 9–12%
  expert; no dash-dance; grab/spotdodge/special substitute); from a grab it
  throws 9–36% vs expert 86% (the pummel loop is "never throws"); off-stage
  it airdodges where the expert up-Bs. B2's TV distance orders the sets
  the way Bradley did and flags mode-of-16 as worst — adopted as the
  offline decode ranker (with the L9 live gate). D1 launched.
- 2026-08-29 23:40 — D1 ran (`eval_runs/0829_noise_floor/RESULTS.md`): d_up
  resolves 10% differences; loops/min needs the 2× law exactly; deaths
  need 3× at n≤8; conversion % and armed/min are not comparison metrics.
  Re-read: AWBC NULL stands; B1-vs-B2 loops (4.3×) resolved; "deaths climb
  as buttons cool" retracted as unresolved (1.8× inside a 2.7× floor).
  Next: D1b (TV-distance floor on the same batches), then A2 / C1.
- 2026-08-29 23:32 — **Dash probe** (`scripts/interp_dash_probe.exs`,
  `eval_runs/0829_dash_probe/RESULTS.md`): at the exact expert states that
  precede a dash, ep10's main-stick head puts 74.5% mass on full tilt and
  samples a flick 74.3% of the time (expert 69.3%); at standing-hold states
  0.7% (expert 0%). **Not a decode artifact.** The head is calibrated; the
  bot never dashes because it never reaches dash-initiation states in its
  own play (WAIT 0.4% of its frames). Closed-loop state drift — a training
  question (exposure bias / which states the recipe teaches it to be in),
  per the no-decode-rules rule.
- 2026-08-29 23:51 — batch 2 ran (D1b, E2, A2, C1, C4, B3); summary in
  `eval_runs/0829_evaldir2/README.md`. E2: not data quantity. A2/C4: wrong
  recovery routes at half the expert's success; dies at half the expert's
  percent. C1: B1 wins 20% of exchanges vs Bradley (47% whiff-punished) —
  disagrees with loops/coach; needs its floor + a blind pair. B3: the grab
  is UNCERTAINTY (4.1 bits) not a confident loop. Mechanism probe for
  mode-of-N's walk-off launched (`scripts/interp_mode_mechanism.exs`).
- 2026-08-30 01:15 — mode-of-N mechanism probe (`eval_runs/0830_mode_mechanism/RESULTS.md`):
  the vote's per-frame press rates ≈ the expert's; what it destroys is
  stick variety and press EDGES — modal output is a held direction (full
  left 23–37% in edge situations) with no button edge, so B lands as
  side-B/laser, never up-B, and the hold walks off. Any mode-seeking decode
  keeps the dense channel and loses the sparse one.
- 2026-08-30 01:30 — **Architecture note (from the mode-of-N mechanism):**
  the production head (`Heads.build_controller_head`) computes the 6
  controller heads IN PARALLEL from the trunk — no intra-frame
  conditioning at train (parallel CE) or inference (fused sampler). The
  "autoregressive" name refers to an unused `build_autoregressive` path.
  slippi-ai's `AutoRegressive` head conditions each component on the
  previous components' SAMPLES within the frame. Consequence here: an up-B
  is P(B)·P(stick up) as independent draws; the model cannot express "B
  implies stick-up" beyond what the trunk state carries. Candidate cause
  for airdodge-over-upB (A2) and option spam; a TRAINING change (head
  structure), not a decode one. Queued as a recipe question for v1.1.
- 2026-08-30 19:30 — **8a bracket 1 DISCARDED; heads refit with the real
  recipe; sampler fused.** Bracket 1 measured a fit bug, not the head: both
  arms fit with plain CE at 1e-3 (no pos-weights/smoothing/focal/edge/
  entropy) → 78% unforced deaths, 0/8 games to cap, ep10 reference 8/8.
  The A2/C4/coincidence numbers from `0830_arhead_score/` are artifacts of
  that — do not cite. Fixes: fit mirrors the source *_config.json loss
  recipe (pos-weights resolved from captured targets); AR sampler fused to
  ONE XLA program (24 ms → 2.53 ms/decision; bracket-1 AR arm was 50%
  stale); AR banner now warning-level so quiet-mode knob assertions see it.
  Run 3: ARhead val 3.892 vs INDhead 4.374. Rescore →
  `eval_runs/0830_arhead_score2/`. Law reaffirmed: a refit head must carry
  the training run's FULL loss recipe — val loss looked fine both times;
  only the live rung caught it.
- 2026-08-30 18:55 — **8a head-only fit ran** (plan item 8a; the lib work
  items 2–7 landed the same day): frozen ep10 trunk, 12.14 M rows (fox-only
  per-file detection, dittos both ports), 4 epochs. **ARhead val 2.231 vs
  INDhead 2.768 — 0.77 bits/frame, ≈ the audit's 0.86 TC prediction.**
  The independent head pays almost exactly the total correlation the audit
  measured, on the same trunk and data. §6 CPU-bracket scoring launched
  (`scripts/arhead_score.sh` → `eval_runs/0830_arhead_score/`); the
  pre-registered rule gates on A2 routes + live P(up|B), not on val loss.
- 2026-08-30 17:15 — **E1 (by file): v1's corpus is 43% NON-FOX demonstrations**
  (`eval_runs/0830_corpus_mix/RESULTS.md`). `--train-character` only filters
  files, never selects the port; v1's run set neither, so the loader took
  port 1 of every file: fox 56.7%, falco 9%, marth 5.7%, puff 4.6%, …
  Consequences: (a) "expert port 1" baselines in A1/B2/entropy are
  character-mixed — fox-only re-reads need per-file detection; (b) the 8a
  head fit uses per-file fox detection (dittos skipped) — deliberate,
  symmetric across arms; (c) v1.1/v2 recipe question: character-aware port
  selection (+ both ditto ports, same-split) yields ~5,470 clean fox games.
- 2026-08-30 16:30 — **Port-2 side-flip check** (Bradley played, bot port 2;
  `eval_runs/0830_port2_{B1,mode16}/sd.md`): mode-16's walk-off SDs went
  left 4 : right 3 (all 7 deaths that weren't hits were walk-offs) vs
  port-1's 24:3 LEFT. The walk-off direction FOLLOWS spawn side / held
  modal direction — model left-asymmetry rejected; consistent with the
  mode-mechanism law (vote holds a direction). B1 plain on port 2: zero
  walk-offs; its SDs (10:2 left-ish) are recovery deaths, the A2 class.
  Small n (3 games/set). Task 1 of HANDOFF_2026-08-30 closed.
- 2026-08-30 13:05 — **Joint-head audit** (`scripts/joint_head_audit.exs`,
  `eval_runs/0830_joint_head_audit/RESULTS.md`): within-frame total
  correlation of (buttons, main_x, main_y) = 0.86 bits/frame and it does
  NOT shrink when conditioning on action-state + situation (0.84–0.86) —
  the dependency is between the same frame's inputs. P(stick up | B,
  offstage) = 42% vs 15% marginal (2.9×): an independent head makes an
  up-B out of one B press in seven; the expert two in five. Plan:
  `docs/planning/AUTOREGRESSIVE_HEAD_PLAN.md` (pre-registered).
