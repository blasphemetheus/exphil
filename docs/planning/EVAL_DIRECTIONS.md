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
| `loop_report` / LoopStats | d_up presses, taunts, frozen/held input, repeated action cycles | per-game averages; loops/min is not a pathology score by itself. 08-31: stub games (<150 KB) excluded+counted, cells now mean~median (the D2 331-taunts/min lesson) |
| `coach_report` / FailureScan | armed approaches, conversions, dropped punishes, deaths, passivity, neutral losses | armed/min drifts 7× day to day |
| `interp_passk` (Leg S) | pass@k − pass@1 on decision frames | mode-seeking; never rank decodes on it (L9). 08-31: AR-aware — AR checkpoints sample SEQUENTIALLY via `Agent.get_action_samples` (n draws from one forward's logits would score the AR head as independent) |
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
| B1 | Action-family match | Did the bot pick the same *category* (aerial/grab/shield/movement/special) as the expert in that state? Softer pass@1. | **ran** | `scripts/action_family_match.exs` → `eval_runs/0830_family_match/RESULTS.md` (overall family TV ~0.32–0.37 across all sets; offstage: AR/IND 0.60/0.63 vs B1_human 0.26 — family mix offstage is the gap) |
| **B2** | **Distribution distance per situation** | KL / total-variation between the bot's next-action histogram and the expert's, per situation. Flags narrowing decodes (argmax, mode-of-N) offline — the pass@1 replacement as a decode ranker. | **ran — adopt** | same script; mean TV orders B1/B2/B3 (0.46–0.47) < ep10 (0.53–0.56) < mode-16 (0.72), matching the human read and the live rung |
| B3 | Entropy per situation | Policy-output entropy by situation from captured logits; low entropy where the expert is diverse = a loop waiting to happen. | **ran** | `scripts/interp_entropy_by_situation.exs` → `eval_runs/0829_entropy/RESULTS.md` (grab: 4.1 bits, expert decisive) |

### C. Temporal structure — the "scrappy / harder to hit" axis

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| C1 | Neutral-exchange outcomes | Segment games into exchanges; who won each and how (first hit / trade / whiff punish). Dense "harder to hit" number. | **ran + floor** | `eval_runs/0829_neutral_exchange/RESULTS.md`; floor (08-30, `floor.md`): three same-decode CPU batches score 64.0/63.8/63.4% — spread **0.6 pp**. The metric is TIGHT on the CPU rung; B1's 20% vs Bradley is far outside any floor → the human-rung disagreement is real (blind pair still owed) |
| C2 | Punish quality | Damage per opening, combo length, % openings ending in a kill, vs expert. | **ran** | `eval_runs/0830_punish_quality/RESULTS.md` — damage/opening ≈ expert (10.9 vs 10.9) but 5.1 hits/opening vs expert 1.72 (many small hits, no finisher) and kill conversion 0.0% vs 0.62%/opening |
| C3 | Reaction latency | Time-to-action after opponent lands / grabs ledge / bot lands, vs expert. Dithering. | **ran** | `eval_runs/0830_reaction_latency/RESULTS.md` — after opp lands: expert med 8f, ep10 12f, AR 32f/IND 45f (≥cap 47–64%!) — the refit heads DITHER on opponent landings; self-lands normal (7f) |
| C4 | Stock-loss forensics | Every death classified: unforced walk-off / failed recovery / edgeguarded / combo'd / neutral kill. Extends `sd_scan`. | **ran** | `scripts/death_classifier.exs` → `eval_runs/0829_death_classifier/RESULTS.md` (bot dies at 52–71% vs expert 109%) |

### D. Reliability of the instruments

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| **D1** | **Noise floors** | Same checkpoint, different days/batches: the natural spread of every metric. Which numbers can ever resolve a real difference? | **ran** | `scripts/noise_floor.sh` → `eval_runs/0829_noise_floor/RESULTS.md`: d_up 1.1×, held 1.2×, loops 2.0×, deaths 2.5–3×, conv% 3.5×, armed 7×. D1b ran: TV floor 0.05 on the mean, 0.1 per situation (`eval_runs/0829_noise_floor/tv_floor.md`) |
| D2 | Human-vs-CPU transfer table | For each metric, CPU-bracket value vs Bradley-session value for the same checkpoint. Which metrics the CPU rung can stand in for. | **ran** | `eval_runs/0830_d2_transfer/RESULTS.md` — dpad/taunts transfer (~0.8× human/CPU factor); loops/min: CPU overstates ~2×, ordering only; ep10_human taunt row RESOLVED 08-30: one degenerate 100 KB stub game at 3600/min poisoned the mean (not d-pad, not port-map); loop_report stub-filters + reports medians since 08-31 |
| D3 | Decode-vs-model sensitivity | From banked temperature sweeps: which metrics are purely decode-driven (press rates) vs model-driven. | **ran** | `eval_runs/0830_decode_sensitivity/RESULTS.md` — dpad/min, longest action/input runs = decode-driven (rho=±1.0, 4x range); taunts/min NOT knob-ordered (noisy); loops/min decode-leaning |

### E. Data-side audits

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| E1 | Corpus mix by frame | Characters / stages / players by frame; concentration (one player's style dominating the mode). | **ran (by file)** | `eval_runs/0830_corpus_mix/RESULTS.md` — v1 imitated port 1 blindly; 56.7% fox, 43% opponent characters. By-frame + player-concentration refinement still open |
| E2 | Rare-event coverage | How many expert examples of the exact situations the bot fails in. Hundreds = selection problem; dozens = data. | **ran — not data quantity** | `scripts/rare_event_coverage.exs` → `eval_runs/0829_rare_events/RESULTS.md` (thousands of labels/epoch for every missing behaviour) |
| E3 | Expert pathology baselines | Pummels, taunts, standing lasers per game in *expert* play, so "too much" has a denominator. | **ran** | `eval_runs/0830_expert_pathology/RESULTS.md` — expert (fox-only, 345 games): taunts 0.06/game, d-up 0.15/game (bot arms press d-up 79–94/MIN — 1000×+ expert), pummels/grab 0.14, throws/grab 0.56, specials 13.5/min |
| E4 | Left walk-off poison | Is the corpus poisoned with repeated one-sided SDs? | **ran — falsified** | `scripts/sd_scan.exs`, `eval_runs/0829_sd_scan/RESULTS.md` |

### F. Defense and commitment (Bradley's 08-31 live-look asks)

| id | instrument | question | status | script / results |
|---|---|---|---|---|
| **F1** | **Defense scorecard** | When hit (esp. sent offstage): does the subject hold survival DI/drift toward stage/ledge, or throw the stick elsewhere / burn airdodge in danger? Stick-vs-stage-direction during hitstun + drift-after-hitstun + airdodge-while-in-danger rate + survival-given-situation, vs expert. Quantifies "you can hold a direction and live; it doesn't." | **ran** | `scripts/defense_scorecard.exs` → `eval_runs/0831_session_score/defense_scorecard.md` — the gap is NOT the drift (AR 87.9% toward-stage vs expert 91.2) but hitstun DI (67 vs 83) and above all the **airdodge panic button**: 33% of offstage hitstun episodes vs expert 2.4%, dying 53% of the time within 90f (expert 18%) |
| F2 | Punishable-commitment rate | How often does the subject initiate a laggy option (smash, grab, spotdodge, whiffed special, landing-lag aerial) while the opponent is in threat range and actionable — and P(punished \| committed)? The mechanism behind C1's whiff-punish outcomes ("it puts itself into whiff punish"). | **ran** | `scripts/commitment_scorecard.exs` → `eval_runs/0831_session_score/commitment_scorecard.md` — pathology is 3.2× committal VOLUME (19.8–22.1/min vs expert 6.2), not per-commitment timing; opponent confound noted |
| **F3** | **Position-dependence probe** | Bradley's 08-31 evening ask: "does it actually value where the opponent is?" Mechanistic counterfactual on live-look states: PERCEPTION (does any head's distribution move when opp position is perturbed — mirror/far/close-left/close-right) and DIFFERENTIATION (approach_delta = E[main_x \| opp right] − E[main_x \| opp left]; P(grab) near vs far). Separates "doesn't see position" (curation lever) from "sees it, doesn't select on it" (selection lever, the F1 signature). | **ran** | `scripts/probe_position_dependence.exs` → `eval_runs/0831_position_probe/RESULTS.md` — perception ALIVE; approach_delta **−0.20..−0.24** (steers AWAY, survives :neutral filter); P(z) ~0.30 per neutral frame barely tracking range (grab spam is baseline, not targeting); AR≈IND → trunk property. **F3b sweep (09-01)**: no approach band at ANY distance (delta negative 15→130 units, both heads); retreat GROWS with range (−0.07 @15 → −0.26/−0.28 @90); point-blank = defensive-option mode (steering weakest, P(z) peak 0.33) → `RESULTS_sweep.md` |

Coverage note (08-31): airdodge-offstage = A2 first-route; whiff-punish
as OUTCOME = C1 exchange classes; no-combo = C2; no-dash-dance = A1 +
the dash probe (mechanism: closed-loop drift, not decode). F1/F2 are the
two named complaints with no instrument.

## Order of work

*(Historical — the full program A→E ran 08-29→08-30; every direction has a
RESULTS. The doc now serves as the instrument registry + dated log.)*

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

- 2026-09-01 — **F3c neutral-range scorecard: Bradley's pushback CONFIRMED,
  the 08-31 "corpus retreat-laser zone" story RETRACTED**
  (`eval_runs/0901_neutral_range/RESULTS.md`, `scripts/neutral_range_scorecard.exs`).
  Expert Fox neutral is APPROACH at range — toward rises 0.095 → 0.609
  with distance (%twd 70.5 vs %awy 8.0 at 140+), dash-dance band at
  40–70 (dash 26.8%), aerial laser a seasoning (peak 6.2% at 70–100).
  The bot inverts nearly every column (toward negative everywhere <140,
  dash 0.1–0.3%, jump ~1%, aerial laser BELOW expert in the SH-laser
  band, grounded laser ABOVE everywhere — matches Bradley's "no SH
  laser / FH double laser"). Consequence: the trunk's all-range retreat
  is NOT the corpus marginal — BC loses the expert's distance-conditional
  steering (closed-loop drift family). The approach data IS in the
  corpus; weight shifts from curation toward training/decode losing it.
  Also 09-01: **plan (c) V-rollout selector build STARTED**
  (`scripts/vrollout_eval.exs` + dynamics model saved to
  `checkpoints/dynamics_fox_v11AR.bin`, `--save` added to the spike).
  Two recorded negatives on the way: V-on-raw-embeds fails (linear
  ANTI-correlates rank 0.406; MLP exactly chance 0.503) → v2 design
  scores the imagined embed window (window−k real + k predicted) through
  the POLICY'S OWN TRUNK with an MLP V on trunk features (the 08-31
  critic's V only worked on trunk+raw phi). Run in flight.
- 2026-08-31 evening — **v1.2-ARrefit live look (Bradley): survival much
  improved, airdodge panic visibly down, occasional SDs remain; NEW
  headline complaint = no positional play** (identical at either ledge /
  center, option-spam without targeting, never dash-dance→JC upsmash) —
  `eval_runs/0831_livelook_v12ar/IMPRESSIONS.md`. Three results the same
  evening: (1) **critic ladder rerun on ARrefit still under the bar** —
  selector 10.8 vs mode 7.2 in-dist / 4.0 vs 3.0 fresh (+3.6/+1.0 vs
  v1.1-AR's +2.7/+1.4): restoring the wire did NOT widen the selector
  margin → the selection gap isn't waiting on the AR wire
  (`eval_runs/0831_critic_refit/`). (2) **G3b dynamics spike PASSED its
  pre-declared gate** after two debug rounds (embed_frames returns a
  dataset not a tensor; incremental-concat OOM): held-out 1-step R²
  0.996, cos@10 0.827 → **V-rollout selector (plan c) unblocked**
  (`eval_runs/0831_dynamics_spike/RESULTS.md`). (3) **F3 position probe
  built + ran on live-look states**: the bot SEES the opponent (all
  heads move under counterfactual displacement) and systematically
  steers AWAY (approach_delta −0.20..−0.24, survives :neutral filter);
  grab ~30% per neutral frame barely tracking range; AR≈IND → trunk
  property. Verdict: opponent position is encoded as THREAT, never as
  target — approach/punish is missing behavior, feeding curation target
  #1 and the V-rollout selector (`eval_runs/0831_position_probe/RESULTS.md`).
- 2026-08-31 15:05 — **Critic ladder on coherent candidates: PARTIAL-plus**
  (`eval_runs/0831_critic_ar/RESULTS.md`). First selector WIN over the
  free majority vote (13.7 vs 11.0 in-dist, 6.5 vs 5.1 fresh; 08-29 it
  lost), gap-recovered >50% both corpora — but margin over mode-of-N
  +2.7/+1.4 pts, under the pre-registered ≥5 → not wire-live. Shuffled
  control at 11.5 (vs selector 16.5) shows most of the lift is generic
  action-frequency preference. Rerun queued on v1.2-ARrefit (whose wire
  is restored). Also 08-31 afternoon: **coincidence probe** — unfreeze
  ATROPHIED the AR wire (L_cond 2.67→1.14, R_state flat; instrument
  validated against both live lifts) → v1.2 refit chain queued; **F2
  commitment scorecard** — pathology is 3.2× committal VOLUME (19.8–22.1
  /min vs expert 6.2), not per-commitment timing; **F1 defense
  scorecard** — the defense gap is the airdodge panic button (33% of
  offstage hitstun episodes vs expert 2.4%, dies 53% within 90f), drift
  itself near-expert.
- 2026-08-31 08:30 — **v1.1 unfreeze trained AND scored overnight**
  (`eval_runs/0831_v11_unfreeze/TABLE.md`, `eval_runs/0831_v11_score/RESULTS.md`).
  Corpus FIXED via `--select-character-port` (E1); trunk-transplant resume
  worked live. val AR 5.480 vs IND 5.973 (the conditioning gap at scale).
  §6 verdict: **NO SIGNAL between arms** — routes inverted (IND up-B 15.6
  vs AR 9.4) while outcomes favor AR (died 28.1 vs 35.6, deaths/game 1.13
  vs 2.13), everything at/under D1 floors at n=8 → **the live look
  decides** (g6). The un-confounded headline: BOTH arms crush ep10 on
  recovery (died 28/36% vs 61%) — unfreeze+corpus works. Flagged open:
  the 8a coincidence lift inverted post-unfreeze (AR 1.28× vs IND 1.93×
  offstage) — "trunk absorbs the coincidence when unfrozen" is the
  interesting candidate, small-n the boring favorite; teacher-forced
  probe queued behind the live look. Ops: ENOSPC at 02:53 → GOTCHA #106.
- 2026-08-31 02:05 — **0831_legS_ar LANDED** (`eval_runs/0831_legS_ar/RESULTS.md`):
  joint pass@1/pass@16/headroom — ep10 16.5/44.5/+28.0, INDhead
  8.1/36.6/+28.5, ARhead 8.5/38.6/**+30.1**. Pre-registered readings:
  (1) AR−IND pass@1 +0.4 = UNRESOLVED — conditioning is invisible to
  open-loop match; the AR payoff lives in rare coincidence events pass@k
  barely weights (why 8a gated on routes, not match). (2) **Headroom
  SURVIVES on the AR base** — selection gap is NOT a factorization
  artifact → critic program keeps priority; G3b (learned dynamics) gate
  OPENS after the unfreeze. (3) ep10 fox-only ≈ 08-28 numbers —
  instrument continuity, Leg S conclusions survive E1. Bonus: both
  frozen-trunk refits at HALF ep10's pass@1 (buttons-driven) while
  ARhead beats ep10 live on recovery — pass@1 is not a skill score,
  measured twice in one table. Ops law: no inline scripts under
  systemd-run (`${var}` got eaten by the quoting stack); script FILES only.
- 2026-08-31 01:00 — **Leg S made AR-aware and relaunched on the head arms**
  (`eval_runs/0831_legS_ar/PREREG.md`, unit `legS-ar`, running). interp_passk
  drew six components independently from one forward's logits — correct for
  independent heads, silently wrong for AR checkpoints (ignores the
  conditioning AND reuses one path's conditional logits). New:
  `Sampling.sample_autoregressive_n` (the fused mode-of-N machinery minus
  the vote, key-seedable) via `Agent.get_action_samples` (debounce-free,
  side-effect-free). Three arms — ep10 / v1.1_INDhead / v1.1_ARhead —
  fox-detected ports (E1), n=16, T=0.5/0.5, seed 20260831. Pre-registered
  readings: joint pass@1 AR−IND (≥3 pts = conditioning visible open-loop)
  and headroom AR vs IND (shrinks ≥5 pts = part of the old "+29 selection
  gap" was factorization; survives = critic program keeps priority on the
  AR base). Also 08-31: loop_report stub filter + mean~median cells;
  trunk-transplant resume (`--head` mismatch / `--reinit-head`) landed for
  plan items 8/9.
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
- 2026-08-30 22:10 — **Fox-only expert re-reads** (post-E1; `--expert-char`
  added to situation_hist/edge_scorecard/death_classifier;
  `eval_runs/0830_foxonly_rereads/README.md`): every conclusion stands and
  the recovery gaps GROW — fox-only expert: up-B 20.6% (was 16.3), dj 28.9,
  airdodge 3.7, none 9.9; unforced deaths 19.0% (was 26.9) vs bots' 61–65%
  (~3.3×); expert deaths are 50% edgeguarded at ~109%. B2 TV +0.01–0.02
  uniform, ordering unchanged. Also: D2's ep10_human taunt row RESOLVED —
  one degenerate stub game (3600 taunts/min, dpad 0) + ditto warmups in the
  session dir; sane games ≈ 2/min. Lesson: session dirs need stub AND
  degenerate-game filters; prefer medians for session groups.
- 2026-08-30 21:00 — **Bradley's live look at v1.1-ARhead** (the g6 gate on
  the 8a SIGNAL): "like a good player BMing or sandbagging — spamming bad
  options, but then occasionally doing really good strings; not punishing
  well." Read: the joint structure is REAL to human hands (the strings) and
  matches the metrics' shape — C2's many-small-hits/no-finisher is the
  "not punishing", C3's post-landing dithering is the "spamming bad
  options" between strings. Verdict: structure confirmed, selection/polish
  not — consistent with the frozen-trunk caveat. Next per plan: items 8/9
  (3-epoch unfreeze from ep10 with the AR head, IND control).
- 2026-08-30 20:15 — **Instruments first-run batch** (E3, C2, C3, B1-family,
  D3, C1 floor, D2 — all landed; statuses updated above). Cross-cutting
  reads: (1) C3 is the sharpest new lens — the refit heads DITHER after the
  opponent lands (AR med 32f / IND 45f vs expert 8f, ep10 12f) while
  self-landing reactions are normal; a candidate mechanism for "spectates
  after hitting". (2) C2: damage/opening matches the expert (10.9) but 5.1
  hits/opening vs 1.72 and kill-conversion 0 — many small hits, no
  finisher. (3) E3 denominators: expert taunts 0.06/game, d-up 0.15/game —
  the bots' d-up 79–94/MIN is ~30,000× expert; with D3 showing dpad/min
  rho=1.0 with buttons-T, the taunt pathology is decode-side pressure on a
  button the expert simply never touches. (4) C1's CPU-rung floor is 0.6 pp
  — exchange win rate is our tightest live metric.
- 2026-08-30 20:10 — **8a bracket 2: SIGNAL**
  (`eval_runs/0830_arhead_score2/RESULTS.md`). ARhead vs INDhead, healthy
  games both arms. A2: up-B first-route 8.3% vs 0.0 (expert 16.3), up-B+dj
  6× IND, airdodge halved (20.8 vs 53.2), recovery deaths 35.4% vs 51.1%.
  Live coincidence: P(up|B)/P(up) = 2.20× (offstage 2.65×) vs IND's 1.05×
  — expert is 2.9–3.2×. TV within floor of IND (0.62 vs 0.61); both above
  ep10's 0.53 (frozen-trunk refit lags overall polish). Per the
  pre-registered rule: AR = default-head candidate; **Bradley's live look
  is the gate**; if confirmed, plan items 8/9 (unfreeze) chase polish +
  structure together.
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
