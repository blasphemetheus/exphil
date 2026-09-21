# ExPhil: Goals

**Last Updated:** 2026-09-19 (rewritten around the V3/V3.1 generalist line
and the identity program; previous version 2026-08-03 is in git history).

This document states **what the bot should do** and where each line stands,
measured. Inventory lives in the appendix so we don't rebuild it. Every
number traces to an artifact; the live day-to-day list is the newest
`HANDOFF_*.md`.

---

## What we're building

Three bots and a teacher:

1. **A generalist Fox** that plays Melee like the players it learned from —
   and, on request, like *one* of them by name (Track D).
2. **A Fox that multishines perfectly** under any delay rung (Track B — met).
3. **A Mewtwo that goes in with short-hop fair and converts** (Track A — the
   decision problem; still the hard one).
4. **A coach** that steers a human into a drill situation and gives feedback
   (Track C — long-term; its prerequisites are this program's roadmap).

Low-tier characters remain the long-term interest. High-tier replay data
drives the generalist and the identity work because that is where the
corpus is.

---

## Where each line stands (measured, 2026-09-19)

| Track | Goal | State | Evidence |
| --- | --- | --- | --- |
| **D — generalist Fox** | plays a full game as a human Fox would; conditions on player identity | **Playable and improving.** V3 (4.4 passes) → best held-out 1.94; V3.1 (weight decay + cosine LR, warm start) val 2.253 → 2.193 → 2.158 across 3 passes, final pass running. Bradley: "definitely the best version of a bot I've made" (many characters). Style conditioning **live**: `--style-tag` flips the jump button and c-stick habits to the named player's. | `HANDOFF_2026-09-18.md`, `STYLE_IDENTITY.md` S6(c) |
| **B — Fox multishine** | multishine under any rung | **Met** at rungs 0/2/4; playable local demo; proof method generalizes to any drill | `MULTISHINE_PIPELINE_PROOF.md`, `LOCAL_MULTISHINE_DEMO.md` |
| **A — Mewtwo decision** | armed approaches/min ≥ 1.0; conversion ≥ 25 %; ≥ 2 connected aerials per opening | **Unchanged since July**: 0.17 / 5 % / unmeasured. Neutral teacher approved; five learned rounds, none promoted; conversion event scorer not started | `MEWTWO_NEUTRAL_TO_COMBO_HANDOFF.md` §10 |
| **Identity** (enables D, feeds C) | every game trains under an identity, not id 0 | **S0-S6 done.** 58 % of the corpus conditioned (real tags + 39 style clusters); open-set top-1 0.67, closed-set 0.90 at 3 candidates; aliases adjudicated (merges/holds), FOX/LI collisions voided for V3.2 | `STYLE_IDENTITY.md`, `STYLE_RECONCILIATION_2026-09-17.md` |
| **Sim** (enables A, later RL) | trusted, settable-state engine for drills/eval/search | **Plan only.** Validation gate (input-replay parity vs our `.slp`) not run; Mewtwo/G&W admitted on a local branch; other sessions extending character coverage | `MELEE_SIM_USES.md` |
| **Direct exhibition** | ≥ 1 stock off a human over Slippi Direct, with consent | Not attempted with the generalist; the multishine bot did it locally. Rule unchanged: Direct only, never matchmaking (EXPH#288). | — |

---

## Track D — the generalist Fox (NEW; the V2 → V3 line)

**Goal:** a Fox that plays a whole game — neutral, punish, recovery, edgeguard
— the way the corpus's players do, and that can be asked to play like a
specific one.

**Recipe that works (V3.1):** GRU 512×2, BPTT unroll 80 / overlap 0, batch
128, F32 + highest EXLA arithmetic, AR head, T=1.0 sampling, reaction 0,
identity channel via `--player-tag-map`, AdamW with **weight decay 0.05**,
**cosine LR**, non-finite-gradient guard + fatal-batch capture. Deployment:
`--stateful-step --live-af --reaction-delay 0 --temperature 1.0`, never
`--deterministic`.

**What the V3 run taught (three divergences, all captured):** a single
non-finite gradient turns global-norm clipping's scale into NaN and writes
NaN into every weight while the loss still looks fine; the GRU hidden-kernel
norms drift upward without weight decay until the 80-step backward
overflows; dropout amplifies it. Weight decay reverses the norm drift, the
guard makes the rare batches harmless, the LR step-down alone was worth
−0.75 held-out. Skip counts still escalate late in training (10 → 41 → 188
per pass) — the backward horizon is the next lever.

Gates (all measured on the same instruments as the drills):

| Gate | Metric | Now | Target |
| --- | --- | --- | --- |
| D1 — plays | live: latency 1, 0 errors, no idle/shield lock, moves | ✅ (40 % horizontal input, 0 stocks lost vs CPU 6 in 30 s) | keep |
| D2 — learns | held-out teacher-forced CE on tagged games | 1.94 (V3 215k); V3.1 trainer-val 2.158 | monotone per pass |
| D3 — conditions | paired anonymous-vs-registry scores differ exactly on identified files; `--style-tag` moves the fingerprint toward the player | ✅ exact; jump-button tell reproduces per player | effect on option mixes, n ≥ 10 games/arm |
| D4 — executes | L-cancel press offset, short-hop rate, wavedash angle vs the imitated humans | below all four probed humans | within their range |
| D5 — beats | ≥ 1 stock off a human over Direct | not attempted | do it |

**Next for D:** V3.2 = V3.1 recipe + unroll 40 (or per-timestep gradient
value clipping), S4 alias merges + FOX/LI voided in the tag map, 8 fresh
passes; norm + skip logging per epoch; D4 measured properly (the execution
gap is a training-budget/sampling question, not an imitation limit).

---

## Track A — Mewtwo: the DECISION problem (unchanged in substance)

**Goal:** approach with short-hop fair; chain fair→fair when the option exists.

| Gate | Metric | Now | Target | Tooling |
| --- | --- | --- | --- | --- |
| **A1 — goes in** | armed approaches/min | 0.17 | ≥ 1.0 | `ReplayStats.approach_stats/2` |
| **A2 — connects** | conversion rate | 5 % | ≥ 25 % | `ReplayStats.conversion_stats/2` |
| **A3 — chains** | connected aerials per opening | *unmeasured* | ≥ 2.0 | **needs building** — still true |

Diagnosis stands: it learned the sequence, not the decision. The neutral
teacher (`MewtwoNeutralTeacher`) is approved; no student decides. The next
packet is the fair-conversion EVENT scorer and one reproducible first-fair
contact (`MEWTWO_NEUTRAL_TO_COMBO_HANDOFF.md` §10). What changed since
August: the **sim** is the intended instrument for A — randomized-start
conversion drills at scale — once its validation gate passes and Mewtwo is
admitted; and the generalist recipe (Track D) is the proven way to train a
whole-game imitator, which Mewtwo will need once decisions are labelled.

---

## Track B — Fox: the EXECUTION problem — MET

Multishine proven at rungs 0/2/4 with the recorded-teacher-clip method,
held-out validated, playable locally (`LOCAL_MULTISHINE_DEMO.md`). The
method (proof contract, teacher clips cold+warm, gates in fixed order,
declared held-out) is the template for any drill. Historical detail:
`MULTISHINE_PIPELINE_PROOF.md`, `LATENCY_ARCHITECTURE.md`. Open: no
default multishine policy installed in the play scripts.

---

## Track C — the Coach (long-term)

Unchanged (`COACH_ROADMAP.md`). Two of its prerequisites moved this month:
the identity model (who is playing) exists, and the fingerprint instrument
(what habits a player has) exists and is calibrated — the "what's good per
situation" model does not.

---

## Method: the escalation ladder (still the rule)

1. Fix what the expert teaches — supervise the decision, not the buttons.
2. More / better imitation data — identity conditioning is now real; the
   corpus is 28k Fox games; the sim can generate on-distribution labels.
3. RL only after 1 and 2 are exhausted — the sim makes this affordable
   when its time comes.

---

## Standing rules (accumulated)

- Never crown on stand-dummy numbers; rank at the deploy rung.
- Every retrain carries all validated mixes; declare a new round for a new
  budget; keep failed rounds on disk.
- Delayed recovery labels come only from recorded teacher clips.
- No `mix` on the dev box while any exphil beam is live (NIF invalidation
  kills it — includes CPU-only jobs).
- Non-finite gradients are skipped and captured, never applied; a
  divergence is replayed before any relaunch.
- A conditioning channel is proven live only by a measurement that changes
  when the channel changes.
- Slippi Direct only, with consent; never matchmaking.

---

## Explicitly NOT goals right now

- Architecture bake-offs (GRU is fine; the bottleneck was the optimizer and
  the data channel, not the backbone).
- Five-character program at scale — after Mewtwo decides.
- League / population play — downstream of a bot that can beat a human.
- Play-time search in the sim — a bespoke decode rule in spirit.

---

## Next work (2026-09-20) — three lines, three docs

The program now runs as three parallel lines, each with its own planning
doc, goals, and status ledger. GOALS.md stays the big picture; the docs
hold the plans.

1. **Imitation squeeze** — `IMITATION_SQUEEZE.md`: pilot-first (5 %
   matched-seed fits + cliff resumes), no long run until the pilots pick
   the levers; V3.2 assembled from the winners.
2. **RL on the prior** — `RL_ON_PRIOR.md` (+ `SIM_INTEGRATION.md`, the sim plan/tracker): self-play/PPO on V3.1-ep3 in
   melee-sim-light, KL-to-prior, gates R0–R5 (R0 = sim adapter parity).
3. **Coach and style products** — `COACH_STYLE_PRODUCTS.md`: style report
   card, play-like-X slot fine-tune, sparring launcher, C1–C3 overlays.

Still queued from before: sim validation gate human lane (needs the
declared scene profile), Direct exhibition (D5, shared with R5).

---

## Appendix: what already exists

Listed so we don't rebuild it. Not a scorecard.

**Evaluation & diagnosis**
- `ExPhil.Interp.ReplayStats` — approach/conversion/opening stats (A1, A2)
- `ExPhil.Interp.StyleCard` — per-character gates incl. opener diversity
- `ExPhil.Eval.ScenarioScan` / `scripts/scenario_suite.exs` — situational probes
  via input-prefix virtual savestates
- `ExPhil.Eval.NeutralScan` — opener taxonomy + entropy
- `ExPhil.Eval.FailureScan`, `ExPhil.Eval.GapLedger`, `scripts/coach_report.exs`,
  `scripts/auto_bookmarks.exs` — the gap flywheel
- `ExPhil.Eval.Coverage`, `ExPhil.Data.SituationIndex`,
  `scripts/find_situations.exs` — occupancy diffing + situation retrieval

**Training**
- `scripts/dagger_drill.exs` — the drill (conversion + opener weighting,
  style conditioning, probe-reg, optional `--stream-chunk-size`)
- `ExPhil.Data.TrainingShards` — memory-flat streaming (unblocks large corpora)
- `scripts/curate_bc.exs` — corpus curation by initiation richness
- `scripts/human_drill.exs`, `scripts/selfplay_rollouts.exs`,
  `scripts/build_seed_dir.exs` — drill/rollout data generation
- 43 backbones via Edifice; ~2700 tests

**Corrections to the old version of this doc**
- Self-play / RL and "PPO integration" were marked ✅ **Complete**. They are
  **not**. PPO had never executed once; six surface bugs are now fixed and an
  architecture mismatch remains (MLP actor-critic vs temporal Mamba trunk).
  See `docs/planning/PPO_STATUS_2026-07-23.md`. This matters because PPO is
  rung 3 of the ladder above — it is not a quick pivot.
