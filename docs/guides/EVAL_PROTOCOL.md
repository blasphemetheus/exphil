# Evaluation Protocol — paired, sequential, budgeted

**Created 2026-08-04 (ML_FIELDS_ROADMAP F1).** The scarcest resource in
this lab is Dolphin game time, and the worst mistake of the program so
far (crowning ms_g6_sp1 on a stand-dummy score it then inverted against
a human) was an EVALUATION failure, not a training failure. This doc
formalizes the statistics we were already half-using so every eval
decision spends the minimum games for a stated error budget.

## 1. Common random numbers (CRN) — the paired-comparison rule

**FD and BF stand-dummy runs are pixel-deterministic** (3-run identity
verified repeatedly; 2026-08-04 OOD runs scored bit-identical across
r1/r2/r3). DL and YS are NOT (the YS absorber is stochastic 2-of-3).
Determinism is a variance-reduction gift: two policies evaluated under
the same recipe on FD differ by POLICY only, nothing else.

Rules:
- **Deterministic stages (FD, BF): ONE run per (policy, recipe).** The
  other two runs of the traditional 3 are pure waste — they reproduce
  the first bit-for-bit. Spend them on another rung instead.
- Always compare candidates under the IDENTICAL recipe: same stage,
  dummy script, delay, decode flags, `--deterministic` sampling, same
  ISO/dolphin build. A comparison across recipes is not a comparison.
- **Stochastic stages (DL, YS): 3 runs minimum**, and treat the
  RUN-LEVEL outcome (collapse / no-collapse) as the datum, not the
  mean — the YS distribution is bimodal (285/min or ~40/min, nothing
  between), so a mean over 3 runs is an artifact.
- Cross-machine caveat: determinism holds per-machine; the 5090/laptop
  harness differs by a frame (memory: 5090 one frame faster). Never mix
  machines inside one comparison.

## 2. Sequential stopping — stop when the verdict is in

Fixed game counts overspend on blowouts and underspend on close calls.

- **Deterministic rungs are already sequential**: one game IS the
  verdict for that rung.
- **YS collapse rate** (bimodal binary): treat as Bernoulli. Stop early
  when the conclusion cannot flip within the planned budget: e.g.
  comparing collapse counts out of 3, a 0-vs-2 split after two runs each
  is already decided for ranking purposes; a 1-vs-1 needs the third run.
  For finer estimates (calibrating a rate), 3 runs cannot distinguish
  1/3 from 2/3 — do not pretend otherwise; either accept the coarse
  bucket {never, sometimes, mostly} or budget 8-10 runs.
- **Human sessions** (the truly expensive rung): decide the stopping
  question BEFORE the session ("does this policy chain >= 2 vs a
  human?") — a yes/no per game, stop at the first decisive game. The
  08-04 session showed 2 games suffice for a zero-vs-nonzero verdict.

## 3. Successive halving — ranking N candidates

Never run the full protocol on every checkpoint. Budget concentrates on
survivors:

1. **Round 0 (offline, seconds/policy):** opponent-sensitivity rung
   (`probe_opponent_dependence.exs`, promote_check rung 0; LOW=robust)
   + fixture agreement offset sweep. Drop anything anomalous.
2. **Round 1 (1 deterministic game/policy):** FD stand at the DEPLOY
   rung, scored by CHAINS. Keep the top half (or all within ~10% of the
   leader — the deterministic score has no sampling error, so small
   gaps are real).
3. **Round 2 (3 YS runs/survivor):** collapse-rate bucket.
4. **Round 3 (3 vs-policy games/survivor):** moving-opponent rung.
5. **Human games:** only for the single promotion candidate, with the
   pre-registered stopping question.

Ten candidates cost ~10 offline probes + 10 FD games + ~15 YS/vs games
+ 2-3 human games — versus 90+ games under flat 3×3-per-policy.

## 4. The calibration ledger

Every human session adds a labeled (offline scores → human outcome)
pair. Append here; when n reaches ~8-10, fit the discriminator and
revisit promote_check thresholds. Score CHAINS from replays
(`analyze_shine_source.exs`), never qtrace press counts.

| date | policy | opp-sens (rung 0) | FD chain @deploy | YS collapse | human shines/game | human max chain |
|---|---|---|---|---|---|---|
| 2026-08-03/04 | ms_g4_d2mix | 1.34 | 423 | 2/3 | 40 | 2 |
| 2026-08-03/04 | ms_g2_mdq_ss | 1.83 | 367 | — | 21-29 | 1 |
| 2026-08-03/04 | ms_g6_sp1 | 3.84 | 409 (id 3) | — | 0 | 0 |
| 2026-08-05 | **ms_g10b_human** | 1.44 | 421 | **0/11** | 89-109 self (28-33/min) | **3** (netplay; lag peak 5 @ 99.8%) |
| 2026-08-05 | ms_g10b_human (LOCAL, ~2f) | — | — | — | 47-94 self | **22** — delay regime costs 22→3, same day/opponent |

Reading so far (n=3): FD chain does NOT rank human outcome; opp-sens
ranks it inversely and perfectly. Neither is calibrated — that is what
this table is for.

## 5. Error-budget language for promotion decisions

A promotion claim must state, in the handoff:
- which rungs ran, under which recipe (stage/delay/dummy/decode);
- the chains numbers AND the run-level YS outcomes;
- what would have REVERSED the decision (the pre-registered condition —
  cycle-3's P1/P2/P3 header is the model);
- for anything touching humans: how many games, and the stopping rule
  used.

"It won the comparison" without the recipe and the reversal condition
is how g6 got crowned.

## Cross-references

- `scripts/promote_check.sh` — the rung implementation (0-3 above).
- `docs/planning/HUMAN_PLAY_FINDINGS_2026-08-04.md` — why dummy
  rankings invert.
- `docs/planning/ML_FIELDS_ROADMAP.md` F1 — the research framing.
- GOTCHA #79 / `test/support/metric_floor.ex` — any NEW metric used in
  an eval needs its floor test first.
