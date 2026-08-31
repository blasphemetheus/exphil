# INTERP_GEN_V1 — interpretability program for the generalist line

**Status stamp (2026-08-31): essentially CLOSED.** G1/G2/G4/G7 done with
verdicts (results ledger below); G6's curation targets fell out of G1
(defensive/disadvantage pocket, punish continuation); **open: G3b**
(learned dynamics model — redefined 08-31, see G3; the table-based
CycleSim adaptation is DROPPED) and **G5** (style separability) — both
parked, low priority. The program's live successors are
`EVAL_DIRECTIONS.md` (the eval instrument bank + log) and
`AUTOREGRESSIVE_HEAD_PLAN.md` — the joint-head audit that found the
0.86 bits/frame factorization cost is this program's method applied to
the action space. Note the G1+G2 "combined read" below ("what we lack
is a selection rule") was later sharpened by that audit: part of the
selection gap was the independent-head factorization itself (8a SIGNAL,
08-30).

Started 2026-08-26, the morning fox_gen_v1's decode story broke open
(argmax = absorbing crouch loop; temp 0.5 = live repertoire + 1.0
armed approaches/min + 3/4 conversions vs CPU). Companion to
DIRECTIONS_2026-08-25 (D1/D2) and the closed specialist interp program
(INTERP_ROADMAP.md, P0-P6). Subject checkpoint:
`checkpoints/fox_gen_v1_20260825_210355_ep*.bin` (296-wide, GRU-60).

**The organizing principle (inherited, twice-proven):** behavior is
invisible to loss, and interp converts decode-time and training-time
knobs from search problems into measurement problems. Measure first;
the decision is then usually obvious.

Each instrument below: what it measures → the decision it informs →
tooling status. Ordered by (decision value x cheapness).

---

## G1. Entropy-by-situation map → derived temperature schedule

**Measures:** per-head (buttons, main_x/y, c_x/y, shoulder) output
entropy of the policy, conditioned on the 47 `ExPhil.Situations`
labels, over a sample of corpus states (teacher-forced histories).

**Decisions informed:**
- Global temperature stops being eyeballed: the entropy profile says
  where the model is legitimately multi-modal (sample) vs confident
  (sharpen).
- Per-head temperature split: if button entropy << stick entropy, a
  single global T is provably the wrong shape.
- State-adaptive T (v2 of decode): schedule T from the model's own
  entropy — the "adaptive scaling" idea grounded in data.

**Tooling:** policy forward pass exposes logits (margin-export
machinery from the early-reject program); Situations labeler is
batch-capable. New glue: one script that walks corpus windows, labels
them, and accumulates per-(label, head) entropy stats.
**Status:** [x] done (2026-08-26) — see results ledger.

## G2. History-vs-state dominance probe → does v2 train with scheduled sampling

**Measures:** exposure-bias severity directly. Same game state, two
synthetic 60-frame histories (self-crouch loop vs normal movement);
measure the action-distribution shift (KL / top-action flip rate)
attributable to the policy's own history channel. Sweep over a set of
states for a population number.

**Decisions informed:**
- If history dominates state → absorbing basins are structural →
  **fox_gen_v2 trains with scheduled sampling** (the SS-on-queue
  move that broke the specialist delay campaign open, applied to the
  generalist's history window).
- Quantifies how much decode temperature is compensating for a
  training-time defect.

**Tooling:** Data embedding path can build synthetic windows; policy
predict on {1, 60, 296}. New script, small.
**Status:** [x] done (2026-08-26) — see results ledger (verdict: v2 does
NOT need scheduled sampling).

## G3. CycleSim closed-loop basin study → offline decode tuning

**Measures:** in the offline closed-loop simulator (no Dolphin), the
escape rate from the crouch/idle basin as a function of temperature /
decode strategy; steps-to-escape distributions; basin inventory
(what OTHER absorbers exist beyond crouch?).

**Decisions informed:** decode-strategy tuning at seconds-per-
experiment (temperature brackets, sticky sampling, nucleus) without
burning live sessions; regression harness for future checkpoints.

**Tooling:** CycleSim exists (gate passed, specialist-era). Adaptation
needed: generalist state coverage (it was built around multishine
cycles). **Status:** [dropped 2026-08-31] — CycleSim is a transition
table scraped from fixture replays (exact on visited states, dead
off-graph, positions frozen); a generalist roams the whole state space,
so the adaptation has no viable scope. Redefined as **G3b** below.

## G3b. Learned dynamics model → generalist offline rollouts (successor to G3)

**Measures / provides:** a one-step dynamics model
`state' = f(state, action)` trained on the banked corpus (~90M frames;
the AR-head streamed-capture infra already emits exactly the needed
(state, action, next-state) tensors). Gives the critic follow-on its
rollout engine (CRITIC_D2_DESIGN NULL branch: V + short rollouts over
`sample_autoregressive_n` candidates) and re-enables basin studies at
seconds-per-experiment.

**Fidelity ladder (decided 2026-08-31, platfighter question):** learned
model for breadth, audited by mechanics-true headless Dolphin
spot-checks — the missing "start from an arbitrary state" piece is the
parked improoover thread (.slp moment → bootable .gci savestate).
`~/git/platfighter` is NOT a Melee simulator by design (original game,
no hitboxes yet) and is not coupled here; its `crates/sim` architecture
is the template if a hand-built MeleeSim is ever justified — only after
the learned model measurably fails on fidelity. Validation gate mirrors
CycleSim's: offline/live parity, measured on the BOT'S OWN rollout
distribution, not the corpus's.

**Status:** [ ] spike owed — GATED on the 0831_legS_ar reading (if AR
headroom collapses, the selector program is demoted and this stays
parked; if headroom survives, this is the next build).

## G4. Linear probes on the trunk → is the BC-then-RL bet sound

**Measures:** what the GRU trunk linearly encodes: opponent percent,
offstage-ness (own + opponent), frame advantage, kill-percent
proximity, stage identity, **stage internals (FoD heights / PS
transform)** — the W4 question re-asked of a model that actually
trained with --stage-internals wired.

**Decisions informed:**
- RL fine-tuning (D2) can only cheaply sharpen decisions over features
  the trunk represents. Rich readout → the BC prior is fertile; go.
  Poor readout → fix representation at training time first (the P4
  lesson: no drill policy read techs).
- Whether --stage-internals earned permanence (default-on for v2?).

**Tooling:** P4 probe methodology + Inspect.moment; trunk activations
need a capture hook (margin-export pattern). **Status:** [x] done
(2026-08-28) — see results ledger.

## G5. Style separability (SAE / clustering) → is OGSwaglord cheap

**Measures:** do different players' segments separate in trunk space
(linear probe for player identity / SAE feature analysis on trunk
activations over per-player corpora)?

**Decisions informed:** if identity is already separable, style
conditioning (D4, OGSwaglord) is nearly free — add the conditioning
pathway to an existing representation. If blended, D4 needs
player-token training (learn_player_styles) from scratch in v2.

**Tooling:** cross-arch SAE exists (specialist program); player
registry provides labels. **Status:** [ ] deferred until G1-G4 read.

## G6. High-entropy pockets → v2 curation targets

**Measures:** per-situation val loss / entropy after 10 epochs — the
pockets where the model stayed uncertain (thin or contradictory data).

**Decisions informed:** v2's oversampling mix, aimed by measurement
(the P5-validated loop) instead of by watching failures live. Also
feeds the fight-state program: pressure situations are prime suspects.

**Tooling:** G1's script gets this nearly for free (same walk, add
loss accumulation). **Status:** [x] absorbed into G1's findings
(2026-08-26): curation target #1 = defensive/disadvantage states;
target #2 = punish continuation (finishing, not entering). Feeds the
v1.1/v2 recipe.

## G7. Blind input audit → what does it actually read

**Measures:** output divergence under single-family input
perturbations (opponent character id, percents, stocks, stage id) —
the audit-game harness generalized.

**Decisions informed:** converts "it seems to react to X" into yes/no;
prioritizes which conditioning inputs are dead weight vs load-bearing
(a dead opponent-character channel would matter for matchup work).

**Tooling:** blind auditor ready (audit round 2 infrastructure).
**Status:** [x] done (2026-08-27) — see results ledger.

---

## Execution order

*(Historical — ran in roughly this order 08-26→08-28; only G3/G5 remain.)*

1. **G1 + G6** (one script, one corpus walk) — improves today's
   decode knob; yields curation targets.
2. **G2** — decides v2's scheduled-sampling question.
3. **G7** — cheap audit while G3/G4 spikes are open.
4. **G4** — the RL-readiness verdict, before D2 resourcing.
5. **G3** — CycleSim adaptation spike (scope-check first).
6. **G5** — when D4 becomes live.

Standing constraint: all of these run offline on the GPU — NO-MIX law
applies against any live training/eval beam; schedule around sweeps.

## Results ledger

### G1 — entropy map, ep10, 12 files, 15,259 sampled frames (2026-08-26)
Full table: `eval_runs/0826_gen_v1_sweep/entropy_map.txt`. Nats; uniform
refs buttons(8 Bernoulli)=5.55, 17-bucket=2.83, 5-bucket=1.61.

Baseline `__all__`: buttons 3.85, main_x 1.36, main_y 1.48, c_x 0.71,
c_y 0.74, shoulder 0.67.

**Findings and the decisions they make:**
1. **The heads live in completely different entropy regimes.** Buttons
   sit at 3.85/5.55 = **69% of uniform** — the model is genuinely
   unsure which buttons to press. Sticks sit at 1.36/2.83 = **48%**,
   c-stick at **25%**, shoulder at **41%**. A single global temperature
   is therefore the WRONG SHAPE: T=0.5 that tames the button tail
   over-sharpens the c-stick, and T that frees movement makes buttons
   chaotic. → **per-head temperature is not an optimization, it's a
   correction.** First v2-decode change; buttons want the coldest T.
2. **Entropy tracks game-theoretic reality, which is a soundness
   check.** Lowest-entropy states are the ones with a correct answer:
   `edge_danger` (buttons 3.16), `tumble` (3.39), `shine_cancellable`
   (buttons 3.76 but main_y **0.91** — the lowest stick entropy in the
   table, i.e. the model KNOWS which way to hold on a shine-cancel).
   Highest are genuine mixups: `pummel_throw_decision` (4.37),
   `being_tech_chased` (4.26), `walltech_available` (4.18),
   `shield_pressure_theirs` (4.17, and shoulder 1.03 = the highest
   shield-head entropy anywhere — correct: that IS the shield mixup).
   The model has *situation-appropriate uncertainty*, which is what a
   population-BC model should have.
3. **disadvantage (4.14) / in_hitstun (4.13) >> advantage (3.75) /
   combo_active (3.79).** The model is decisive when ahead and
   uncertain when behind — consistent with masters' DI/escape choices
   being genuinely mixed, but ALSO the signature of thin, high-variance
   supervision on defense. → **G6 curation target #1: defensive/
   disadvantage states for the v2 mix.**
4. `neutral` (3.75) is BELOW baseline while `conversion_open` (3.92) is
   above — the model is more certain in neutral than mid-conversion.
   That inverts the drill-era failure (which was passive in neutral and
   fine in punish) and is consistent with the sweep's 35% conversion /
   9-13 dropped-punish counts: **the generalist's weakness is finishing,
   not entering.** → aim D2's value model and v2 curation at punish
   continuation, not at approach.

### G2 — history-vs-state dominance, ep10, n=256 windows (2026-08-26)
`HISTORY/CURRENT ratio = 0.391` (prefix-swap KL 0.733 mean / 0.44
median; last-frame swap KL 1.877 / 1.27). Frozen-tile stay-mass:
idle-tiled 0.202 vs active-tiled 0.208 — **no idle stickiness**.

**Verdict: exposure bias is NOT structural in this model.** The policy
is current-state-dominated (last frame moves the distribution ~2.6x
more than 50 frames of history), and a frozen/idle history creates no
preference for continuing to idle. Two consequences:
1. **v2 does NOT need scheduled sampling.** The SS-on-queue move that
   broke the specialist's delay campaign open would be solving a
   problem this model doesn't have. Budget it elsewhere. (Re-run this
   probe on any v2 that changes the history window or backbone.)
2. **The argmax crouch-loop is therefore a DECODE pathology, not a
   memory pathology** — it's the mode-collapse of a multi-modal
   distribution (G1 finding 1: buttons at 69% of uniform entropy), not
   the model conditioning on its own idleness. This is exactly why
   T=0.5 fixed it live while argmax could not, and it re-points the fix
   at decode strategy (per-head T, nucleus, Best-of-N with a value
   model) rather than at retraining.

**Combined read (G1+G2):** the model contains the behavior, holds
appropriate uncertainty, and does not self-trap. What we lack is a
*selection rule* over its distribution. That is a decode-and-value
problem — i.e. D2's value model is the highest-leverage next
investment, and it now has interp evidence behind it rather than
argument alone.

### G7 — blind input audit, ep10, 5 files, 7,035 windows (2026-08-27)
Full table: `eval_runs/0826_gen_v1_sweep/blind_audit.txt`. Per-head mean
|logit delta| vs baseline; "ratio" = sum / opp_x_far (positive control).

| perturbation | sum | ratio | flip% | reading |
|---|---|---|---|---|
| opp_x_far (control) | 1.32 | 1.00 | 7.0 | position is the dominant input (sanity: audit works) |
| opp_char_swap | 0.084 | **0.06** | 0.4 | **DEAD — opponent character channel is not read** |
| stage_swap | 0.53 | 0.40 | 2.7 | load-bearing (stage identity matters) |
| own_percent_hi | 0.41 | 0.31 | 2.5 | load-bearing |
| opp_percent_hi | 0.30 | 0.23 | 1.5 | read, weaker than own percent |
| opp_stock_last | 0.18 | 0.14 | 0.9 | weak |
| own_stock_last | 0.15 | 0.11 | 0.7 | weak |

**Findings and the decisions they make:**
1. **The opponent-character channel is effectively dead** (0.06× the
   position signal; 0.4% action-flip). Whether the learned character
   embedding collapsed or the trunk down-weighted it, fox_gen_v1 does
   NOT behaviorally distinguish matchups. Hard result for matchup work
   (D12 low-tier) and for "the bot plays Marth differently from Fox":
   the conditioning exists but carries no load. Fix the representation at
   training time (G4 says whether the trunk even encodes it), or accept
   that v1 is matchup-agnostic.
2. **Stage is load-bearing (0.40×)** — second-strongest input after
   position. `--stage-internals` earned its keep (feeds G4's permanence
   question).
3. **Own percent (0.31×) > opponent percent (0.23×)** — the model tracks
   its own damage more than the opponent's; kill-confirm-on-opponent
   behavior is thinner than self-preservation.
4. **Stocks are the weakest read of all** (own 0.11×, opp 0.14×) —
   last-stock behavior is nearly invisible to the output. Candidates for
   v2 curation emphasis alongside G1's defensive/disadvantage pocket.

**Caveat:** single-family mean-|logit-delta| is a first-pass sensitivity
measure; a "dead" channel here means "moves the marginal output little",
not "the trunk has no linear encoding of it" (that is G4's question).

### G4 — linear probes on the trunk, ep10, 40 files, 46,348 train rows (2026-08-28)
Full table: `eval_runs/0826_gen_v1_sweep/g4_RESULTS.md`. Balanced accuracy of a
linear probe on the GRU trunk's hidden state vs the raw embedding (input floor).

**The RL-readiness verdict, split in two:**
- **Fertile:** hitstun (own 0.924 / opp 0.912) and offstage (own 0.933 / opp
  0.827) are richly, linearly encoded — the punish/edgeguard signals a value
  model (D2) needs are there to sharpen. → the BC prior is fertile for the
  core interaction game.
- **Gaps:** percent (opp 0.524 / own 0.459 vs input ~0.8–0.9) and stage
  identity (0.489 vs 1.000) are ~half-discarded; opponent character (0.332 ≈
  majority) is fully discarded. → kill-confirm, recovery-routing, and matchup
  awareness are NOT readable from the trunk by a linear value head.

**The G7 disambiguation, answered:** the raw embedding DOES carry opponent
character (input 0.460, weak) and the corpus DOES have variation (Fox vs 7+
matchups, never a ditto) — so the dead character channel is NOT a collapsed
embedding or a data gap; it dies **in the trunk**. Matchup-awareness is a
representation fix (character must bypass/survive the trunk), not a decode or
data fix.

**Caveats:** linear-only (stage is used non-linearly per G7); the 12-file
stage=0.009 was a small-eval fluke; input floor needs ≥40 files to be reliable
(raw-ID scale is ill-conditioned).

