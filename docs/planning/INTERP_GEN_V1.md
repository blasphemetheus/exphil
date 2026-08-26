# INTERP_GEN_V1 — interpretability program for the generalist line

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
**Status:** [ ] script owed (G1 first — directly improves today's knob)

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
**Status:** [ ] script owed (G2 second — decides v2's biggest knob)

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
cycles). **Status:** [ ] adaptation spike owed; scope unknown until
opened.

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
need a capture hook (margin-export pattern). **Status:** [ ] probe
harness adaptation owed.

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
loss accumulation). **Status:** [ ] rides on G1.

## G7. Blind input audit → what does it actually read

**Measures:** output divergence under single-family input
perturbations (opponent character id, percents, stocks, stage id) —
the audit-game harness generalized.

**Decisions informed:** converts "it seems to react to X" into yes/no;
prioritizes which conditioning inputs are dead weight vs load-bearing
(a dead opponent-character channel would matter for matchup work).

**Tooling:** blind auditor ready (audit round 2 infrastructure).
**Status:** [ ] cheap, run after G1/G2.

---

## Execution order

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

(append findings here as instruments run)
