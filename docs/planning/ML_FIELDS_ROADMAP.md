# ML Fields Roadmap — what else transfers like interp did

**Status doc — created 2026-08-04.** Companion to
[INTERP_ROADMAP_V2.md](INTERP_ROADMAP_V2.md); same organizing rule:
**every field earns a slot only via a named current customer.**

## Why interp worked here (the transfer test)

Interp paid off because (1) complete ground truth makes every claim
falsifiable; (2) it converts Dolphin-time questions into GPU-seconds
questions; (3) the bottlenecks were diagnostic, not capacity. A field
transfers if it has the same profile: methodology that turns existing
artifacts (activations, replays, checkpoints, deterministic evals) into
verdicts at GPU-minutes cost, validatable against ground truth or a
behavioral A/B. Fields ranked below by that test.

## F1 — Evaluation science / sequential statistics

**Customer:** promote_check (uncalibrated thresholds, one human
datapoint per policy); the ms_g6_sp1 crowning was an EVALUATION
failure, not a training failure. Dolphin game time is the scarcest
resource in the lab.

- [ ] **Formalize CRN paired evals**: FD/BF determinism = common random
      numbers; write the paired-comparison protocol down (same seeds /
      stages / dummy scripts across candidates) so every A/B gets the
      variance reduction on purpose instead of by habit.
- [ ] **Sequential stopping**: run evals until the verdict is
      significant, not for a fixed game count. A blowout should cost 1
      game, not 3+. (Sequential probability ratio test or a simple
      Bayesian stopping rule — the math is one page.)
- [ ] **Racing / successive halving for checkpoint selection**: 1 cheap
      game across all candidates, survivors get more. Replaces "3+
      Dolphin games x every checkpoint" (the original v1 motivation!)
      with a budget that concentrates on the frontier.
- [ ] **Calibrate promote_check thresholds** as data accumulates: each
      human session adds labeled (dummy-score, human-score) pairs;
      maintain the scatter, fit the discriminator when n allows.

Success gate: a written eval protocol where the expected Dolphin-games
cost per promotion decision drops measurably (target: half), with the
g6-inversion class of error given an explicit false-positive budget.
Cost: near-zero compute; pure methodology.

## F2 — Uncertainty quantification + OOD detection

**Customer:** the absorber (task #34) — mechanistically an
out-of-distribution event (policy walks into uncovered states, behavior
becomes uncontrolled; LEACE-live showed the identical failure shape).
Second customer: the dummy-vs-human inversion (human-game states are
far-OOD from training states → predictable failure).

- [ ] **OOD scalar on trunk activations**: Mahalanobis distance to the
      training activation distribution (or kNN / energy score). Cheap:
      fit on existing captures. Floor-test per GOTCHA #79.
- [ ] Validate on the YS contrastive pair: does the OOD score rise at
      absorber ENTRY? (Second instrument for W1, independent angle
      from margin probes.)
- [ ] Validate on the human replays: are ms_g6_sp1's human-game states
      further OOD from ITS training distribution than g4's are from
      g4's? If yes → OOD distance joins W2 as a promote_check rung.
- [ ] Later: ensemble disagreement (2-3 cheap heads) as a second
      uncertainty signal; conformal-style calibrated abstention if a
      live fallback controller ever exists.

Success gate: one OOD score that separates absorbed-from-good YS runs
at entry AND ranks g6 < g4 on human-state coverage. Same-activations
composition with the interp stack — no new capture infrastructure.

## F3 — Continual learning / catastrophic forgetting

**Customer:** the curation loop's binding constraint. Cycle 1 destroying
core skill (380 -> 72.9) is textbook catastrophic forgetting; snippet
dosing (cycle 2) is rehearsal — the field's oldest remedy, rediscovered
empirically. The field has sharper tools that could RAISE the safe dose.

- [ ] **KL-distillation anchor**: old policy (g4) as teacher on clean
      frames during retrain; new data trains free, core skill is pinned
      by the anchor rather than by data composition. Likely the
      biggest single lever; also the exact machinery RL (#29) needs.
- [ ] **Adapter/LoRA-style specialist delta**: train the fight-state
      fix as a small delta on frozen g4 weights. Directly answers the
      cycle-3 P3-branch question ("does this need a separate specialist
      checkpoint?") with a third option: same checkpoint, switchable
      delta.
- [ ] **EWC-style weight anchoring**: use interp to identify which
      weights carry the cycle (attribution already exists), penalize
      their movement. The interp-flavored variant of the same idea.
- [ ] A/B against snippet dosing at equal compute: does any of these
      beat the current rehearsal recipe on the stand-gate +
      pressure-gain pareto?

Success gate: one retrain where the stand gate holds at a data dose
that previously broke it (cycle-1's composition), or an adapter that
switches between core and fight-state behavior without a full retrain.

## F4 — Data-centric AI / influence estimation

**Customer:** the P5 loop (already data-centric AI, informally) and the
70-minute retrain cost. W3 (projection filtering) is the entry point;
the formal versions go further.

- [ ] **Influence scoring** (TracIn-style: grad-dot between training
      frames and a target behavior) — "which training frames CAUSED the
      absorber" is the removal-side complement to W3's addition-side
      filtering. Validate any claim by actually retraining (ground
      truth available — the field's hardest problem is free here).
- [ ] **Minimal-set experiments**: what's the smallest replay subset
      reproducing g4-level skill? Every 10x reduction is a 10x faster
      iteration loop. Cycle 3's 20k-human-frames-vs-800k-synthetic
      question is the first datapoint.
- [ ] Duplicate/near-duplicate audit of the drill pool (frame-level
      redundancy is likely enormous in cycling data — dedup may shrink
      training cost outright).

Success gate: one documented removal (influence-flagged frames deleted,
retrain, behavior delta as predicted) or one minimal-set result that
halves retrain time at equal skill.

## F5 — Offline RL (the bridge to #29)

**Customer:** RL staging (#29, parked) and the 5090's emulator-throughput
ceiling. Middle rung between BC and online RL: learn better-than-the-
data policies from replays already on disk, at BC compute cost, no
Dolphin in the loop. Replays carry full reward signal (stocks, damage)
that BC ignores.

- [ ] **Advantage-weighted BC** first (simplest: weight imitation loss
      by observed outcome advantage). One flag on the existing trainer,
      conceptually adjacent to conversion-weighting (#35) which already
      validated outcome-weighted data.
- [ ] Return/outcome conditioning (Decision-Transformer-style "win
      token") as the second experiment; the DT backbone already exists
      in the zoo.
- [ ] IQL-style value learning only if the cheap variants show signal —
      it also produces the value function online RL needs anyway.
- [ ] Deferred until: cycle-3 arc concludes and a human-replay corpus
      of usable size exists (offline RL amplifies data quality issues;
      GOTCHA #84 discipline applies).

Success gate: an advantage-weighted retrain that beats plain BC on the
same data at the deploy rung (chains, promote_check), demonstrating
policy improvement beyond imitation without an emulator.

## Explicitly weak fits (recorded so they stay parked)

- **Learned world models** (Dreamer-class): CycleSim covers offline
  simulation for cycling; Dolphin is a free perfect simulator. GPU-weeks
  for a worse copy of what exists. Revisit only if planning-at-inference
  becomes a thread.
- **Opponent modeling / population-based training**: real, but needs
  environment throughput this machine doesn't have; parks with #29.
- **Domain adaptation theory**: the dummy->human gap IS a domain gap,
  but the practical lever is data (human corpus) not adaptation
  machinery; F2's OOD measurement covers the diagnostic need.
- **Architecture search / scaling**: explicitly deprioritized
  (CLAUDE.md); architecture is not the bottleneck.

## Sequencing vs INTERP_ROADMAP_V2

F1 is free-standing (no compute, immediate). F2 composes with W1/W2 and
should ride along with them. F3 fires when cycle 3's verdict lands (its
branches decide whether the anchor/adapter question is live). F4 extends
W3. F5 waits for the cycle arc + human corpus. Nothing here displaces
the V2 workstreams; F1-F2 are the ones worth starting this week.

## Experiment log

| Date | Field | Experiment | Result |
|---|---|---|---|
| | | *(append as they land)* | |
