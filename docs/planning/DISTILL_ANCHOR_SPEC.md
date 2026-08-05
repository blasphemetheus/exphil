# F3 spec — KL-distillation anchor + adapter route for the curation loop

**Written 2026-08-04 during the cycle-3b window (ML_FIELDS_ROADMAP F3).
Implementation-ready; do not implement until no training beam is live.**

## Problem

The curation loop's binding constraint is catastrophic forgetting: cycle 1
(whole rollouts) destroyed the core skill (380 -> 72.9); cycle 2's snippet
dosing (rehearsal) protects it but caps how much new-distribution data a
retrain can absorb — and the pressure gain vanished at the safe dose. We
are trading skill acquisition against skill retention through DATA
COMPOSITION alone. The field's sharper tools pin retention in the LOSS or
the PARAMETERS instead, freeing the data budget for the new skill.

Trigger: cycle-3b's verdict. P3 makes this the main road; P2 makes it a
lever to raise the snippet dose; even P1 leaves it useful for cycle 5+.

## Route A — KL-distillation anchor (primary; implement first)

Anchor the student to the FROZEN production policy's outputs on
clean-cycle data, while new (snippet/human) data trains unanchored.

### Key design choice: precompute teacher logits, never run the teacher in-graph

The teacher (g4) is frozen — its logits per frame are constants. Running
a second network inside the training graph doubles compile surface and
walks straight into the closure-tensor gotcha (#3). Instead:

1. After pool assembly (dagger_drill.exs ~line 595, where
   `fixture_frame_lists ++ bc ++ rollout ++ snippet_frame_lists` concat),
   run the teacher's `predict_fn` over the ANCHOR SUBSET's embedded
   windows once (inference beam, same machinery as
   `Activations.load_heads` + the probe scripts' chunked loop) and store
   the 6-head logits per frame.
2. Attach as extra per-frame targets (same mechanism as precomputed
   embeddings; RAM: 6 heads x ~(8+17+17+17+17+5) logits x f16 ≈ 160B/frame
   — negligible).

### Anchor mask

Distill ONLY on clean-cycle frames: fixture + rollout frames get
`distill: true`, snippet/human frames `distill: false` (tag at the same
concat point — provenance is only knowable there). Rationale: anchoring
on the new distribution would fight the very update we want; the mask is
what makes this "retain core, learn new" rather than "stay g4".

### Loss

```
total = imitation_loss
      + distill_w * mean_over(distill_mask) [
          sum_buttons  KL(Bernoulli(sig(t_i)) || Bernoulli(sig(s_i)))
        + sum_softmax_heads KL(softmax(t/τ) || softmax(s/τ))
        ]
```

- τ = 1 initially (the teacher IS the target behavior, no need to soften).
- Wire into `Loss.build_loss_fn(policy_model, opts)` as
  `distill_weight` + per-batch `teacher_logits`/`distill_mask` tensors
  threaded like existing batch fields. Respect the ±60 logit clamp
  BEFORE softmax/sigmoid (the NaN lesson lives there).
- CLI: `--distill-from checkpoints/ms_g4_d2mix.bin --distill-weight W`.
  Teacher checkpoint must have the same embed layout (assert
  `embed_size` match at load; g4-lineage all 336).

### Pre-registered A/B (cycle 4)

Same aligned human snippets, same recipe, three arms at equal compute:
  A1 snippet-dose baseline (cycle-3b's recipe as-is)
  A2 + distill_w=0.5 anchor, snippet fraction UNCHANGED
  A3 + distill_w=0.5 anchor, snippet fraction DOUBLED (the actual point:
     does the anchor let us absorb more new data safely?)
Gates: stand d3 chains (deterministic, 1 run) must hold within 10% of
g4's 423 for A2/A3; win = A3 holds stand AND beats A1 on the YS
collapse-rate bucket / AbsorberEntry count.

## Route B — adapter/specialist delta (secondary; the P3 road)

Freeze all g4 params; train only a low-rank additive delta.

- Where: the trunk's input projection + the 6 head input matrices
  (rank r=8-16). GRU recurrent kernels stay frozen (cheap, and keeps
  the cycle dynamics intact by construction).
- Axon: `Axon.namespace`d delta layers summed with frozen base weights;
  freezing via the existing `dagger_drill_freeze.exs` machinery
  (already does param freezing — check its `frozen` handling first,
  reuse over reinvent).
- Deployment semantics: one checkpoint + a ~100KB delta file; the agent
  can hot-swap "core" vs "fight-state" mode — this IS the specialist-
  checkpoint answer of prereg P3, without shipping two full policies.
- A/B: adapter-trained fight-state fix vs full-retrain fix, both gated
  on stand d3 (adapter should hold it near-perfectly by construction —
  that's its selling point; the question is whether the delta has
  CAPACITY for the new skill).

## Order of work (post-verdict)

1. P2 or P1: implement Route A loss plumbing (~30 min: Loss opts, drill
   flags, teacher-logit precompute) + run the A1/A2/A3 prereg.
2. P3: implement Route B first (it directly answers the branch), Route A
   second as the comparison arm.
3. Either way: record verdicts in ML_FIELDS_ROADMAP F3's experiment log.
