# F3 spec — KL-distillation anchor + adapter route for the curation loop

**Written 2026-08-04 during the cycle-3b window (ML_FIELDS_ROADMAP F3).
Implementation-ready; do not implement until no training beam is live.**

> **ROUTE A IMPLEMENTED 2026-08-13** (same batch-field insertion point
> as the F5/AWBC plumbing, as the specs anticipated). What shipped:
>
> - `Policy.Loss.distill_kl/4` — masked Bernoulli+softmax KL, f32 +
>   ±60 min/max clamp per the NaN lessons, head sizes from student
>   logits' static shapes, teacher as one concatenated [batch, 81]
>   tensor. Tested (6 cases incl. clamp + mask-selection).
> - Loss builder: `distill_weight > 0` builds a 6-arg loss
>   (`teacher_logits`, `distill_mask` per batch); mutually exclusive
>   with probe-reg by explicit raise.
> - Batch plumbing: teacher rows precomputed PER SEQUENCE (same
>   indexing as embeddings, f16 storage ~160B/frame as specced),
>   gathered by seq idx in `Data.batched_sequences(distill_teacher:,
>   distill_mask:)`; batches lacking the fields fall back to zero mask
>   (KL exactly 0).
> - `dagger_drill.exs --distill-from CKPT [--distill-weight 0.5]
>   [--distill-tau 1.0]` — teacher via `Activations.load_heads`,
>   embed-size asserted, provenance mask per spec (fixture/rollout/
>   opening/y-aug = anchor true; bc/snippet = false), stamped through
>   the multi-delay expansion.
> - Smoke (fixture pool, teacher ms_awbc_b1): 21,166 teacher rows
>   (3.4MB f16), all-clean anchored, trains + exports; KL term verified
>   live (epoch-1 loss 0.215 @ w=0.5 vs 17.70 @ w=50).
>
> The A1/A2/A3 prereg below remains UNRUN — scheduled after the g16
> (champion + AWBC) arc so recipe changes stay one-at-a-time. Route B
> (adapter) remains unimplemented.

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

### Pre-registered A/B v2 (2026-08-19 — SUPERSEDES the cycle-4 prereg below)

Written while g17 (LR retune) runs; launch AFTER the retune verdict
settles the stage-3 base ("one recipe change at a time"). Run script:
`eval_runs/0819_f3_distill/run_f3_arms.sh` (fill BASE_LR + refs from
the g17/90-epoch winner at launch; fallback base = g16 @ 2e-4,
refs fox 253.6/min c203, mewtwo 109.8/min c14).

**Question**: does a KL anchor to the frozen base checkpoint (on
clean-cycle frames only) let the recipe absorb MORE human-snippet data
without losing the core skill — the thing data composition alone could
not do (cycle-1 collapse, convC3 dilution)?

**Teacher** = the retuned base checkpoint itself (fixed-grad stack,
same embed layout by construction). NOT ms_g15: old-grad-era weights
as a distill target would conflate the retune question with the anchor
question. "g15-as-teacher" is a registered follow-up only.

**Arms** (all = base recipe incl. --awbc; training is from scratch each
arm, anchor pins the fresh student to the base's outputs):
- **A1** = the base itself (already run — zero GPU; reuse its gates).
- **A2** = + `--distill-from BASE --distill-weight 0.5 --distill-tau
  1.0`, snippet dose UNCHANGED. Measures the anchor's own cost.
- **A3** = anchor as A2, snippet dose DOUBLED (`--snippet-frames
  "X.frames,X.frames"` — comma-glob preserves duplicates). The point.
- **A4** (control; run ONLY if A3 holds G1) = doubled dose, NO anchor.
  Attribution arm: if A4 drops G1 where A3 held it, the anchor did it.

**Gates** (sync headless, per eval_live_protocol):
- G1 stand-fox d3 x3: A2/A3 must hold >= 0.90 x base G1.
- G2 stand-mewtwo d3 x1: >= base (no collapse; chains reported).
- Behavior: analyze_behavior on gate replays — SD/death counts,
  empty-hop breaks (the human snippets' skill surface).

**Decision rules** (registered before launch):
- A2 fails G1 -> anchor costs core skill at w=0.5; ONE fallback arm at
  w=0.25 before abandoning Route A at full scale.
- A3 holds G1 AND G2 >= base -> anchor unlocks dose; run A4 for
  attribution; F3 graduates to a standing recipe lever.
- A3 holds G1 but readouts flat -> anchor safe but pointless at 2x;
  ONE escalation arm at 4x dose before verdict.
- A3 fails G1 -> anchor insufficient for dose escalation; Route B
  (adapter) becomes the F3 road.

**Registered watch items** (record, don't gate):
- KL term magnitude per epoch (smoke ref: 0.215 @ w=0.5 epoch 1).
- AWBC x distill interaction: both concentrate on clean-cycle frames;
  the anchor does NOT protect the "no shine ahead" bucket AWBC starves
  (0.15 mean weight) because that bucket is also anchored to the
  base's already-starved behavior. If non-shine skills matter, that is
  a MIX-share fix, not this anchor — do not read A2/A3 as evidence
  either way.
- Teacher-row RAM at full scale (~380k anchored frames x 160B ~ 61MB
  f16 — expected fine; abort read if precompute walltime dominates).

### Pre-registered A/B (cycle 4) — SUPERSEDED by v2 above (kept for record)

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
