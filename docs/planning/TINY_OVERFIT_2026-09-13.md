# Tiny overfit experiment

Predeclared before training: 2026-09-13.

- GRU, hidden size 64, window 16, autoregressive head, clean loss, no dropout.
- One trained reaction delay: 2. Previous-action conditioning and queue depth 3.
- Canonical fixture (7,079 frames) plus six validated teacher windows (720 frames).
- Maximum 40 epochs, target-loss stopping disabled, all epoch snapshots retained.
- Primary candidate: normal best-loss export, selected before live evaluation.
- Save the initial parameters; evaluate those as an untrained control.
- Old projected-label comparator: same initialization and epoch budget, same
  teacher states with tails trimmed to match corrected target count, routed
  through the existing expert snippet path. This compares target/history
  recipes, not a perfectly isolated single-label intervention.
- Fixed evaluation: two standing starts, one sustain handoff, three recoveries;
  two runs each at temperature 1.0, reaction delay 2, no manual delay-ID override.
- Prefix history uses committed decisions, matching delayed recorded training
  targets; physically replayed prefix inputs are unchanged. Fixed before evaluation.
- Pass: every run has zero prefix divergence/errors, a complete 120-frame
  response, verified recorded input timing, and maximum chain at least 10.
- Do not call low loss success. Do not extend training or select epochs by
  these test outcomes without recording a separate exploratory experiment.

These are training-state revisits, deliberately testing memorization and the
harness—not held-out generalization. No existing checkpoint is overwritten.
Artifacts and scripts: `eval_runs/0913_tiny_overfit/`.

## Results

The corrected recipe completed all 40 epochs and exported the normal best-loss
candidate. Training loss reached approximately 0.0001; this was **not a
behavioral pass**. No checkpoint was promoted and no evaluation-based snapshot
selection, delay tuning, or extra epochs were performed.

All 12 responses contain 120 frames with zero prefix divergence/errors. One
response fails recorded-input timing (119/120 sent commands match); its score
is withheld. Only 4 of the 11 timing-valid responses meet the chain-10 gate.

| Handoff | Run 1 maximum chain | Run 2 maximum chain |
| --- | ---: | ---: |
| Neutral start 2228 | Withheld: timing invalid | 3 |
| Neutral start 2566 | 2 | 1 |
| Sustain 900 | 14 | 14 |
| Recovery 4 | 1 | 1 |
| Recovery 75 | 11 | 12 |
| Recovery 146 | 1 | 1 |

Evidence: `corrected.log`, `eval_corrected.json`, replay recordings, and all
epoch snapshots under the artifact directory.

The saved-initialization control also completed 12 full responses with zero
prefix divergence/errors, but **all 12 fail recorded-input verification**.
Its behavioral scores are withheld; this is not a valid trained-versus-untrained
comparison. Both evaluation processes exit 2 as required by the timing gate.
Offline raw-replay audits (`timing_initial.json`, `timing_corrected.json`) find
no fully matching offset in the tested 0–8-frame range for the invalid runs.
This does not establish an extra-frame delay: controller representation or
delivery mismatches still need diagnosis, especially for untrained outputs.

Validation: both shell scripts pass `bash -n`; the drill and experiment Elixir
scripts parse successfully; captured source/checkpoint hashes match; working
tree whitespace checks pass. Seven focused recorded-loader/numerical-guard
tests pass. No shared native dependency rebuild was performed.

## Confirmed batching defect

`audit_windows.exs` exercises the actual lazy `Data.batched_sequences` path
with provenance-encoded embeddings and the experiment's real frame lists.
`window_audit.json` reports 7,785 shifted frames and 7,770 sequences:

- 90 windows cross a clip boundary (six boundaries times 15 preceding frames).
- All 29 teacher targets classified as off-loop by this training-state audit
  receive preceding context from an unrelated clip.

These targets are **supervised with wrong history**, not absent. The recorded
loader and label shifting preserve boundaries, but the drill flattens the lists
and the lazy temporal sampler does not enforce them. Teacher-future validation
remains valid; it did not test recurrent input context. The 29 count uses the
training-state classifier, not the earlier issuance-audit classification.

This is a concrete training defect and a plausible contributor to the failed
starts/recoveries, not proof that it is their sole cause.

## Protocol amendment and next experiment

The projected/HOLD comparator was prepared (`held.frames`,
`recipe_comparison.json`) but training/evaluation are deferred after discovering
the shared sampler defect. It contains the same 708 target states, 16 changed
target controllers, and 708 explicit previous-controller overrides. No
equal-budget comparison or isolated label-effect claim is available.

1. Fix temporal batching to respect episode boundaries **without discarding
   early start/recovery targets**. Simply removing crossing windows would omit
   precisely those targets. Supply real input-only pre-handoff context with
   masked prefix loss, or explicitly validated padding/reset semantics.
2. Add a regression that checks actual emitted window provenance and retained
   early-target coverage, not just label shifting.
3. Diagnose recorded-input mismatches without relaxing the timing gate.
4. Record a separate corrected-context experiment and repeat the fixed budget,
   untrained control, and start/sustain/recovery gate before any broader sweep.

## Follow-up: boundary histories corrected

The eager-ingestion drill now retains list boundaries via
`Data.from_frame_lists/2`. Lazy windows stop at clip boundaries and frame gaps;
early windows repeat their first embedding, matching Agent's cold-start
padding. Every real target is supervised once under uniform stride-1 sampling;
padding has no extra target/loss rows. Action queues reset at boundaries.
Sampling/loss weights, distillation masks, and probe labels remain target-aligned.
The changed recipe is included in the resume fingerprint.

The real-data audit (`window_audit_boundary_safe.json`) reports **7,785
sequences, zero cross-clip windows, all 29 off-loop targets retained, and zero
off-loop targets with wrong-clip context**. Run it explicitly with
`audit_windows.exs --boundary-safe --out NEW_REPORT.json`; the original audit
and failed experiment remain preserved.

This correction uses cold-start padding, not missing pre-handoff replay frames.
It fixes cross-clip contamination but does not prove equivalence to a warmed
committed-prefix evaluation. No new 40-epoch run or behavioral promotion is
implied. Input-verification mismatches remain a separate unresolved issue.

Validation: 66 data/history/weight tests pass (six normally excluded benchmark
tests excluded). An exploratory benchmark-inclusive run hits the existing
flat-batching process-dictionary cache assertion; that path was not changed.
GPU optimizer/export preflight passes in 20 seconds with 2 GB peak RSS:
`eval_runs/0913_teacher_ingestion/clip_boundary_preflight.log`. No native rebuild.
