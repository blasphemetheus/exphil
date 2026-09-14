# Recorded teacher training

Executed teacher rollouts have recorded futures. They are not student rollouts
that should be relabeled with projected expert commands.

## Export verified windows

`scripts/check_recovery_targets.exs` checks the issued teacher commands against
the run's Slippi recording and checks delayed training targets against future
issued commands. Only complete, passing, non-diverged teacher runs qualify.

```bash
mix run scripts/check_recovery_targets.exs \
  --scores teacher_scores.json --delay 2 --allow-on-loop \
  --out NEW_report.json --export NEW_teacher.frames
```

`--allow-on-loop` admits sustain-only controls; without it every run must include
off-loop targets. It does not disable timing or target checks. Both outputs are
exclusive-create. The export contains only the teacher-controlled response
windows, not scripted replay prefixes, and is **unshifted** despite validation
at the requested delay. Replay hashes and the validation report accompany it.

## Recorded futures are the ONLY delayed recovery labels (2026-09-13)

At a training shift k > 0 the scripted expert supervises only states on
its canonical loop: `MultishineExpert.label_ahead/4` abstains (`:skip`) off
the loop and `Labels.at_delay/3` drops those frames (`off_loop: :drop`, the
default). The old rule held the current recovery input at every k and was
measured wrong on 18/21 recovery frames at shift 4 against the teacher's
executed futures. `dagger_drill.exs --off-loop-labels hold` (and
`audit_ms_pool_labels.exs --off-loop hold`) reproduce the legacy rule for an
equal-budget comparison only. Every recovery a policy must learn at a delay
therefore has to be a recorded teacher clip ingested as below.

## Ingest without relabeling

Pass `--recorded-frames 'PATH/*.frames'` to `scripts/dagger_drill.exs`, separately
from `--snippet-frames`. The latter intentionally treats mined snippets as
expert-relabeled and must not be used for these recordings.

The recorded loader requires causal convention, zero existing label delay,
explicit `:recorded` provenance, nonempty contiguous lists, and no
`:prev_controller` override. Episode boundaries survive loading and shifting.
The eager-ingestion drill now uses `Data.from_frame_lists/2`, which preserves
clip boundaries and frame-number gaps through lazy temporal batching. Every
early target is retained: short histories repeat the first clip embedding,
matching the live agent's cold-start padding. Padding adds inputs, not labels.
Previous-action queues reset at boundaries, even when two clips happen to have
consecutive frame numbers. Sampling/loss weights, teacher masks, and probe
labels use the actual supervised-frame mapping.

This is a cold-start training convention, **not reconstructed replay-prefix
history**. A committed-prefix live handoff still needs a fresh behavioral gate.
Existing flat-frame callers and streaming shards are not automatically migrated
or certified. Changed drill resume fingerprints reject pre-fix trainer resumes.
See [experiment and batching correction](../planning/TINY_OVERFIT_2026-09-13.md).
Targets are shifted exactly once at the configured training delay; previous
committed actions are constructed by the normal training path. These lists are
not automatically marked as clean-cycle distillation anchors.

Currently this option supports eager training only. Streaming combinations and
unmatched globs fail explicitly rather than silently omitting recordings.
Resume fingerprints include recorded-file content hashes and the new guard
version; old trainer snapshots must not silently resume under changed semantics.

## Input-only recorded history

Recorded envelopes may mark a contiguous prefix with `input_only: true`.
These frames populate history and previous-action queues but never become
supervised targets. Each clip must end with a nonempty supervised suffix.
Use `RecordedContext.slice/4` to select actual contiguous replay frames; missing
history fails explicitly. Apply the normal label-delay shift after slicing.

For the window-16, queue-3 proof, warm clips retain 18 pre-handoff frames.
Boundary-aware lazy temporal batching keeps the first response target, measures
stride from that target, and maps sampling weights to supervised indices.
Early-prefix weighting counts response targets, not context frames. Single-frame
batching and eager temporal batching reject context rather than training on it.
This is in-memory lazy window construction, not support for streaming shards.

See [validated context experiment](../planning/RECORDED_CONTEXT_2026-09-13.md).

## Epoch acceptance

The old absolute-loss/100x-drop rejection is replaced by numerical validation:
loss must be numeric and policy parameter tensors must be nonempty and finite.
Healthy tiny losses are legitimate, including zero. Regularized finite negative
objectives are not automatically numerical failures either.

Invalid numerical epochs cannot become best candidates or normal snapshots;
the existing bounded restore/halt path remains. This does **not** detect every
finite behavioral collapse. Best-loss exports are candidates, not promoted bots;
closed-loop start/sustain/recovery gates and recorded-input timing remain required.

## Current evidence

`eval_runs/0913_teacher_ingestion/validated/` contains two neutral-start windows,
three recovery windows, and one sustain window: 720 causal frames, 708 targets
at reaction delay 2. All recorded input and target checks pass. Standing starts
begin in grounded action 14 and each produces a chain of 14 with no prefix drift.

`eval_runs/0913_teacher_ingestion/run_preflight.sh NEW_NAME` rehearses ingestion,
embedding, one optimizer step, numerical validation, and checkpoint serialization
using prebuilt dependencies. This is not an overfit run or a trained-bot proof.
Only `validated/*.frames` uses the final safe-decoding envelope; initial exports
outside that directory are retained as failed-prototype evidence.
