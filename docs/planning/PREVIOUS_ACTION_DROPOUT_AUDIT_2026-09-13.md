# Remaining misses and training-only previous-action dropout

Follow-up completed: [bounded no-dropout fit](NO_DROPOUT_FIT_2026-09-13.md)
fixes the recovery miss and passes12/12 local responses; canonical frame0 still
fails. The proposed experiment below is now complete, not awaiting execution.

Audited the **same** zero-state/F32 candidate:
`c250c1b4e1f572364fbfe1d84ca488d52f8eea8484eebd8cef733d7d88cac5ef`.
No optimizer updates, new checkpoint selection, or live run.

## Findings

Both misses incorrectly press X when the target releases all buttons and holds
the main stick down. Other controller components are correct.

| Case | Observation | P(X), unmasked | Correct target probability |
|---|---|---:|---:|
| canonical index0 | frame0, action361/af2 | 0.927712 | 0.072267 |
| recovery4 index6 | frame10, action363/af17 | 0.610294 | 0.389704 |

The small recovery probability difference from the original report is consistent
with the previously documented GPU numerical variation; both evaluations fail.
Each unaugmented window has only its own exact match among all7,785 pool targets:
no exact input/label conflict was found for either miss (`probe_v2.json`).

### Canonical miss: dropout cannot directly change this input

At the cold start, all three previous-action queue slots are already absent and
the history window repeats the first observation. Across32 seeded 10% dropout
masks, 0% dropout with two different seeds, and even100% dropout, this window is
**byte-identical**. Its prediction is unchanged and wrong in every case.
Therefore the canonical miss cannot be explained by dropout altering this input.
An indirect effect on learned weights is not ruled out.

### Recovery miss: strong sensitivity, not proof of historical causation

- Default drill dropout is0.1; fixed teacher-fit evaluation uses0.0.
- `Data.precompute_frame_embeddings/2` independently removes available queue slots
  once. Those corrupted embeddings are cached and reused across all epochs.
  Repeating an early target x64 repeats the **same** corrupted window, not64 masks.
- Across32 reproducibly seeded 10% masks, the recovery target is correct in4/32.
  Its joint target probability ranges **0.000122–0.999823**.
- Removing every queue slot changes25 input coordinates and makes it worse:
  P(X)=0.999683. Removing input history wholesale is not the fix.
- The zero-dropout windows are byte-identical across seeds, and every variant
  keeps labels unchanged.

The original cached training mask/RNG state was not saved, so these are
**counterfactual masks**, not a reconstruction of which mask the model trained on.
We have evidence of input-distribution mismatch and sensitivity, not proof that
dropout caused this particular training failure.

## Next controlled experiment

For the minimal memorization proof, disable **augmentation**, not previous-action
inputs: keep `--prev-action --queue-depth 3`, add `--prev-action-dropout 0.0`.
Use the saved fresh matching initialization from
`eval_runs/0913_zero_f32_fit/round21/initial.bin`, a fresh optimizer, and the same
21-epoch budget, dataset, prefix weights, and frozen gates. No distillation and
no threshold relaxation. This experiment has **not** been started.

If the canonical cold-start target remains wrong, examine its fit/weight separately:
it is not one of the recorded-teacher targets given64x weight. Do not assume that
disabling dropout alone fixes both misses.

## Coverage and artifacts

- `scripts/measure_teacher_fit.exs --include-canonical-rows` exposes canonical
  misses without changing the default compact report or readiness criteria.
- `scripts/audit_previous_action_dropout.exs` compares actual cached input windows,
  checks zero-dropout seed independence and unchanged targets, and measures
  checkpoint predictions under seeded masks.
- Two focused data regression tests pass. The broader66-test data suite has one
  unrelated process-dictionary `frames_array` cache assertion failure; initial
  `--no-start` execution additionally lacked EXLA services, so it was rerun with
  the application started. See logs rather than treating the broad suite as green.
- Evidence: `eval_runs/0913_dropout_audit/fit.json`, `probe_v2.json`, and test logs.
  Reproduce with `bash eval_runs/0913_dropout_audit/run.sh NEW_OUTPUT_DIRECTORY`.
