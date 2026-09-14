# Recorded recovery context and teacher validation

## Completed

- Validated a recorded teacher on the three human-hit handoffs (502, 1389,
  3152), with P2 neutralized after handoff. All three restart: strict cycle
  counts 39/36/37, maximum chains 40/37/38, no deaths or prefix drift.
- All 1,080 issued inputs match readback; all 1,074 delay-2 targets match.
  Case 502 has no off-loop targets, so the final validation explicitly uses
  `--allow-on-loop`. This removes only the off-loop coverage requirement, not
  command or label validation. Cases 1389/3152 provide 31/23 off-loop targets.
- Added input-only recorded prefixes, preserving every early recovery target.
  Cold and warm versions of nine clips give 3,564 targets and 162 context-only
  frames. Warm prefixes are 18 frames: window 16 minus one plus queue depth 3.
- Audited the first 18 supervised windows per clip against the actual Agent:
  324 checks pass, maximum absolute difference 1.1920928955078125e-7 under
  tolerance 1e-6. Sources are replay-hash checked before export.
- Fixed two discovered live/batch normalization mismatches: negative game
  countdown frames and negative hitstun/action-frame counters. Live embedding
  now uses the same lower bounds as batch embedding. Added regression tests.
- Rechecked the unchanged checkpoint with those fixes: warm controls still
  pass 6/12. Neutral/sustain pass; recovery 4/75/146 fail both repetitions,
  with maximum chains 1/0/0. No timing failures, errors, truncation or drift.
- Focused regression suite: 16 tests, zero failures. No training this round.

## Artifacts and reproduction

Root: `eval_runs/0913_context_recovery/`.

- `teacher/eval_candidate.json`: live teacher responses.
- `teacher/quality.json`: strict replay cycle scores.
- `teacher/validation_with_on_loop.json`: accepted input/target validation.
- `clips_v6/report.json`: completed window audit and source hashes.
- `clips_v6/*.frames`: the only ready cold/warm exports; earlier clip directories
  are incomplete failed attempts and must not be included in a training glob.
- `warm_recheck/eval_candidate.json`: unchanged-model control.
- `tests_final.log`: focused regression result.
- `bash eval_runs/0913_context_recovery/prepare.sh NEW_DIRECTORY`: reproduce
  exports and window audit on CPU using source-loaded modules.

Teacher execution uses native issued-input latency 1; exported labels are
verified at delay 2. Window parity uses parsed action frames and committed
teacher history. This does not prove behavior under arbitrary human pending
inputs or general offstage recovery. No history reset on hits is introduced.

## Next bounded fitting protocol

1. Freeze the canonical pool plus these 18 clips, replacing rather than duplicating
   the old six cold recordings. Keep F32, zero-state windows, delay 2 and no
   previous-action dropout. Preserve the current playable checkpoint.
2. Extend per-case fit measurement to distinguish cold/warm clips and measure
   the first 18 *supervised* targets, not the input-only prefix. Require both
   early and whole-response fit before claiming a fitting success.
3. Freeze an update budget before training: this larger pool has 10,641 targets;
   weighting each recorded clip's first 18 targets by 64 gives 31,053 draws,
   or 486 batches per epoch at batch size 64. Nine epochs would be 4,374
   updates; blindly repeating 21 epochs would exceed the previous budget.
4. Validate the candidate with cold and warm familiar controls, then the three
   isolated interruptions and the original replay-opponent interruptions.
   Keep strict-cycle, readback, drift and truncation gates unchanged.

The finite cold-start proof already passes. The remaining objective is robust
restart with real history, not merely lower aggregate training loss.
