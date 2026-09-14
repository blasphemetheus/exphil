# Epoch metric correction

## Fixed

The drill's normal and forensics paths now accumulate all batch objectives via
`ExPhil.Training.EpochLoss`, instead of returning only the final minibatch loss.
Autoregressive imitation loss divides by the sum of frame weights, so the epoch
aggregate uses that same denominator. Policy families that ignore frame weights
use batch row counts. Partial final batches no longer receive disproportionate
weight. Empty epochs, invalid loss/mass, and aggregate overflow fail closed;
a later healthy batch cannot erase an earlier invalid loss.

The aggregate feeds logging, best-checkpoint selection, target-loss stopping,
and plateau tracking. Logs identify the metric, batch count, and denominator
mass. Resume fingerprints version the change. The final message no longer
rounds tiny loss to zero or calls every epoch-cap termination "Converged".

This is a weighted average of **training-step objectives while parameters are
changing**, not a frozen checkpoint's full-pool loss. Batch-local normalization
and auxiliary regularizers retain their batch-local meaning. A separate fixed
checkpoint evaluation remains necessary to claim the dataset was memorized.

## Validation

- Eleven metric/numerical-health tests pass: unequal batch sizes, fractional
  weights, regrouping, unweighted/lazy batches, invalid/empty epochs, overflow,
  and ranking/stopping despite a misleading tiny final batch.
- GPU smoke artifacts: `eval_runs/0913_epoch_metric_validation/`.
- Normal and `--nan-forensics` one-epoch checks both complete and export, exit 0:
  122 batches and mass 7,785 each, aggregate losses approximately 7.3754 and
  7.3767 respectively. Peak RSS 2.1 GB. These optimizer smoke exports are not
  promoted models or a new behavioral experiment; native dependencies were
  not rebuilt. The inhibitor's shutdown child prints exit 1, but each main
  training process exits 0.
- The initial normal smoke reached a valid aggregate (122 batches, mass 7,785),
  then exposed an existing optional-flag bug: `opts[:snapshot_all] and ...`
  raises when the option is omitted. This is tracked separately, not fixed by
  the metric patch. The bounded proof recipe already supplies `--snapshot-all`;
  the smoke checks use that same explicit setting on retry.
- No new 40-epoch run or behavioral promotion is part of this correction.
- Shell syntax, Elixir parsing, and working-tree whitespace checks pass.

## What remains before a reliable bot

1. **Measure actual fit at a fixed checkpoint.** Report teacher-forced loss and
   button/action accuracy separately for neutral starts, sustain, and each
   recovery handoff. Rare off-loop targets can be hidden by a good global mean.
2. **Locate the first remaining live disagreement.** If those targets are not
   learned, address sampling/coverage or capacity using measured errors. If
   they are learned offline, compare exact live observations, decision history,
   and in-flight commands at the first divergence instead of adding epochs.
3. **Pass the scoped closed-loop proof.** The last candidate passed 7/12 versus
   0/12 untrained, with valid harness checks throughout; require 12/12 for the
   fixed cold-history start/sustain/recovery gate before promotion.
4. **Prove continuously running behavior.** Test warmed recurrent history and
   held-out/disturbed starts rather than only training-state cold restarts.
   Current recovery cases concern regaining the multishine rhythm onstage;
   offstage return-to-stage recovery would require its own examples and gate.

Known scope limits remain: flat-frame callers/streaming shards have not all been
migrated to boundary-safe histories, and unrelated online timing configurations
are not certified by the synchronous scenario harness. None of those alternate
paths is required for the current narrow proof recipe.
