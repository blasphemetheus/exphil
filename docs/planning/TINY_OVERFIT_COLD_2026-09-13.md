# Boundary-safe, cold-history tiny overfit

Predeclared 2026-09-13, before training.

- Same GRU64, window16, autoregressive head, clean loss, no dropout recipe.
- Reaction delay 2, delay ID 2, previous action and queue depth 3.
- Same 7,079 canonical frames and six validated teacher windows (720 frames).
- Boundary-safe batching: 7,785 targets, repeat-first padding and queue resets.
- Initialize from the previous experiment's saved **untrained** `initial.bin`,
  not its trained candidate; fresh optimizer. Save this run's initialization.
- Exactly the same maximum 40 epochs; target-loss early stop disabled. Retain
  all snapshots; evaluate the normal best-loss export, not a live-selected epoch.
- Six fixed training-state handoffs, two runs each, temperature 1.0, reaction 2.
- Use `--prefix-history cold` and the corrected pipe_v2 input contract. Physical
  prefix replay and pending delivery are unchanged. No delay-ID override.
- Evaluate the saved untrained initialization under the same conditions.
- Pass requires all 12 trained responses: zero errors/drift, complete 120-frame
  responses, valid input verification, maximum chain at least 10.
- No automatic extra epochs, snapshot fishing, or sweep. No old-label comparator
  in this run: the objective is a harness/memorization proof, not label causality.
- This does not prove continuously warmed recovery or held-out generalization.

Artifacts and scripts: `eval_runs/0913_tiny_overfit_cold/`. Original experiment
artifacts are preserved. No shared native rebuild or overlapping GPU jobs.

## Results

Training completed all 40 epochs, exit 0, peak RSS 2.2 GB. Minimum logged loss
is 0.0000034773893 at epoch 38; epoch 40 loss is 0.0029444338. **These are final
minibatch losses, not epoch averages** (see metric defect below). The normal
best-loss export remains the candidate, as predeclared. The CLI's rounded
`Converged: loss=0.0` message is not an exact-zero loss or a behavioral result.

The saved initial parameter trees compare equal to the previous experiment's
untrained initialization (`initial_comparison_verified.log`); file hashes differ
because the exports are not byte-identical. An initial CPU comparison attempt
without selecting BinaryBackend failed to load tensors; the explicit CPU retry
passes.

The trained candidate passes **7/12**, failing the required 12/12 gate.
All 12 runs have valid pipe_v2 input timing, zero prefix drift/errors, and
complete 120-frame responses. No scores are withheld for harness invalidity.

| Handoff | Trained run 1 chain | Trained run 2 chain |
| --- | ---: | ---: |
| Neutral start 2228 | 1 | 1 |
| Neutral start 2566 | 13 | 1 |
| Sustain 900 | 13 | 14 |
| Recovery 4 | 12 | 6 |
| Recovery 75 | 7 | 13 |
| Recovery 146 | 13 | 13 |

Training completes successfully; the candidate evaluation script exits 1 from
the behavioral assertion, not a timing/transport exception. The untrained
control completes with **0/12 behavioral passes**, also with valid timing,
zero drift/errors, and complete responses in all 12 runs. Its script exits 1
from the same behavioral assertion. No checkpoint is promoted.

Thus all 24 responses pass the scoped harness-validity checks, but the trained
bot still fails reliability. The 7/12 versus 0/12 result is a descriptive result
on training-state revisits, not a generalization claim. Machine-readable summary:
`eval_runs/0913_tiny_overfit_cold/summary.json`.

Both shell scripts pass syntax checks, captured source hashes match, and
working-tree whitespace checks pass. No training/evaluation process remains
running. The shutdown child's exit-1 message in the training log is not the
training process status (which is 0).

## Newly confirmed metric defect

In `scripts/dagger_drill.exs`, the normal epoch reduce replaces the previous
loss with `metrics.loss` for every batch. The forensics path also returns the
final batch's loss. That scalar is then logged as epoch loss and used for
best-checkpoint selection and target-loss stopping.

Consequently, a tiny printed loss does **not** demonstrate that the full pool,
or the rare off-loop targets, was overfit. Candidate selection depends on the
last shuffled minibatch. The earlier interpretation of tiny loss as successful
dataset fitting was too strong. This defect is confirmed from the executed code;
it is not proof of the sole cause of the remaining live failures.

The predeclared run is retained unchanged: no alternate snapshot selection or
extra training after seeing scores. Next fix should aggregate a correctly
weighted epoch metric and, separately, evaluate a fixed full-pool/per-handoff
teacher-forced loss and action accuracy at the exported checkpoint. Then repeat
the same bounded closed-loop proof. Do not simply increase the epoch budget.
