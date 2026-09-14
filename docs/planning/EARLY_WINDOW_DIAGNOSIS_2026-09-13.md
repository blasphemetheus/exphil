# Three early-window failures: input and inference diagnosis

## Result

- [x] Reconstruct the complete 7,785-target training pool and isolated teacher clips.
- [x] Compare the three failed windows and search the full pool for exact duplicates.
- [x] Compare F32 evaluation, BF16 input-only, and reconstructed training precision.
- [x] Test batch size/row dependence using identical copies of each failed input.
- [x] Isolate random recurrent initial state with a graph-only zero-state counterfactual.
- [ ] Implement a versioned deterministic recurrent-state contract with regressions.
- [ ] Make compute/input precision explicit across training, export, evaluation and live sampling.
- [ ] Re-measure fitting under the corrected contract before another training budget.

No training, checkpoint changes, gate relaxation, or Dolphin evaluation.
Candidate remains the ordinary epoch20 export of the 21-epoch run.

## Inputs and targets

The complete-pool and isolated per-case window tensors are **bit-identical**
for all three failed targets (max difference 0). These are actual loader windows,
with the same delay-2 labels, queue3, window16 and cold repeat-first padding.
Each window's hash appears exactly once among the 7,785 unique training targets:
**no identical-input conflicting labels were found for these three windows**.
This does not prove there are no ambiguous near-neighbors or conflicts elsewhere.

The nearest windows by unscaled mean squared embedding distance are adjacent
states in the same clip (recovery4 index1; sustain900 index0/index1). Their
different actions can be legitimate temporal decisions; this distance alone
does not establish a label error. Exact target identities remain:
- recovery4 index0: X;
- sustain900 index1: B without X;
- sustain900 index3: B+X.

## Confirmed recurrent initial-state defect

The policy's shared `Edifice.Recurrent` builder calls Axon's GRU overload without an
explicit hidden state. In the installed Axon source, this creates a **random
Glorot initial hidden state**, rather than zeros. Its shape includes the batch
dimension. For fixed stored RNG keys, changing batch size or row changes the
hidden state that precedes otherwise identical observations.

This is not dropout. Training itself builds its cached predictor in inference
mode; dropout is zero in this recipe. Shuffled training moves a target between
batch rows, whereas the per-case evaluator uses fixed rows and live inference
uses batch size 1. Resetting the observation/committed-action buffer does not
make the internal recurrent state zero.

Direct test: 64 identical copies of each input and teacher-forcing target,
same checkpoint parameters and F32 graph:

| Window | Legacy P(X) range over 64 identical rows | Rows pressing X | Zero-state P(X), all 64 rows |
|---|---|---:|---:|
| recovery4 index0 | 0.1912–0.5417 | 5/64 | 0.3564 |
| sustain900 index1 | 0.8849–0.9848 | 64/64 | 0.9595 |
| sustain900 index3 | 0.1000–0.3351 | 0/64 | 0.1678 |

The counterfactual replaces only the two random initial-state graph nodes with
zeros; no parameters or files are changed. Within-batch spread becomes exactly
zero on all three cases, isolating the initial-state contribution. Batch1 versus
batch64 still has smaller numerical probability differences, approximately
0.0020–0.0084; full cross-shape numerical identity is not claimed.

**Zero state does not repair the existing model's three wrong decisions.**
This is a correctness defect to fix before interpreting another overfit proof,
not evidence that the current checkpoint can be promoted by flipping a switch.

## Additional precision mismatch

The drill leaves Imitation's default compute precision at BF16. Its training
loss casts inputs to BF16, and its Axon model applies mixed precision (F32
parameters, BF16 compute, F32 outputs, excluding normalization layers).
The policy export omits this precision metadata. The ordinary evaluator and
Agent reconstruct the uncast F32 model. Thus they do not execute the exact
training arithmetic even with identical parameters and input histories.

Reconstructing that training precision on the fixed checkpoint does **not**
make the three errors disappear:

| Window | F32 teacher-action probability | BF16 training-precision probability |
|---|---:|---:|
| recovery4 index0 | 0.4769 | 0.3122 |
| sustain900 index1 | 0.0326 | 0.0311 |
| sustain900 index3 | 0.1717 | 0.1127 |

All three remain conditional argmax errors. The six teacher cases still have
early match counts 18,18,17,18,18,16 in report order. The precision mismatch
is real, but is **not a sufficient explanation of failed fitting**.
Numbers here are within this diagnostic reconstruction; do not interpret tiny
differences from prior runs as optimization progress.

The live autoregressive sampler separately executes its head from parameters;
merely casting the trunk or stamping a metadata field is not sufficient to
claim end-to-end precision parity.

## Next order

1. Explicit zero recurrent initial states for new windowed GRU training and
   inference, with a versioned export contract and explicit legacy behavior
   for old checkpoints. Do not silently reinterpret all existing policies.
   Add identical-row, row permutation, batch1/batch64 and train/export/reload
   regressions with numerical tolerances. Retain pretrained weight names.
2. Declare precision in checkpoints and centralize its reconstruction. Either
   train this small proof in F32 to match deployment, or implement and verify
   BF16 throughout the sequential autoregressive sampler as well as the trunk.
3. Recheck these windows under that contract. If still underfit, inspect their
   frozen gradients/loss contributions and representation, not just add epochs.
   Preserve the original 0.95 early confidence and matched-teacher live gates.

## Evidence and reproduction

`eval_runs/0913_early_window_diagnosis/report.json` contains exact-match and
nearest-neighbor searches, pooled/isolated input comparisons, batch variants,
and all six BF16 teacher-fit summaries. `gru_state.json` contains every repeated
row's probability and the zero-state counterfactual. Production modules were
not modified by this inspection.

The first `gru_state.log` run stopped at an overly strict cross-batch tolerance
before saving its report. The corrected diagnostic records rather than hides
residual differences; its successful log is `gru_state_v2.log`. The original
failure is retained. The successful report confirms zero within-batch spread,
not perfect cross-batch equality.

Reproduce: `bash eval_runs/0913_early_window_diagnosis/run.sh NEW_DIRECTORY`.
The runner refuses existing output and an already-running BEAM. No native rebuild.

Key source seams: `lib/exphil/networks/recurrent.ex` (`build_recurrent_layer`),
`deps/axon/lib/axon.ex` (`gru/3`, `rnn_state`),
`lib/exphil/training/imitation.ex` (precision policy and inference-mode predictor),
`lib/exphil/training/imitation/loss.ex` (input precision cast), and
`lib/exphil/training/imitation/checkpoint.ex` (export config).
