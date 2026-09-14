# Same-initialization no-dropout bounded fit

Authorized follow-up to the previous-action dropout audit. Only the augmentation
setting changes: `--prev-action-dropout 0.0`. Keep previous-action inputs, queue3,
zero-state/F32 GRU64 x2, AR head, window16, delay2, all7,785 targets and prefix64x
weighting. Start from `0913_zero_f32_fit/round21/initial.bin`, not trained weights,
with a fresh optimizer. Shuffling is not a paired-seed replay of the earlier run.

Budget:21 epochs,228 batches/epoch,4,788 updates. Select ordinary best aggregate
loss. Preserve six-case confidence gate and matched-teacher12-response live gate.
No budget extension, threshold relaxation, or checkpoint fishing.

Launcher: `bash eval_runs/0913_no_dropout_fit/run.sh`.
Artifacts: `eval_runs/0913_no_dropout_fit/round21/`.

- [x] Finish bounded fitting and verify matching initialization.
- [x] Measure selected checkpoint, including canonical rows, and apply frozen gate.
- [x] Run live validation only if qualified; record outcome and next blocker.

## Outcome: frozen local proof passes

All21 epochs completed (4,788 updates). Ordinary best aggregate export selects
epoch20, loss **0.00007980429945778855**. Peak RSS1.9GB. Loaded initialization
parameter trees compare **exactly equal** to the previous zero-state/F32 initial
checkpoint; differing serialized file hashes do not indicate different weights.

Offline recorded teacher targets: **708/708 correct**, early **108/108**.
All six confidence gates pass; worst early joint target probability **0.997332**.
The previous recovery4 index6 miss is resolved. Canonical remains **7,076/7,077**:
the cold frame0 X-release miss persists (target probability0.036739). The frozen
gate covers the six recorded cases, not perfect canonical memorization; no gate
was changed or relaxed to permit live evaluation.

The **same selected checkpoint** passes **12/12** matched-teacher local responses:

| Handoff | Two observed maximum chains |
|---|---|
| neutral2228 | 14,14 |
| neutral2566 | 14,14 |
| sustain900 | 13,14 |
| recovery4 | 13,13 |
| recovery75 | 13,13 |
| recovery146 | 13,13 |

Every response has120 observed frames, valid input timing, no truncation,
no replay divergence, and no run errors. Each records13 cycles,14 shine entries,
and zero empty hops. Runtime uses libmelee action-frame convention, reaction
delay2, cold history, and temperature1.0. Expected cold-start warnings reflect
the deliberately cold contract; they are not unexpected missing-history errors.

Candidate: `eval_runs/0913_no_dropout_fit/round21/candidate.bin`

SHA256: `404a1f543889ddd4193380745fbc603c6c94bb4a200a3caa7cbca6ddf823182a`

Evidence: `initial_comparison.log`, `fit.json`, `fit_gate.json`,
`eval_candidate.json`, live traces/replays, and `checkpoints.sha256` in the run
directory. Launcher exits0 with its original strict live assertion satisfied.

## What this establishes / next

Local human session: `bash scripts/play_multishine_proof.sh` from the repository.
Bot is Fox on port1, human GameCube adapter on port2. The wrapper source-loads the
current zero-state builder and corrected libmelee action-frame conversion, selects
reaction-delay2 and temperature1.0, and records replays in a timestamped directory.
This human/async session is a broader test, not the already-passed scenario suite.

The corrected pipeline can train repeated multishining and the recorded
reentry/recovery responses, then reproduce them closed-loop in local Dolphin.
This is a **finite, trained-handoff proof**, not reliable arbitrary gameplay or
general offstage recovery. No model was automatically installed as a default.

Next: keep this checkpoint frozen and test longer local episodes plus held-out
starts/perturbations. Retain the canonical cold-frame miss as a known limitation;
do not add distillation or more optimization before measuring broader behavior.
The no-dropout comparison supports disabling augmentation for the tiny proof,
but is not a paired-seed causal experiment (shuffle RNG was not replayed).
