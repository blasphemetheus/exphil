# Targeted cold-prefix fitting preparation

Follow-up: [21-epoch round completed](EARLY_PREFIX_ROUND21_2026-09-13.md).
Teacher matches improve to 705/708, but sustain and recovery4 fail frozen
readiness; live evaluation is withheld. The preparation status below is historical.

## Status

- [x] Add opt-in recorded-teacher prefix sampling to the drill.
- [x] Verify label preservation and actual boundary-safe sampled windows.
- [x] Audit the complete proof pool: every target retained, no crossed clips.
- [x] Add a frozen early-case confidence/readiness gate and test failure paths.
- [x] Confirm the existing candidate fails the new gate; no false success claim.
- [x] GPU preflight: one disposable train step, numerical checks and exports pass.
- [ ] Authorize/run the next bounded experiment; no multi-epoch run started here.
- [ ] Frozen fit gate, then the unchanged matched-teacher live gate.

## Sampling recipe

New flags: `--recorded-prefix-weight 64 --recorded-prefix-frames 18`.
Both are positive integers; the weight is a **copy count**, not a loss multiplier.
Without the flags the existing training path is unchanged. If omitted, prefix
length derives from `window + queue_depth - 1`, which is 18 for this proof.

Only clips loaded through `--recorded-frames` are marked. Labels are shifted
as before; marks remain attached to target observation indices. The canonical
fixture is not marked. Repeat the first 18 targets of every executed teacher
clip, including neutral starts as a retention/confidence control. Do not select
just the failed runs or omit easy cases after seeing evaluation results.

18 covers the cold temporal/committed-action-history startup region. The twelve
known conditional argmax errors are within its first six frames. Repetition
uses the existing window sampler: full histories are preserved; clips are not
shortened, relabeled, concatenated into fake histories, or restarted at each
sample. Other sampling boosts combine by maximum, not multiplication.

Actual full-pool audit (`eval_runs/0913_early_prefix_fit/sampling.json`):

| Quantity | Unweighted | Prefix x64 |
|---|---:|---:|
| Unique supervised targets | 7,785 | 7,785 |
| Early teacher targets | 108 | 108 |
| Early teacher draws | 108 | 6,912 |
| Other draws | 7,677 | 7,677 |
| Total draws per epoch | 7,785 | 14,589 |
| Early fraction | 1.39% | 47.38% |
| Batch-64 steps per epoch | 122 | 228 |
| Crossed-clip windows | 0 | 0 |

Each of the six cases supplies 1,152 early draws. Every other target occurs
exactly once per epoch. The audit checks actual emitted window counts, not
only desired weights. Warmup/decay step configuration now accounts for the
expanded epoch when this feature is enabled. Resume fingerprint includes the
versioned prefix-copy rule and length, preventing optimizer resume across a
silent recipe change.

## Frozen readiness gate

`scripts/check_early_teacher_fit.exs` consumes the unweighted output of
`scripts/measure_teacher_fit.exs`, not the resampled training objective. It
checks the report's SHA-256 against the actual checkpoint before gating.

Require all six named cases and all 118 distinct target indices per case.
For each case's first 18 targets, require:
- 18/18 teacher-forced conditional all-component argmax matches;
- teacher-action joint probability at least **0.95 on every early target**.

The probability threshold is a predeclared readiness filter, not a guarantee
of successful sampled control. AR analog logits are teacher-forced; this gate
does not measure free-running joint-decoder accuracy. Review the full-pool
and later-target metrics too; do not mask forgetting with the early score.
The same 12-response matched-teacher chain-10 gate remains mandatory afterward.

Existing candidate baseline (unchanged checkpoint):

| Case | Early argmax matches | Minimum teacher-action probability |
|---|---:|---:|
| Neutral 2228 | 18/18 | 0.8212 |
| Neutral 2566 | 18/18 | 0.8209 |
| Sustain 900 | 14/18 | 0.1392 |
| Recovery 4 | 14/18 | 0.1626 |
| Recovery 75 | 16/18 | 0.1753 |
| Recovery 146 | 16/18 | 0.1759 |

The baseline gate correctly exits 1. Weighting is implemented and validated;
**improved model fitting has not yet been demonstrated**.

## Preflight and next bounded run

`bash eval_runs/0913_early_prefix_fit/run_preflight.sh` completed with main exit
0: one JIT train step, finite loss/parameters, policy export and trainer snapshot
all pass; peak RSS approximately 2.0 GB. Temporary exports are removed. The
inhibitor child prints its known shutdown warning. This is not another fitting
experiment and does not replace the existing candidate.

The next run should use the same saved initial parameter tree and fresh optimizer,
unchanged GRU/head/window/delay/labels and clean loss, with the new prefix flags.
Account for the extra optimizer steps: **21 weighted epochs = 4,788 updates**,
close to but below the previous 40 x 122 = 4,880 budget. Do not silently call
40 weighted epochs the same compute budget (that would be 9,120 updates).
The preflight's max-epoch setting is not consumed because it exits after one
disposable step. No multi-epoch launcher has been run.

Predeclare checkpoint selection using corrected aggregate loss, then evaluate
that one checkpoint with frozen per-case metrics and the readiness gate. Do not
search saved epoch snapshots after observing live results. The old candidate
was selected with the former terminal-minibatch metric, so this is not a pure
single-variable causal comparison. If fitting still fails, inspect per-target
conflicts and model-input collisions instead of extending the budget blindly.

## Validation and reproduction

18 focused tests pass (four new sampling/gate regressions plus existing
epoch/health/snapshot/teacher-likelihood tests). Artifacts live under
`eval_runs/0913_early_prefix_fit/`, including sampling audit, baseline gate,
preflight and test logs, and final source/input checksums.

No native rebuild, model promotion, or full training run. Existing live proof
results remain in `eval_runs/0913_matched_handoffs/` and are unchanged.
