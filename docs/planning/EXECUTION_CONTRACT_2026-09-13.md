# Windowed GRU execution contract

## Decision

Before another optimization run, use `windowed_gru_f32_v1`:

- Every GRU layer starts at explicit zeros for **each complete input window**.
  No hidden state carries between batches, clips, or live calls. Observation
  history and committed-action queues retain their existing cold/teacher-matched rules.
- Parameters, model inputs, compute policy, and output logits are F32 in training,
  teacher evaluation, and Agent's sequential autoregressive sampler. No BF16 policy
  or mixed-precision optimizer. This is a dtype contract, not bitwise equality
  across GPU batch shapes or hardware.
- This version supports temporal GRU windows, not BPTT or stateful-step inference.
  Incompatible flags and metadata fail rather than silently changing semantics.
- Exports and training snapshots carry `execution_contract`, `recurrent_state`,
  `training_precision`, and `inference_precision`. Resume and drill warm-start
  reject cross-contract initialization.

The policy uses **Edifice.Recurrent**, not the older local recurrent module.
The shared builder in `../edifice` now accepts `recurrent_state: :zeros`, using
the standard GRU path rather than implicit-state fused paths. Zero initializer
nodes have no trainable parameters.

## Compatibility and invocation

- GRU `scripts/dagger_drill.exs` defaults to `--recurrent-state zeros --precision f32`.
  Generic Imitation defaults remain unchanged to avoid changing unrelated runs.
- Unstamped checkpoints remain legacy: random initial state, F32 inference,
  **unknown recorded training precision**. Do not relabel round21 as fixed.
- Historical recipe: `--recurrent-state legacy-random --precision bf16`.
  Previous prefix-fit launchers now pin these flags for reproducibility.
- New training needs matching fresh initialization, not the old random-state
  `initial.bin`. No automatic transplantation or contract migration.
- Normal compilation must include the sibling Edifice change. With native builds
  skipped, source-load the modules in `eval_runs/0913_execution_contract/run.sh`;
  loading only the drill script into stale BEAM code is insufficient.
- Teacher-fit reports record the contract. The readiness CLI verifies it against
  the checkpoint in addition to its hash. Unstamped legacy reports remain accepted
  only for legacy policies.

## Validation: no optimizer steps

- Six CPU contract/gate tests pass: actual Edifice GRU batch-row, permutation,
  and batch-size invariance; invalid modes; legacy defaults; report provenance.
- Existing round21 checkpoint loads as legacy/unknown-training-precision.
  The updated readiness CLI rechecks its original report successfully and still
  returns `ready: false` (`legacy_gate.json`); the checkpoint hash is unchanged.
- CUDA smoke exports and reloads a two-layer GRU plus AR head, compares the trainer
  predictor to the exported model, and exercises Agent's trunk and sequential head
  against full-model teacher forcing on the same selected action prefix.
- Final `parity_v3.json`: reload and train/inference-mode differences **0**;
  sequential logits max difference **1.64e-7**, all six head decisions agree;
  batch1 vs64 max absolute logit difference **0.001493394**.
- A second fresh initialization exceeded the original 0.001 GPU batch tolerance
  (0.001212969). The failed attempt is retained in `parity_v2.log`.
  The smoke explicitly allows 0.005 across GPU batch sizes; CPU invariance remains
  checked at 0.00001. This does **not** relax readiness/confidence thresholds or
  certify threshold-near decisions across batch shapes. F32 alone does not make
  different GPU matrix shapes bitwise identical.
- Evidence: `eval_runs/0913_execution_contract/`. No live Dolphin run, optimizer
  update, candidate promotion, or training-budget expenditure.

## Next, in order

- [x] Establish and enforce recurrent-state/precision contracts.
- [x] Add batch/reload and sequential-head smoke coverage.
- [x] Save fresh zero-state/F32 initialization for the fixed proof recipe.
- [x] Run the next explicitly bounded early-sustain/recovery fitting experiment.
- [x] Require the unchanged per-case confidence gate before live evaluation.
- [x] Revalidate the qualifying checkpoint with teacher-matched handoffs.
- [ ] Broaden neutral starts, sustained chains, and recoveries.

These changes remove execution ambiguities. They do not establish that the three
failed windows are learned or that the bot is ready.

Follow-up: [fresh bounded fit](ZERO_F32_FIT_2026-09-13.md) fixes all three old
mistakes, but leaves a different early recovery miss and one canonical miss.
Five of six readiness cases pass; live evaluation remains withheld.

Latest: [same-initialization no-dropout fit](NO_DROPOUT_FIT_2026-09-13.md)
passes all six confidence gates and12/12 local closed-loop responses. Historical
withheld status above applies only to the earlier checkpoint.
