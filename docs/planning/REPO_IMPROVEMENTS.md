# Repository improvement plan

Created 2026-09-10 from the repository review. All seven recommendations are
tracked here for incremental implementation. This is the execution checklist;
[CURRENT_STATUS.md](CURRENT_STATUS.md) is the project entry point, and dated
handoffs retain experimental history.

## Working rules

- Protect the active job. The user clarified on 09-10 that Claude is running a
  sweep rather than training. Its process, launcher, inputs, artifacts, and
  dependency builds must not be changed by this work.
- Documentation can be updated now. Before runtime work, establish a separate
  checkout with independent dependency and build directories, including any
  local Edifice/Nx/libmelee_ex dependencies. A separate `_build` alone does not
  isolate shared native libraries or source read by a later training stage.
- Do not run Mix, training, Dolphin, GPU benchmarks, cleanup, or dependency
  updates in the active checkout while training continues. Do not edit scripts
  or libraries an active multi-stage launcher may read on its next stage.
- Preserve existing uncommitted work. At review time this includes
  `scripts/dagger_drill.exs`, `scripts/gate_sweep.sh`,
  `scripts/snippet_mine.exs`, and the `0910_*` evaluation directories.
- Implement one reviewable change at a time. Record actual validation below;
  an implementation awaiting tests is not complete. Do not commit, push,
  restart, or deploy as part of this checklist without separate authorization.
- Keep current checkpoint compatibility explicit. New metadata must distinguish
  missing legacy information from a verified modern value.

## Order and status

`Planned` → `In progress` → `Awaiting validation` → `Validated, staged` → `Done`.
Staged means tested in isolation but not yet activated in the sweep checkout.

| ID | Change | Priority | Status | Dependency |
| --- | --- | --- | --- | --- |
| R1 | Checkpoint-driven evaluation and diagnostics | High | Validated, staged | Sweep completion before activation |
| R2 | Canonical training delay | High | Validated, staged | Sweep completion before activation |
| R3 | Required semantic contract tests in CI | High | Planned | R4 for clean CI bootstrap |
| R4 | Reproducible dependencies and run provenance | High | In progress | Required companion APIs are not yet published |
| R5 | Evaluation identity and comparability | Medium | Planned | R2; reuse R4 provenance |
| R6 | Explicit run completion and watchdog status | Medium | In progress | Launcher/watchdog tested; automatic training events remain |
| R7 | Current documentation and supported entry points | Medium | Awaiting validation | Runtime walkthrough after R1/R4 |

Start with R7 while training continues. For runtime work, establish R4's
reproducible bootstrap, then R3's existing contract suite; implement R1 and R2
against that test foundation, followed by R5 and R6. Expand R3 as the new
contracts land. Priority describes impact, not a requirement to ignore build
dependencies.

**09-10 adjustment:** R4's remote defaults cannot yet use the local companion
revisions: required APIs are unpublished. Proceed with independent R6 tooling
and isolated local snapshots; do not silently downgrade dependencies or publish
other repositories to unblock the checklist.

## R1 — Checkpoint-driven evaluation and diagnostics

**Evidence:** `scripts/eval_model.exs` rebuilds a windowed model, whereas current
BPTT checkpoints need carry-aware evaluation. The failure is documented in
`FIXES.md` under the 2026-09-09 late additions. Several diagnostic calls pass
`batch.states` directly to a multi-input autoregressive predictor.

- [x] Define a shared evaluator that reconstructs the model and input layout from
  checkpoint metadata, with explicit handling of legacy artifacts.
- [x] Support BPTT carry, reset at game/discontinuity boundaries, and prevent
  unrelated replay sequences from sharing recurrent state.
- [x] Separate teacher-forced likelihood metrics from sampled gameplay metrics;
  autoregressive inputs must match the metric being measured.
- [x] Migrate the main evaluator and diagnostics to the common interface; migrate
  the sampler probe where its live-sampling semantics match.
- [x] Validate a small GRU + autoregressive BPTT model through a training step,
  export, fresh-process load, and evaluation. Cover game resets and legacy
  windowed evaluation. Reuse existing checkpoint and stateful parity fixtures.

**Done when:** an exported current-format BPTT policy evaluates without manual
architecture reconstruction, and diagnostics handle its input contract. Report
unsupported architectures explicitly; do not silently evaluate a different model.

## R2 — Canonical training delay

**Evidence:** streaming/BPTT consumes `frame_delay`; the standard pipeline
consumes `action_delay`; `LabelConvention.reaction_delay/1` currently takes their
maximum for metadata. Those descriptions can disagree with executed training.

- [x] Introduce one resolved reaction-delay value at configuration resolution.
  Retain old training flags as aliases, preserving their provenance long enough
  to reject explicit conflicts (including explicit zero versus nonzero).
- [x] Handle CLI, YAML, presets, resume metadata, and direct library callers;
  defaults must not masquerade as explicit user choices.
- [x] Route standard, streaming, BPTT, validation, and mixed drill data through the
  resolved convention, shifting each source exactly once.
- [x] Use the executed value in checkpoint metadata and comparability keys.
- [x] Preserve legacy checkpoint interpretation and live Dolphin `--frame-delay`
  numbering. Keep multi-delay conditioning and augmentation explicit.
- [x] Test both aliases on each pipeline, conflicting inputs, old checkpoints,
  discontinuities, and multi-delay data. Use identifiable successor actions.

**Done when:** changing the loading strategy cannot silently change the requested
reaction delay, and saved metadata describes the targets actually used.

## R3 — Semantic contracts in CI

**Evidence:** `test/test_helper.exs` excludes `:nif`, `:snapshot`, and `:property`;
the current workflow never enables them. Three real-replay causal alignment
tests carry the `:nif` tag.

- [ ] Add a required CPU/native contract job with Rust/NIF prerequisites and
  explicit test selection. Reuse existing tests before adding new ones.
- [ ] Include real-replay causal alignment, non-default subject ports,
  train/live embedding parity, and checkpoint export/load contracts.
- [ ] Include relevant snapshot and property tests explicitly; keep expensive
  GPU or Dolphin requirements out of the CPU job.
- [ ] Add R1/R2 regressions as they land, and make selected-test counts visible
  so a renamed tag cannot produce an empty successful gate.
- [ ] Verify bootstrap and the selected suite in an isolated clean checkout.
  Document separately scheduled GPU parity checks if hardware is available.

**Done when:** an isolated CPU CI job runs these semantic checks, and representative
label, perspective, and reload regressions fail the gate.

## R4 — Dependencies and run provenance

**Evidence:** `mix.exs` defaults to sibling Edifice/libmelee_ex paths, while CI
checks out only ExPhil. Remote fallback depends on `DOCKER_BUILD`; the current
lockfile has no entries for those local dependencies.

- [ ] Default to remote versions/revisions compatible with the APIs currently
  used by ExPhil; retain explicit `EDIFICE_PATH` and `LIBMELEE_EX_PATH` overrides.
  Do not assume the latest published Edifice release contains local changes.
- [ ] Lock both dependencies and verify the Nx/EXLA/fork combination on CPU;
  document the separate fused-kernel environment.
- [ ] Record ExPhil, Edifice, libmelee_ex, and Nx/EXLA identities, dirty-tree
  status or content identity, resolved config, seed, and toolchain in a run
  manifest. A package version alone cannot identify a modified local checkout.
- [ ] Keep generic architecture, recurrent state, and generic checkpoint/spec
  contracts in Edifice; keep Melee embeddings, labels, heads, and scoring in
  ExPhil. Add cross-repository contract tests at this boundary.
- [ ] Verify clean clone → dependency fetch → compile → CPU contracts, without
  relying on developer sibling directories. Verify explicit local overrides too.

**Done when:** clean CI and local development resolve intentional dependencies,
and a saved run identifies the code that produced it.

## R5 — Evaluation identity and comparability

**Evidence:** `Training.Comparability.key/1` records delay, embedding, loss, and
training delay sets, but not the evaluation corpus or evaluation protocol.
`Config.compute_manifest_hash/1` hashes paths, not replay contents.

- [ ] Define a versioned evaluation record, separate from training provenance.
- [ ] Fingerprint replay contents and the actual held-out split, with deterministic
  ordering and reusable ingestion-time hashes to avoid repeated large reads.
- [ ] Include evaluator version, metric definition, weighting, and relevant
  state/carry/decode settings in offline comparison identity.
- [ ] Record stage, opponent, deployment delay, decoding settings, seeds, and
  run-level samples for gameplay comparisons. Keep offline and gameplay
  comparison rules separate.
- [ ] Reject missing/mismatched identities by default in ranking tools; retain an
  explicit exploratory override with an explanation in the output.
- [ ] Test equal content at different paths, changed content at the same path,
  different splits/protocols, JSON round trips, and legacy missing metadata.

**Done when:** ranking cannot treat different evaluation datasets or protocols as
interchangeable merely because model configuration keys match.

## R6 — Run completion and watchdog status

**Evidence:** `scripts/train_watchdog.sh` reports apparent success after process
exit if a checkpoint exists or the log mentions export/convergence. An earlier
checkpoint can survive a later crash.

- [x] Add an atomic, versioned, run-specific status artifact with start/end,
  progress, exit status, artifact paths, and diagnostic outcome.
- [x] Distinguish successful completion, intentional early stop, failure,
  diagnostics failure, and unknown/interrupted termination.
- [x] Make the launcher record child exit status; missing terminal status must
  remain unknown/failed after an uncatchable crash, never inferred success.
- [x] Make the watchdog verify run identity and terminal status. A saved
  checkpoint is recoverability evidence, not completion evidence.
- [x] Test success, early stopping, crash after an intermediate checkpoint,
  stale artifacts, diagnostics failure, and absent/truncated status files using
  temporary fake processes/files; no real training or GPU is needed.
- [ ] Wire training callbacks/launchers to report successful epochs, artifact
  paths, early stopping, and diagnostic outcomes automatically. The generic
  `update` command supports these now; existing callbacks do not emit them.
- [ ] Verify one future launch end to end with the reporting callbacks. Until
  then, opt-in wrapped commands report actual process completion but cannot
  detect errors swallowed by the command itself.

**Done when:** a dead run with an old checkpoint cannot be reported as successful.
Install the new protocol only on future launches; do not retrofit the active run.

## R7 — Documentation and entry points

- [x] Add a current-status entry point linking experimental evidence, known
  evaluator limitations, deployment cards, and this checklist.
- [x] Correct README setup and training commands; distinguish historical
  benchmarks from current validated performance.
- [x] Correct contributor setup for both companion repositories and document
  the current flag-table/backbone-spec extension points.
- [x] Put current commands at the top of the training guide, explicitly marking
  legacy examples below. Preserve the generated flag reference.
- [x] Check links and command flags statically without starting Mix.
- [ ] After R1/R4, exercise the clean-checkout quick start and evaluation flow,
  then remove temporary setup/evaluator caveats.

**Done when:** a newcomer has one tested route from setup through training and
evaluation, and current priorities are not contradicted by undated examples.

## Change and validation log

### 2026-09-10 — Plan established

- Captured all seven review recommendations with dependencies and acceptance
  criteria. Runtime work is still planned, not implemented or tested.
- Training status comes from the user's report, not from a process inspection.
- No Mix, training, evaluator, GPU, service-management, or dependency-update
  commands were run during the review or planning work.

### 2026-09-10 — R7 documentation pass

- Added `CURRENT_STATUS.md` and linked both new planning pages from the README
  and documentation index.
- Switched the primary training examples to `train.exs`, documented both sibling
  dependencies and overrides, and corrected contributor extension points.
- Marked old benchmarks/trainer examples as historical, removed the stale
  passing-test count, and surfaced the BPTT evaluator limitation.
- Static validation: documented training flags checked against the parser and
  Config; Markdown local file targets checked; two existing training-guide links
  corrected. `git diff --check` passed before the final link corrections.
- Runtime setup, dependency compatibility, training, and evaluation were not
  exercised. R7 remains awaiting that walkthrough after R1/R4; R1–R6 remain open.
- Next implementation: R4's isolated dependency/bootstrap work, followed by R3's
  existing semantic test gate, then R1/R2. Do not start by recompiling this tree.

### 2026-09-10 — First runtime changes: R6 launcher and watchdog

- Created independent, non-hardlinked source clones under
  `/tmp/exphil-improvements.MggyAe/` for ExPhil, Edifice, and libmelee_ex.
  No shared dependency/build directories were copied or compiled.
- R4 investigation: GitHub HEAD for Edifice is
  `637b230d1d71ab9df7c22e77609d1949d11af6e0`, while local HEAD is
  `0edf76c3b5c40a198877763f315cfb395221e407`. Local-only commits include
  `Recurrent.build_backbone_with_carry/2` and the carry-checkpoint initial-state
  fallback. libmelee_ex remote HEAD is
  `60602e110af50bc8a9eca540795d7d1bf6d1b1b3`, local HEAD is
  `71114d2b6340e8c00d3f1884f5fdc225c872ffa2`. Do not pin inaccessible commits
or claim the remote pair supports today's code. Nothing was pushed.
- Added `scripts/run_status.py`: atomic, synchronized JSON records, exclusive
  run paths, child exit propagation, signal forwarding, progress updates,
  early-stop reporting, and persistent diagnostic-failure reporting.
- Updated `scripts/train_watchdog.sh` to check run ID, launcher PID, and terminal
  status. A checkpoint or export log alone now produces exit 2 (unknown), not
  success. Existing liveness monitoring remains available without a status file.
- Added [run-status operations guide](../operations/RUN_STATUS.md) and a separate
  CPU-only GitHub Actions job that needs no Mix/dependency bootstrap.
- Validation in the isolated clone: **17 Python unittest cases passed**;
  `bash -n scripts/train_watchdog.sh` and `git diff --check` passed. Tests use
  fake jobs and stub GPU queries/notifications. Installed code/test files were
  compared byte-for-byte against the tested copies.
- No existing job was wrapped, restarted, or reconfigured. No Mix, ML library,
  checkpoint, sweep input, or active sweep script was changed. This is opt-in
  tooling for future launches; automatic training callback events remain open.

### 2026-09-10 — R1 BPTT evaluation, then R2 training delay

- Implemented both in the independent checkout. Runtime files in this checkout
  remain unchanged because sweep completion is unconfirmed. The checked items
  above describe the staged implementation, not an activated feature.
- R1: shared metadata-driven forward adapter, GRU carry and discontinuity
  resets, complete tail coverage, frame-weighted teacher-forced metrics,
  standalone policy mode/layout metadata, and autoregressive-safe diagnostics.
  The sampler probe retains its different live-sampling semantics. Unsupported
  modes fail explicitly; BPTT gradient summaries remain skipped.
- R2: canonical `label_delay`, compatible training aliases, explicit conflict
  rejection, CLI/YAML/preset/resume resolution, and unset raw defaults. Standard,
  streaming, BPTT, validation, and mixed frames apply the base shift once per
  source. Export/comparability metadata uses that value. Legacy/live numbering
  stays unchanged; unsupported jitter/corpus shifts are rejected.
- Validation: **367 tests and 14 doctests passed, zero failures**, including
  native replay alignment, a tiny training/export/fresh-process evaluator flow,
  per-loader successor targets, resume/export contracts, and generated flag
  reference parity. `git diff --check` and staged patch application checks pass.
- Tests used independent copies of compiled dependencies/native libraries and
  recompiled edited modules on EXLA CPU, not a clean dependency fetch/build.
  This does not close R3/R4 or establish gameplay quality.
- [Durable patch, exact test selection, and activation steps](BPTT_DELAY_HANDOFF.md).
  Activate only after the sweep no longer reads this checkout, rerun contracts,
  then mark R1/R2 done. No commit, push, deployment, or run restart occurred.
