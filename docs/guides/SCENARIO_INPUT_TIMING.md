# Recorded-input timing gate

Policy runs in `scripts/scenario_suite.exs` now validate their input timeline
against the **same run's** Slippi recording before admitting scores to the
summary. A passing prefix drift check alone is not enough.

For requested reaction delay `k`, the suite expects a decision at `t` to appear
in the recording at `t + k + 1`, and a command sent at `t` to appear at `t + 1`.
The verifier checks both, using digital buttons and analog components (0.02
tolerance). Decisions still queued when the response window ends are not scored.
Missing, ambiguous, empty, duplicate-frame, noncontiguous, out-of-order, or
mismatching evidence fails closed.
The reader retries missing/unreadable files for up to one second after bridge
shutdown, allowing asynchronous replay writes to finish without assuming success.

An invalid run has `timing_valid: false`, `score: null`, `pass: false`, and its
raw behavioral score under `unvalidated_score`. It is excluded from the summary,
counted under `invalid_timing_runs`, and causes CLI exit status 2 after artifacts
are written and cleanup finishes. Raw details remain available for diagnosis.
The check establishes delivery timing, not expert correctness or generalization.

Teacher runs retain their separate teacher-future audit. This gate currently
checks policy runs only; it does not certify teacher latency.

## Stress protocol

Run `bash scripts/stress_input_timing.sh NEW_OUTPUT_DIRECTORY` with no existing
BEAM job. This loads the sibling console and local bridge source in memory,
without rebuilding shared native dependencies. Four fresh launches at each of
blocking, 100 ms polling, and 1 ms polling must all pass both timing and control
checks. Source hashes, logs, recordings, raw scores, exit codes, and a combined
summary are retained. The script exits nonzero if any case fails.

Aggressive polling reproduced repeated controller commitments during the
bridge's internal nil-result retry loop. Internal retries now call the new
`Melee.Console.step/3` with `flush: false`; the first attempt still commits input.
This fix applies to retries inside a blocking `MeleePort.step` call. Separate
external `poll: true` calls, pause recovery, and reactive opponents are not
certified by this matrix. Do not infer training readiness from unit tests alone.

## Diagnostics

- `--trace-policy-inputs`: retain issued and sent commands and live action fields.
  Policy timing validation captures the command trace automatically.
- `--no-verify-input-timing`: reproduce legacy diagnostics without score gating;
  do not use these scores as validated training gates.
- `--prefix-history applied|committed`: compare the legacy warm-up against a
  reconstructed committed-decision history. Default remains `applied`; this
  experiment did not resolve the delay-4 failure, so it is not promoted as a fix.
- `--console-timeout 0.0`: diagnostic blocking transport; not a demonstrated cure
  for the variable delivery latency. Historical `--live-af` table experiments
  likewise were not a promoted fix. The scenario suite now defaults to the
  separate `:libmelee` producer-indexing contract, not that historical table;
  `--no-live-af` retains the unconverted policy comparison. See
  [matched-handoff validation](../planning/MATCHED_HANDOFFS_2026-09-13.md).
- `--no-orphan-sweep`: use owned bridge cleanup, not the legacy global sweep.

`scripts/audit_scenario_input_timing.exs --scores SCOREBOARD --out NEW_JSON`
independently compares trace commands with raw replay inputs over offsets 0–8,
and live/recorded action alignment over offsets -2–2. It never overwrites output.
It reports perfect-match offsets as evidence, not as a knob to optimize by score.

Normal `mix run` builds the new modules. If deliberately using `--no-compile`,
load `lib/exphil/eval/scenario_input_timing.ex` with `-r` (and `scenario_history.ex`
for the committed-history experiment). Do not rebuild shared native libraries
while another BEAM is live.

See [the active proof tracker](../planning/MULTISHINE_PIPELINE_PROOF.md).
# Readback contract update (09-13)

The policy gate now uses versioned analog conversion rather than literal
requested-value equality. See [input readback contract](INPUT_READBACK_CONTRACT.md)
for the fixed pipe-trigger path, historical profile selection, component errors,
and matched `--prefix-history cold` mode. The older standalone
`audit_scenario_input_timing.exs` remains a literal-value alignment diagnostic;
use `reverify_scenario_inputs.exs --profile pipe_v1|pipe_v2` for current verification.
