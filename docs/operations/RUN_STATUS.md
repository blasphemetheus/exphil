# Run completion records

For future training jobs and sweeps, launch the top-level command through
`scripts/run_status.py`. It records the command's actual exit status. The
watchdog no longer infers successful completion from an existing checkpoint
or an export message in a log.

This tool uses Python 3's standard library on Linux. It does not initialize
Elixir, EXLA, CUDA, or Dolphin. Do not wrap or restart an already-running job.

## Launch and monitor

Use a fresh directory and unique run ID for each invocation:

```bash
mkdir -p logs/example-run-001
python3 scripts/run_status.py run \
  --status logs/example-run-001/status.json --run-id example-run-001 \
  -- bash scripts/my_sweep.sh > logs/example-run-001/output.log 2>&1 &
run_launcher_pid=$!

bash scripts/train_watchdog.sh \
  --pid "$run_launcher_pid" --log logs/example-run-001/output.log \
  --status logs/example-run-001/status.json --run-id example-run-001
```

Replace `bash scripts/my_sweep.sh` with the actual foreground command. It must
wait for its work and propagate failures: a shell script that backgrounds work
and exits zero, or swallows a failed stage's exit code, cannot provide a reliable
completion signal. The PID passed to the watchdog is the Python launcher PID,
not the child BEAM PID. Launch the wrapper through the normal service manager
for long unattended jobs; the example's shell backgrounding is not a substitute
for session persistence.

Existing status files are never reused, even if they name the same run. A missing,
malformed, nonterminal, wrong-run, or wrong-PID record is unknown, not success.
Legacy watchdog invocations still monitor liveness, but return unknown on exit
unless they supply the new identity and status arguments.

## Progress and diagnostics

The wrapper exports `EXPHIL_RUN_STATUS` and `EXPHIL_RUN_ID` to its child.
Launchers can record successful stages without reconstructing those values:

```bash
python3 scripts/run_status.py update --epoch 3 --artifact checkpoints/example_best.bin
python3 scripts/run_status.py update --diagnostics passed
python3 scripts/run_status.py update --early-stopped
```

Call these only when the corresponding event actually occurred. Epochs cannot
decrease; diagnostic failures cannot be cleared. Artifact paths are reported
references, not assertions that an artifact is valid or complete. Status writes
are locked and replaced atomically, with file and directory synchronization.

Use `update --diagnostics failed` when diagnostics fail even if the command
otherwise returns zero. This produces `diagnostics_failed` and a nonzero wrapper
exit. Intentional early stopping is successful only if the child also exits
zero. Existing training callbacks do not yet emit these updates automatically:
without explicit reporting, epoch remains null and diagnostics is `not_run`.
Integration into the training callbacks is still tracked in R6.

The wrapper forwards SIGINT/SIGTERM to the child's process group and records
interruption. An uncatchable wrapper crash (including SIGKILL) may leave the
child alive and the record nonterminal. The watchdog reports unknown; process
group cleanup belongs to the launcher service's lifecycle policy.

## Inspect and test

```bash
python3 scripts/run_status.py check \
  --status logs/example-run-001/status.json --run-id example-run-001
python3 -m unittest discover -s test/scripts -v
bash -n scripts/train_watchdog.sh
```

`check` and the watchdog return 0 for verified completion/early stop, 1 for a
recorded failure, and 2 for unknown or invalid status. The wrapper preserves the
child's nonzero exit code, maps signal exits to `128 + signal`, and returns 1
for reported diagnostic failure after a zero child exit. Validation tests use
temporary fake jobs and stub GPU queries and notifications.
