"""Publish CUDA.md atomically only after the current diagnostic exits.

Read-only observer of the diagnostic. No Mix, GPU work, restart, or promotion.
"""
import datetime
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/mamba_direct_parallel_memcheck.log"
ARTIFACTS = ROOT / "eval_runs/0925_fox_mamba/crash/direct_parallel_memcheck"
TARGET = ROOT / "CUDA.md"
PIDS = (2753372, 2753971)


def identity(pid):
    try:
        # comm may contain spaces/parentheses; fields after its final ')' start at state.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]  # starttime, field 22
    except (FileNotFoundError, ProcessLookupError):
        return None


def classify(log, completed):
    summaries = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", log)
    errors = int(summaries[-1]) if summaries else None
    disabled = "Sanitizer will be disabled" in log
    fatal = any(s in log for s in ("[FATAL]", "CUDA_ERROR_ILLEGAL_ADDRESS", "** (RuntimeError)"))
    if disabled:
        return "INCONCLUSIVE — sanitizer disabled", errors
    if errors is not None and errors > 0 or fatal:
        return "FAILED — diagnostic reported errors", errors
    if (errors == 0 and completed and completed.get("steps") == 1879
            and completed.get("chunks") == 2 and completed.get("mode") == "parallel"
            and completed.get("start") == 12 and "Completed 1879 steps" in log):
        return "PASSED — this instrumented two-chunk run only", errors
    return "INCONCLUSIVE — missing completion or final sanitizer evidence", errors


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def publish():
    log = LOG.read_text() if LOG.exists() else ""
    completed = read_json(ARTIFACTS / "completed.json")
    last = read_json(ARTIFACTS / "last_step.json")
    verdict, errors = classify(log, completed)
    losses = re.findall(r"step (\d+), chunk (\d+), batch (\d+), loss ([^\s]+)", log)
    notable = [line for line in log.splitlines() if any(s in line for s in (
        "ERROR SUMMARY", "Invalid __", "Invalid global", "Invalid shared",
        "Race reported", "use-after-free", "use-before-alloc", "[FATAL]",
        "Sanitizer will be disabled", "Completed 1879 steps"))]
    evidence = "\n".join(notable[:25])[-10000:] or "No final error/summary line found. Inspect the full log."
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = f"""# CUDA diagnostic findings

Published: {now}. This file is the completion signal requested by the user.

**Result: {verdict}.**

The observed diagnostic and sanitizer processes have exited. This watcher
does not launch any follow-up training. Check for other agents' GPU jobs
before running Mix or rebuilding shared EXLA.

## Completed diagnostic

- Scope: original Fox corpus chunks 12–13; two 64-game chunks, expected 1879 updates.
- Model: two-layer Mamba, hidden512/expanded1024, state16, window80, batch128, f32.
- Concurrent replay preparation enabled; EXLA GPU reservation 45%.
- Compute Sanitizer memcheck and stream-ordered allocation race tracking enabled.
- BEAM was launched directly through scripts/mamba_sanitizer_direct.py to keep
  instrumentation active. Optional CUDA API-return reporting was disabled
  because XLA's missing-symbol lookups produced unrelated API reports.
- Final sanitizer error count: **{errors if errors is not None else 'unavailable'}**.
- Completion artifact: `{json.dumps(completed)}`.
- Last pre-step marker (not necessarily completed): `{json.dumps(last)}`.
- Last logged loss row [step, chunk, batch, loss]: `{json.dumps(losses[-1] if losses else None)}`.

Log: [mamba_direct_parallel_memcheck.log](logs/mamba_direct_parallel_memcheck.log).
Artifacts: `eval_runs/0925_fox_mamba/crash/direct_parallel_memcheck/`.
Nearby paired state/input snapshots are capture0/capture1; the last-step
marker can be up to 99 updates ahead of a saved pair.

```text
{evidence}
```

## Findings established before this run

1. A real native bug was fixed: selective-scan backward ignored failure of its
   640 MiB cudaMallocAsync workspace allocation and still launched a kernel.
   The regression failed before the fix and passed after. Edifice commit c90990a.
2. That fix is insufficient: the repeated asynchronous pipeline still crashed
   at update639 with roughly14 GiB free GPU memory; another attempt failed at81.
3. A nearby saved state/batch ran another1000 updates successfully without
   concurrent replay preparation. Synchronous CUDA execution completed the
   same two-chunk test for1879 updates. Timing/concurrency is implicated;
   the exact remaining native fault is **not established**.
4. Recovery checkpoints now save before training, after the first update and
   every500 updates. Config tensors are copied to CPU for portable serialization.
   They preserve weights/optimizer, not the shuffled data cursor.

## Interpretation and next action for Claude

If the result above passed, it means no checked memory error was reported
**in this instrumented run**. Instrumentation changes timing; a pass does not
clear the known intermittent asynchronous crash or authorize calling the full
training pipeline fixed. If it failed, inspect the first invalid-memory report
and its kernel/buffer details, rather than a later asynchronous transfer error.
If inconclusive, recover the missing evidence before calling it a pass.

Continue with the smaller real-input gpu/cpu background-embedding control in
`scripts/mamba_concurrency_probe.exs` (written but not yet run at handoff).
Reduce any failure to a regression, fix it, and repeat uninstrumented chunk
transitions. The planned >11001-update/16-chunk endurance gate has not passed.
Full-corpus training remains stopped; this watcher does not restart it.

Resume details and exact commands:
[HANDOFF_2026-09-25a.md](docs/planning/HANDOFF_2026-09-25a.md).
Earlier evidence matrix:
[FOX_MAMBA_CRASH_REGRESSIONS.md](docs/planning/FOX_MAMBA_CRASH_REGRESSIONS.md).

Code commits: ExPhil a1420270, Edifice c90990a; handoff 3cea69ab. Nothing pushed
by the diagnostic watcher. No extra Mix/GPU tests were run by this observer.
"""
    temporary = ROOT / f".CUDA.md.{os.getpid()}.tmp"
    try:
        with temporary.open("x") as stream:
            stream.write(result)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication, refusing to overwrite a file another agent created.
        os.link(temporary, TARGET)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Published {TARGET}: {verdict}", flush=True)


def main():
    if TARGET.exists():
        raise SystemExit("CUDA.md already exists; refusing to overwrite the completion signal")
    original = {pid: identity(pid) for pid in PIDS}
    print(f"Watching diagnostic process identities: {original}", flush=True)
    while any(start is not None and identity(pid) == start for pid, start in original.items()):
        time.sleep(5)
    # Allow the parent launcher to flush the final sanitizer summary.
    time.sleep(3)
    publish()


if __name__ == "__main__":
    main()
