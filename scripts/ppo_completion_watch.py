#!/usr/bin/env python3
"""Update the PPO handoff after the existing training service exits; no ML imports."""
import datetime
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "eval_runs/0923_ppo/v1_fixed"
DOC = ROOT / "docs/planning/PPO_LIVE_STATUS_2026-09-23.md"
LOG = ROOT / "logs/exphil-ppo-0923-v1.log"
UNIT = "exphil-ppo-0923-v1.service"
MARKER = "<!-- PPO_COMPLETION_0923_V1 -->"
END_MARKER = "<!-- END_PPO_COMPLETION_0923_V1 -->"
ACTIVE = ("active", "activating", "deactivating", "reloading")


def service_state(unit=UNIT):
    proc = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "LoadState", "-p", "Result"],
        capture_output=True, text=True, timeout=15,
    )
    fields = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    if not fields.get("ActiveState"):
        raise RuntimeError(proc.stderr or "Cannot determine training service state")
    return fields


def report(state):
    rows = []
    path = RUN / "metrics.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A failed process can leave a partial final line.
    last = rows[-1] if rows else {}
    log = LOG.read_text(errors="replace") if LOG.exists() else ""
    completed = "→ eval_runs/0923_ppo/v1_fixed/log.json" in log
    if state["ActiveState"] in ACTIVE:
        outcome = "PPO training is running."
    elif completed and last.get("iter") == 200:
        outcome = "Completed all 200 iterations."
    elif completed and "STOP file found" in log:
        outcome = "Stopped cleanly at the requested STOP file."
    elif completed and "— stopping (the head is drifting" in log:
        outcome = "Stopped cleanly at the KL guard."
    else:
        outcome = "Service exited without a confirmed successful completion; inspect the training log."
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines = [MARKER, "## Automatic run status", "", f"Updated automatically at {now}.", "", outcome,
             f"Service state: `{state.get('ActiveState')}`; result: `{state.get('Result', 'unavailable')}`.",
             f"Completed metric rows: {len(rows)}; final recorded iteration: {last.get('iter', 'none')}.", ""]
    if rows:
        lines += ["| Latest metric | Value |", "| --- | --- |"]
        for key in ("reward", "kl", "entropy", "clip_frac", "value_ev", "vloss", "ms"):
            lines.append(f"| {key} | {last.get(key, 'unavailable')} |")
        lines.append("")
    for prefix in ("head", "trainer"):
        paths = sorted(RUN.glob(f"{prefix}_iter*.bin"), key=lambda p: int(p.stem.split("iter")[1]))
        if paths:
            lines.append(f"Latest {prefix} checkpoint: `{paths[-1].relative_to(ROOT)}` ({paths[-1].stat().st_size} bytes).")
    units = subprocess.run(["systemctl", "--user", "list-units", "exphil-*", "--no-legend", "--plain"],
                           capture_output=True, text=True, check=True, timeout=15)
    lines += ["", "### Services present at this update", "", "```text", units.stdout.strip() or "No exphil services listed.", "```", ""]
    pipeline = ROOT / "eval_runs/0923_ppo/eval_pipeline_state.json"
    if pipeline.exists():
        stage = json.loads(pipeline.read_text())
        lines += ["### Evaluation pipeline", "", "```json", json.dumps(stage, indent=2), "```", ""]
        for name in ("eval_smoke", "eval_prior_control", "eval_candidate"):
            path = ROOT / "eval_runs/0923_ppo" / name / "games.jsonl"
            if path.exists():
                lines.append(f"{name}: {len(path.read_text().splitlines())} game results recorded.")
    lines += ["", "### CPU checkpoint check", ""]
    for name in ("cpu_prior_control", "cpu_candidate40"):
        folder = ROOT / "eval_runs/0923_ppo" / name
        if (folder / "summary.json").exists():
            summary = json.loads((folder / "summary.json").read_text())
            lines += [f"{name} completed:", "```json", json.dumps(summary, indent=2), "```", ""]
        elif (folder / "games.jsonl").exists():
            lines.append(f"{name}: {len((folder / 'games.jsonl').read_text().splitlines())} game results recorded; summary pending.")
        else:
            lines.append(f"{name}: no game results recorded yet.")
    lines += ["", "Training completion is not an R3 gate pass. The preregistered human-range",
              "style checks remain outstanding. No production policy was promoted.",
              "", "Claude can take over from these artifacts. Check for other active training",
              "services before invoking Mix or starting GPU evaluation.", "", END_MARKER, ""]
    return "\n".join(lines)


def update_doc(state):
    original = DOC.read_text()
    block = report(state)
    if MARKER in original and END_MARKER in original:
        before, remaining = original.split(MARKER, 1)
        _, after = remaining.split(END_MARKER, 1)
        revised = before + block.rstrip() + after
    else:
        title, rest = original.split("\n", 1)
        revised = title + "\n\n" + block + rest
    temp = DOC.with_suffix(".md.completion.tmp")
    temp.write_text(revised)
    temp.replace(DOC)


def main():
    if "--preview" in sys.argv:
        print(report(service_state()))
        return
    print(f"Updating {DOC} every 30 seconds until training and evaluation exit", flush=True)
    while True:
        try:
            state = service_state()
            evaluation = service_state("exphil-ppo-0923-eval.service")
            cpu = service_state("exphil-ppo-0923-cpu-check.service")
            update_doc(state)
        except (RuntimeError, subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
            print(f"Status check failed; retrying: {exc}", flush=True)
            time.sleep(30)
            continue
        if "--once" in sys.argv or all(s["ActiveState"] not in ACTIVE for s in (state, evaluation, cpu)):
            break
        time.sleep(30)
    print(f"Updated {DOC}", flush=True)


if __name__ == "__main__":
    main()
