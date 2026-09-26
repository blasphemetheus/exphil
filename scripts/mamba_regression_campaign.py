"""Fail-fast Mamba recovery gates; no full-corpus training or promotion.

Run inside devenv after all other training/probes have finished. The longer
gate crosses the original 11001-update failure point using real corpus chunks.
Do not edit its code dependencies or run other Mix commands while active.
"""
import datetime
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_runs/0925_fox_mamba/regression"
MD = ROOT / "docs/planning/FOX_MAMBA_LIVE_STATUS.md"


def status(phase, child=None, log=None, code=None):
    row = dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               phase=phase, supervisor_pid=os.getpid(), child_pid=child.pid if child else None,
               log=str(log) if log else None, exit_code=code)
    (OUT / "status.json").write_text(json.dumps(row, indent=2))
    MD.write_text("# Fox Mamba live status\n\n" +
        f"Updated: {row['updated_at']}\n\nPhase: **{phase}**. " +
        f"Regression supervisor PID {os.getpid()}; child PID {row['child_pid']}.\n\n" +
        f"Log: `{log}`. Exit: `{code}`.\n\n" +
        "Full-corpus campaign remains stopped. These gates test allocation error handling, " +
        "checkpoint recovery, fused/fallback agreement, and real streaming transitions. " +
        "The final gate uses 16 original 64-game chunks, including the original failure region.\n\n" +
        "All probes use 45% EXLA GPU reservation. Unrelated Ollama is left alone. " +
        "Do not run Mix or edit this loop's dependencies while active.\n\n" +
        "Evidence and limitations: [crash regressions](FOX_MAMBA_CRASH_REGRESSIONS.md). " +
        "No full training restart or automatic promotion occurs here.\n")


def run(phase, command, extra_env=None):
    log = OUT / f"{phase}.log"
    (OUT / f"{phase}_command.json").write_text(json.dumps(command, indent=2))
    with log.open("w") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 env={**os.environ, **(extra_env or {})})
        while child.poll() is None:
            status(phase, child, log)
            time.sleep(5)
    status(phase + ("_passed" if child.returncode == 0 else "_failed"), log=log, code=child.returncode)
    if child.returncode:
        raise SystemExit(1)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--start-at", choices=["unit", "parity"], default="unit",
                        help="parity reuses separately recorded unit/native passes")
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ.update(EDIFICE_FUSED_CUSTOM_CALL="1", EXPHIL_EXLA_PRECISION="highest",
                      EXPHIL_GPU_MEMORY_FRACTION="0.45", EXLA_TARGET="cuda")
    (OUT / "invocation.json").write_text(json.dumps({"start_at": args.start_at}))
    if args.start_at == "unit":
        run("unit", ["mix", "test",
            "test/exphil/training/callbacks/rolling_checkpoint_test.exs",
            "test/exphil/training/callbacks/checkpoint_callback_test.exs",
            "test/exphil/training/checkpoint_roundtrip_test.exs",
            "test/exphil/training/checkpoint_config_backend_test.exs", "--include", "gpu"])
        run("native", ["python3", "scripts/native/test_mamba.py"])
    # Compile dev code once before the GPU stages, never during them.
    run("compile", ["mix", "compile"])
    base = ["mix", "run", "--no-compile", "--no-deps-check"]
    for name, enabled in [("fallback", "0"), ("fused", "1")]:
        run(name, base + ["scripts/profile_fox_mamba.exs", "--hidden", "512",
            "--batch", "64", "--window", "80", "--steps", "3", "--out", str(OUT / name)],
            {"EDIFICE_FUSED_CUSTOM_CALL": enabled})
    a = json.loads((OUT / "fallback/profile.json").read_text())
    b = json.loads((OUT / "fused/profile.json").read_text())
    if len(a["features"]) != len(b["features"]):
        raise ValueError("feature shape mismatch")
    errors = [abs(x - y) for x, y in zip(a["features"], b["features"])]
    maximum = max(errors)
    if not all(math.isfinite(x) for x in errors) or maximum > 1e-4:
        status("parity_failed", code=1)
        raise ValueError(f"fused/fallback feature disagreement: {maximum}")
    (OUT / "parity.json").write_text(json.dumps({"max_abs_error": maximum, "tolerance": 1e-4}))
    for name, start, count in [("transition", 12, 2), ("endurance", 1, 16)]:
        run(name, base + ["scripts/mamba_crash_stream.exs", str(OUT / name), str(start), str(count), "parallel"])
    completed = json.loads((OUT / "endurance/completed.json").read_text())
    if completed["steps"] <= 11001:
        status("insufficient_endurance", code=1)
        raise ValueError("endurance gate did not cross the original failure point")
    status("gates_passed", code=0)


if __name__ == "__main__":
    main()
