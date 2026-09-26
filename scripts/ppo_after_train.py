#!/usr/bin/env python3
"""Run evaluation sequentially after PPO, keeping status on disk."""
import json
import pathlib
import subprocess
import time
import datetime

from ppo_completion_watch import ROOT, RUN, service_state
from ppo_style_report import make_report

STATUS = ROOT / "eval_runs/0923_ppo/eval_pipeline_state.json"
POLICY = "checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin"


def record(stage, **extra):
    data = {"stage": stage, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(STATUS)
    print(json.dumps(data), flush=True)


def evaluate(name, head=None, games=200, frames=28800, envs=32):
    out = f"eval_runs/0923_ppo/{name}"
    record(name, output=out, head=str(head) if head else None, games=games, frame_cap=frames)
    args = ["devenv", "shell", "--", "env", "EXLA_MEMORY_FRACTION=0.35", "mix", "run",
            "--no-compile", "--no-deps-check", "scripts/ppo_eval.exs", "--policy", POLICY,
            "--out", out, "--games", str(games), "--frames", str(frames), "--envs", str(envs)]
    if head:
        args += ["--head", str(head.relative_to(ROOT))]
    with (ROOT / "logs" / f"ppo-{name}.log").open("x") as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((ROOT / out / "summary.json").read_text())
    if summary["games"] != games:
        raise RuntimeError(f"{name}: expected {games} games, got {summary['games']}")
    return summary


def main():
    record("waiting_for_training", training_unit="exphil-ppo-0923-v1")
    while service_state()["ActiveState"] in ("active", "activating", "deactivating", "reloading"):
        time.sleep(30)
    if (RUN / "STOP").exists():
        record("skipped", reason="STOP requested; not starting automatic GPU evaluation")
        return
    # Do not overlap a GPU job another session started during the handoff.
    while True:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True)
        if "beam" not in result.stdout.lower():
            break
        record("waiting_for_gpu", reason="Another BEAM GPU process is active")
        time.sleep(30)
    heads = sorted(RUN.glob("head_iter*.bin"), key=lambda p: int(p.stem.split("iter")[1]))
    if not heads:
        raise RuntimeError("No saved PPO heads available")
    head = heads[-1]
    # A short run exercises both ports, exports and the style fingerprint path.
    evaluate("eval_smoke", head=head, games=4, frames=1800, envs=2)
    fp = ROOT / "eval_runs/0923_ppo/eval_smoke/fingerprints.jsonl"
    if not fp.exists() or len(fp.read_text().splitlines()) != 8:
        raise RuntimeError("Evaluation smoke did not produce both-port style fingerprints")
    control = evaluate("eval_prior_control", games=32, envs=16)
    candidate = evaluate("eval_candidate", head=head)
    record("style_comparison", evaluation="eval_runs/0923_ppo/eval_candidate")
    make_report(ROOT / "eval_runs/0923_ppo/eval_candidate")
    record("completed", checkpoint=str(head.relative_to(ROOT)), control=control, candidate=candidate,
           style_report="eval_runs/0923_ppo/eval_candidate/style_comparison.json",
           note="R3 style/human-range gate remains unassessed; no production promotion")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record("failed", error=str(exc))
        raise
