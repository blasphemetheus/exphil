"""Bounded one-epoch Fox Mamba fit, reload timing, then balanced GRU matches.

Run inside devenv in a systemd user unit. Every phase writes a log and the
live markdown names the child PID. No automatic promotion or repeat loop.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

# No options — but parse anyway so `--help` prints this docstring instead of
# launching the campaign (2026-09-26 01:39: a `--help` probe started the real
# train phase and its status writer clobbered the original campaign's
# train.log/status.json before it was killed; GOTCHA #133).
argparse.ArgumentParser(description=__doc__).parse_args()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_runs/0925_fox_mamba/campaign"
CHECKPOINT = "checkpoints/fox_mamba_v1_20260925/model.axon"
POLICY = CHECKPOINT.replace(".axon", "_best_policy.bin")
GRU = "eval_runs/0923_ppo/eval_candidate/candidate_policy.bin"
MD = ROOT / "docs/planning/FOX_MAMBA_LIVE_STATUS.md"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
os.environ.update(EDIFICE_FUSED_CUSTOM_CALL="1", EXLA_TARGET="cuda",
                  EXPHIL_GPU_MEMORY_FRACTION="0.70", EXPHIL_EXLA_PRECISION="highest")


def status(phase, child=None, log=None, code=None):
    row = dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               phase=phase, supervisor_pid=os.getpid(),
               child_pid=child.pid if child and child.poll() is None else None,
               log=str(log) if log else None, exit_code=code, policy=POLICY)
    (OUT / "status.json").write_text(json.dumps(row, indent=2))
    MD.write_text("# Fox Mamba live status\n\n" +
        "Updated: " + row["updated_at"] + "\n\n" +
        f"Phase: **{phase}**. Unit: `exphil-fox-mamba-v1`. " +
        f"Supervisor PID: {os.getpid()}; active child: {row['child_pid']}.\n\n" +
        f"Current log: `{log}`. Last exit: `{code}`.\n\n" +
        "One epoch, 512-wide two-layer Mamba, window 80, batch 128, F32, " +
        "fused scan, stride 5. Sixteen disjoint validation games. " +
        "No style conditioning. Large streaming caches disabled.\n\n" +
        f"Candidate: `{POLICY}`. GRU opponent: `{GRU}`.\n\n" +
        "After fit: reload/Agent latency, then four FD games with swapped ports. " +
        "This is a development comparison, not a promotion gate or an isolated " +
        "architecture comparison (GRU also received PPO).\n\n" +
        "Do not run Mix or edit code this campaign calls while it is active. " +
        "Existing viewers and unrelated Phoenix are untouched. No commits/pushes.\n\n" +
        "Performance evidence: `docs/planning/FOX_MAMBA_PROFILE_2026-09-25.md`.\n")


def run(phase, args):
    log = OUT / f"{phase}.log"
    command = ["mix", "run", "--no-compile", "--no-deps-check", *args]
    (OUT / f"{phase}_command.json").write_text(json.dumps(command, indent=2))
    with log.open("w") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while child.poll() is None:
            status(phase, child, log)
            time.sleep(10)
        code = child.returncode
    status(phase + ("_finished" if code == 0 else "_failed"), log=log, code=code)
    if code:
        raise SystemExit(code)


if (ROOT / CHECKPOINT).exists() or (ROOT / POLICY).exists():
    raise SystemExit("Existing campaign checkpoint; refusing to overwrite")
run("train", ["scripts/train_fox_mamba.exs",
    "--backbone", "mamba", "--stage-internals", "--hidden-sizes", "512,512,256",
    "--batch-size", "128", "--precision", "f32", "--window-size", "80",
    "--stride", "5", "--dropout", "0.0", "--learning-rate", "0.0002",
    "--replays", "replays/erickfm_ranked/v2_filtered", "--train-character", "fox",
    "--select-character-port", "--stream-chunk-size", "64", "--no-cache-streaming",
    "--label-delay", "0", "--epochs", "1", "--seed", "905", "--head", "autoregressive",
    "--save-best", "--save-every-batches", "25000", "--label-smoothing", "0.0",
    "--no-focal-loss", "--button-pos-weight", "1,1,1,1,1,1,1,1", "--action-oversample", "1.0",
    "--entropy-weight", "0.0", "--neutral-weight", "1.0", "--stick-edge-weight", "1.0",
    "--name", "fox-mamba-v1", "--no-cache", "--checkpoint", CHECKPOINT])
run("agent", ["scripts/profile_fox_agent.exs", POLICY, str(OUT / "agent.json")])
run("match", ["scripts/sim_policy_match.exs", "--a", POLICY, "--b", GRU,
    "--games", "4", "--frames", "18000", "--out", str(OUT / "matches")])
status("complete", code=0)
