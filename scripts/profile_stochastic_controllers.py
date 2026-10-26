"""Sequential full-controller benchmarks; run from ExPhil's repository root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--client", choices=["host", "cuda"], required=True)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--out", default="eval_runs/stochastic_controllers_20261026")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
fox_replay = "/home/blewf/Slippi/2026-10-Mainline/Game_20261022T170737.slp"
policies = [
    ("min_gru", "checkpoints/coh_prev_q_ck4/model_best_policy.bin", fox_replay, "fox"),
    ("gru", "checkpoints/fox_v3_20260917_050901_resume3/model_policy.bin", fox_replay, "fox"),
    ("mamba", "checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin", fox_replay, "fox"),
    ("mamba2", "checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin",
     "replays/greg/mewtwo/Game_20240222T222930.slp", "mewtwo"),
]

def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

sources = ["scripts/profile_recurrent_policy.exs", "scripts/run_recurrent_validation.exs",
           "lib/exphil/agents/agent.ex", "lib/exphil/agents/decode.ex",
           "lib/exphil/networks/policy/sampling.ex", "../edifice/lib/edifice/stateful.ex",
           "../edifice/lib/edifice/stateful/replay.ex"]
files = set(sources + [p for _, p, _, _ in policies] + [r for _, _, r, _ in policies])
hashes = {p: sha(p) for p in sorted(files)}
env = os.environ | {"RECURRENT_CLIENT": args.client, "EDIFICE_FUSED_CUSTOM_CALL": "0"}
manifest = {"client": args.client, "iterations": args.iterations, "hashes": hashes,
            "started_at": time.time(), "runs": []}

for name, policy, replay, character in policies:
    for rows in [1, 8]:
        for stochastic in [False, True]:
            label = f"{name}_{args.client}_{rows}_{'stochastic' if stochastic else 'deterministic'}"
            output = out / f"{label}.json"
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite {output}")
            if args.client == "cuda":
                occupied = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
                ], text=True).strip()
                if occupied:
                    raise RuntimeError(f"GPU occupied before {label}: {occupied}")
            command = ["devenv", "shell", "--", "elixir", "-pa", "_build/dev/lib/*/ebin",
                       "scripts/run_recurrent_validation.exs", "scripts/profile_recurrent_policy.exs",
                       "--policy", policy, "--replay", replay, "--character", character,
                       "--batch", str(rows), "--iterations", str(args.iterations), "--out", str(output)]
            if stochastic:
                command.append("--stochastic")
            print(f"Starting {label}", flush=True)
            with (out / f"{label}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads(output.read_text())
            assert report["stochastic"] == stochastic and report["rows"] == rows
            assert report["client"] == args.client
            ordinary = next(r for r in report["results"] if r["kind"] == "ordinary")
            print(f"  ordinary median/p95: {ordinary['median_us']}/{ordinary['p95_us']} us; "
                  f"distinct groups: {ordinary['distinct_controller_groups']}", flush=True)
            manifest["runs"].append({"file": str(output), "command": command})
            (out / f"{args.client}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

assert hashes == {p: sha(p) for p in sorted(files)}, "Source/checkpoint/replay changed during profile"
manifest["finished_at"] = time.time()
manifest["hashes_verified_after"] = True
(out / f"{args.client}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Completed {len(manifest['runs'])} profiles", flush=True)
