"""Paired rolled/unrolled stochastic full-Agent profiles with seeded action parity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--client", choices=["host", "cuda"], required=True)
parser.add_argument("--out", default="eval_runs/sampling_kernels_20261026")
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
sources = ["scripts/profile_sampling_kernels.py", "scripts/profile_recurrent_policy.exs",
           "scripts/run_recurrent_validation.exs", "lib/exphil/agents/agent.ex",
           "lib/exphil/networks/policy/sampling.ex", "../edifice/lib/edifice/utils/sampling_random.ex",
           "../edifice/lib/edifice/stateful.ex", "../edifice/lib/edifice/stateful/replay.ex"]
files = set(sources + [p for _, p, _, _ in policies] + [r for _, _, r, _ in policies])

def hashes():
    values = {}
    for path in sorted(files):
        with open(path, "rb") as stream:
            values[path] = hashlib.file_digest(stream, "sha256").hexdigest()
    return values

manifest_path = out / f"{args.client}_manifest.json"
manifest = {"client": args.client, "hashes": hashes(), "started_at": time.time(), "runs": []}
if manifest_path.exists():
    previous = json.loads(manifest_path.read_text())
    assert previous["hashes"] == manifest["hashes"], "Cannot resume after source/input changes"
    manifest = previous

for name, policy, replay, character in policies:
    for rows in [1, 8]:
        pair = []
        for kernel in ["rolled", "unrolled"]:
            label = f"{name}_{args.client}_{rows}_{kernel}"
            output = out / f"{label}.json"
            command = ["devenv", "shell", "--", "elixir", "-pa", "_build/dev/lib/*/ebin",
                       "scripts/run_recurrent_validation.exs", "scripts/profile_recurrent_policy.exs",
                       "--policy", policy, "--replay", replay, "--character", character,
                       "--batch", str(rows), "--iterations", "100", "--stochastic", "--out", str(output)]
            if not output.exists():
                if args.client == "cuda":
                    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                                    "--format=csv,noheader"], text=True).strip()
                    if busy:
                        raise RuntimeError(f"GPU busy before {label}: {busy}; resume when free")
                print(f"Starting {label}", flush=True)
                env = os.environ | {"RECURRENT_CLIENT": args.client, "EDIFICE_FUSED_CUSTOM_CALL": "0",
                                    "SAMPLING_REFERENCE": "1" if kernel == "rolled" else "0"}
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                with (out / f"{label}.log").open("w") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads(output.read_text())
            assert report["sampling_kernel"] == kernel and report["client"] == args.client
            assert report["rows"] == rows and report["stochastic"]
            assert all(r["samples"] == 100 for r in report["results"])
            pair.append(report)
            if not any(r["file"] == str(output) for r in manifest["runs"]):
                manifest["runs"].append({"file": str(output), "command": command, "kernel": kernel})
            print(f"  {label}: {report['results'][0]['median_us']} us median", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        assert pair[0]["priming_controller_sha256"] == pair[1]["priming_controller_sha256"], name
        print(f"  {name}/{rows}: seeded controller stream exactly matches", flush=True)

assert hashes() == manifest["hashes"], "Source/input changed during benchmark"
manifest.update(finished_at=time.time(), hashes_verified_after=True, priming_controllers_exact=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print("Completed 16 paired profiles", flush=True)
