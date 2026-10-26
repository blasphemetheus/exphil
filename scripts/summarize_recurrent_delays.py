#!/usr/bin/env python3
"""Validate the predeclared delay screen and report paired seed-level deltas."""
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=pathlib.Path)
parser.add_argument("--allow-partial", action="store_true")
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
min_gru = "checkpoints/coh_prev_q_ck4/model_best_policy.bin"
mamba = "checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin"
gru = "checkpoints/fox_v3_20260917_050901_resume3/model_policy.bin"
contexts = {
    "fd3_min_gru": ("final_destination", 3, min_gru),
    "fd4_min_gru": ("final_destination", 4, min_gru),
    "bf3_min_gru": ("battlefield", 3, min_gru),
    "fd3_gru": ("final_destination", 3, gru),
}
sources = [
    "lib/exphil/agents/agent.ex", "lib/exphil/sim/action_delay.ex",
    "scripts/sim_closed_loop.exs", "scripts/compare_recurrent_delays.exs",
    "scripts/run_recurrent_validation.exs",
    "../edifice/lib/edifice/stateful.ex", "../edifice/lib/edifice/stateful/replay.ex",
    "/home/blewf/git/msl-main/build/melee_core/python/libmelee_core.so",
    min_gru, mamba, gru,
]
hashes = {p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sources}
manifest_path = args.directory / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    if manifest["sha256"] != hashes:
        raise SystemExit("Source/checkpoint/simulator changed since protocol capture")
else:
    manifest = {
        "exphil_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "edifice_revision": subprocess.check_output(["git", "-C", "../edifice", "rev-parse", "HEAD"], text=True).strip(),
        "sha256": hashes,
        "sources": {p: pathlib.Path(p).read_text() for p in sources if p.endswith((".ex", ".exs"))},
        "protocol": "96 arms: two actors, four contexts, four paired seeds 1101..1104, windowed/carried/resync30; eight environments, 1800 frames each. Fixed action FIFO, both players delayed equally; warm-start neutral pending inputs; reset clears pending inputs for finished rows. Delay 3/4 is outside these checkpoints' delay-0 training. NIF simulator, not Dolphin or network jitter. No promotion.",
        "runtime": "RTX 5090, EXLA CUDA highest precision, portable kernels, synchronous NIF simulator",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

metrics = ["sd_per_min", "deaths_per_min", "kills_per_min", "damage_dealt_per_min",
           "damage_taken_per_min", "recovery_rate", "neutral_share", "repeat_prev"]
groups, missing, paired = {}, [], {}
for arch, policy in [("min_gru", min_gru), ("mamba", mamba)]:
    for context, (stage, delay, opponent) in contexts.items():
        rows_by_mode = {}
        for mode in ["windowed", "carried", "resync30"]:
            rows = {}
            for seed in range(1101, 1105):
                label = f"{arch}_{context}_{mode}_s{seed}"
                path = args.directory / f"{label}.json"
                if not path.exists():
                    missing.append(path.name)
                    continue
                row = json.loads(path.read_text())
                expected = dict(label=label, policy=policy, opponent=opponent, seed=seed,
                                stage=stage, frame_delay=delay, opponent_frame_delay=delay,
                                decision_to_application_frames=delay+1, delay_model="fixed_actuation_fifo",
                                envs=8, frames_per_env=1800, character="fox", opponent_character="fox",
                                inference_driver="batch", deterministic=False, sampling_seeded=True,
                                stateful_step=mode != "windowed", stateful_resync=30 if mode == "resync30" else None)
                for key, value in expected.items():
                    assert row[key] == value, (path, key, row[key], value)
                rows[seed] = row
            rows_by_mode[mode] = rows
            if rows:
                groups[f"{arch}/{context}/{mode}"] = {
                    "runs": len(rows), "seeds": list(rows),
                    "means": {m: statistics.mean(r[m] for r in rows.values() if r[m] is not None)
                              if any(r[m] is not None for r in rows.values()) else None for m in metrics},
                }
        for baseline in ["windowed", "carried"]:
            pairs = [(r, rows_by_mode[baseline][s]) for s, r in rows_by_mode["resync30"].items()
                     if s in rows_by_mode[baseline]]
            if pairs:
                paired[f"{arch}/{context}/resync30-minus-{baseline}"] = {
                    m: {"per_seed": {str(a["seed"]): a[m]-b[m] for a, b in pairs},
                        "mean_delta": statistics.mean(a[m]-b[m] for a, b in pairs)}
                    for m in metrics if all(a[m] is not None and b[m] is not None for a, b in pairs)
                }
if missing and not args.allow_partial:
    raise SystemExit(f"Incomplete screen: {len(missing)} missing arms")
report = {"complete": not missing, "completed": 96-len(missing), "missing": missing,
          "groups": groups, "paired": paired,
          "note": "Four seed-level replicas per context; environment rows are not independent replicas. No win-rate or promotion claim."}
path = args.directory / ("partial_summary.json" if missing else "summary.json")
path.write_text(json.dumps(report, indent=2) + "\n")
print(f"{96-len(missing)}/96 arms; wrote {path}")
