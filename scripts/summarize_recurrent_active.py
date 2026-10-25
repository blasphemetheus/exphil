"""Summarize the predeclared paired-seed screen without treating rows as replicates."""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=pathlib.Path)
parser.add_argument("--allow-partial", action="store_true")
args = parser.parse_args()
manifest_path = args.directory / "manifest.json"
if not manifest_path.exists():
    paths = [
        "scripts/sim_closed_loop.exs", "scripts/run_recurrent_validation.exs",
        "lib/exphil/agents/agent.ex", "../edifice/lib/edifice/stateful.ex",
        "../edifice/lib/edifice/stateful/replay.ex",
        "checkpoints/coh_prev_q_ck4/model_best_policy.bin",
        "checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin",
        "checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin",
        "checkpoints/fox_v3_20260917_050901_resume3/model_policy.bin",
    ]
    manifest = {
        "exphil_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "edifice_revision": subprocess.check_output(["git", "-C", "../edifice", "rev-parse", "HEAD"], text=True).strip(),
        "sha256": {p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths},
        "runtime": "RTX 5090, EXLA CUDA highest precision, portable kernels (EDIFICE_FUSED_CUSTOM_CALL=0), compiled local Nx/EXLA development dependencies",
        "runner_source": pathlib.Path("scripts/run_recurrent_validation.exs").read_text(),
        "protocol": "Seeds 1003..1006, FD, 1800 frames, fixed windowed MinGRU opponent; N=8 except independent-head Mamba-2 N=4. No deployment changes.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
manifest = json.loads(manifest_path.read_text())
sim_root = pathlib.Path(os.environ.get("EXPHIL_SIM_ROOT", pathlib.Path.home() / "git/msl-main"))
sim_library = sim_root / "build/melee_core/python/libmelee_core.so"
if "simulator_library" not in manifest and sim_library.exists():
    manifest["simulator_library"] = {
        "path": str(sim_library),
        "sha256": hashlib.sha256(sim_library.read_bytes()).hexdigest(),
        "mtime_unix": sim_library.stat().st_mtime,
        "note": "Simulator shared library from the configured/default root, recorded separately from source checkout revision.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
seeds = [1003, 1004, 1005, 1006]
architectures = ["min_gru", "mamba", "mamba2", "gru"]
modes = ["windowed", "carried", "resync30", "resync60"]
metrics = ["sd_per_min", "deaths_per_min", "kills_per_min", "damage_dealt_per_min",
           "damage_taken_per_min", "recovery_rate", "neutral_share", "repeat_prev"]
groups = {}
missing = []
for architecture in architectures:
    for mode in (["carried"] if architecture == "gru" else modes):
        rows = []
        for seed in seeds:
            path = args.directory / f"{architecture}_{mode}_s{seed}.json"
            if not path.exists():
                missing.append(path.name)
                continue
            row = json.loads(path.read_text())
            assert row["seed"] == seed and row["sampling_seeded"]
            assert row["stateful_step"] == (mode != "windowed")
            assert row["stateful_resync"] == {"resync30": 30, "resync60": 60}.get(mode)
            assert row["stage"] == "final_destination" and row["frames_per_env"] == 1800
            assert row["envs"] == (4 if architecture == "mamba2" else 8)
            assert row["inference_driver"] == ("single" if architecture == "mamba2" else "batch")
            assert row["character"] == ("mewtwo" if architecture == "mamba2" else "fox")
            assert row["opponent"] == "checkpoints/coh_prev_q_ck4/model_best_policy.bin"
            assert row["deterministic"] is False
            rows.append(row)
        if rows:
            groups[f"{architecture}/{mode}"] = {
                "runs": len(rows), "seeds": [r["seed"] for r in rows],
                "env_minutes": sum(r["env_minutes"] for r in rows),
                "means": {m: statistics.mean(r[m] for r in rows if r[m] is not None)
                          if any(r[m] is not None for r in rows) else None for m in metrics},
                "rows": rows,
            }
if missing and not args.allow_partial:
    raise SystemExit(f"Incomplete screen: {len(missing)} missing runs: {missing}")

paired = {}
for architecture in architectures[:-1]:
    for baseline in ["windowed", "carried"]:
        for mode in (modes[1:] if baseline == "windowed" else modes[2:]):
            current = groups.get(f"{architecture}/{mode}", {}).get("rows", [])
            reference = {r["seed"]: r for r in groups.get(f"{architecture}/{baseline}", {}).get("rows", [])}
            pairs = [(r, reference[r["seed"]]) for r in current if r["seed"] in reference]
            if not pairs:
                continue
            paired[f"{architecture}/{mode}-minus-{baseline}"] = {
                m: {"per_seed": {str(a["seed"]): a[m] - b[m] for a, b in pairs},
                    "mean_delta": statistics.mean(a[m] - b[m] for a, b in pairs)}
                for m in metrics if all(a[m] is not None and b[m] is not None for a, b in pairs)
            }
for group in groups.values():
    group["source_files"] = [r["label"] + ".json" for r in group.pop("rows")]
report = {"complete": not missing, "missing": missing, "groups": groups, "paired": paired,
          "note": "Four seed-level replicates per arm. Environment rows are not independent run-level samples. Paired differences name their windowed or carried baseline; no significance or promotion claim."}
out = args.directory / ("partial_summary.json" if missing else "summary.json")
out.write_text(json.dumps(report, indent=2) + "\n")
for key, group in groups.items():
    print(key, group["runs"], {m: round(group["means"][m], 3) for m in metrics[:5]})
