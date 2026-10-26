"""Validate and summarize the fixed October 26 full-controller profile matrix."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_runs/stochastic_controllers_20261026")
names = {"min_gru": "MinGRU", "gru": "GRU", "mamba": "Mamba", "mamba2": "Mamba-2"}
reports = {}
for client in ["host", "cuda"]:
    manifest = json.loads((root / f"{client}_manifest.json").read_text())
    assert manifest["hashes_verified_after"] and len(manifest["runs"]) == 16
    for path, expected in manifest["hashes"].items():
        with open(path, "rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, path
    for name in names:
        for rows in [1, 8]:
            for mode in ["deterministic", "stochastic"]:
                report = json.loads((root / f"{name}_{client}_{rows}_{mode}.json").read_text())
                assert report["stochastic"] == (mode == "stochastic")
                assert report["client"] == client and report["rows"] == rows
                assert report["warmup_preserves_state"]
                assert report["rollback"] == ("public_snapshot_exact" if rows == 1 else "internal_batch_fixture_exact")
                for result in report["results"]:
                    times = sorted(result["times_us"])
                    assert len(times) == result["samples"] == 100
                    assert times[50] == result["median_us"] and times[94] == result["p95_us"]
                    if mode == "deterministic":
                        assert result["distinct_controller_groups"] == 1
                if mode == "stochastic":
                    assert report["results"][0]["distinct_controller_groups"] > 1
                reports[name, client, rows, mode] = report

def result(name, client, rows, mode="stochastic", kind="ordinary", due=0):
    return next((r for r in reports[name, client, rows, mode]["results"]
                 if r["kind"] == kind and r["rows"] == due), None)

def cell(value):
    return "—" if value is None else f"{value['median_us']/1000:.3f} / {value['p95_us']/1000:.3f}"

lines = ["# Stochastic recurrent controller benchmarks — October 26, 2026", "",
    "All entries below are **median / p95 milliseconds for the whole request**, not per row.", "",
    "## Steady recurrent inference", "",
    "| Policy | CPU, 1 policy | CPU, 8 policies | CUDA, 1 policy | CUDA, 8 policies |",
    "|---|---:|---:|---:|---:|"]
for name, label in names.items():
    lines.append("| " + label + " | " + " | ".join(
        cell(result(name, client, rows)) for client in ["host", "cuda"] for rows in [1, 8]) + " |")
lines += ["", "## Deterministic versus stochastic, CUDA", "",
          "| Policy | Rows | Deterministic | Stochastic |", "|---|---:|---:|---:|"]
for name, label in names.items():
    for rows in [1, 8]:
        lines.append(f"| {label} | {rows} | {cell(result(name, 'cuda', rows, 'deterministic'))} | "
                     f"{cell(result(name, 'cuda', rows))} |")
lines += ["", "## Eight-policy stochastic replay and full-window costs", "",
          "| Policy | Backend | Reset 1 row | Reset all | Resync 1 row | Resync all | Windowed |",
          "|---|---|---:|---:|---:|---:|---:|"]
for name, label in names.items():
    for client in ["host", "cuda"]:
        values = [cell(result(name, client, 8, kind=kind, due=due))
                  for kind, due in [("cold_reset", 1), ("cold_reset", 8),
                                    ("resync", 1), ("resync", 8), ("windowed", 8)]]
        lines.append(f"| {label} | {client} | " + " | ".join(values) + " |")
lines += ["", "GRU is BPTT-trained: its valid deployment path retains carry. Periodic resync and",
          "windowed recomputation are deliberately absent.", ""]
(root / "TABLES.md").write_text("\n".join(lines))
print("Verified 32 profiles, sample statistics, sampling diversity, warmup/rollback metadata, and source hashes.")
