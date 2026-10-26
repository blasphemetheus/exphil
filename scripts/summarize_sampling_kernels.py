"""Render paired sampler timings without rerunning inference."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_runs/sampling_kernels_20261026")
lines = ["# Paired stochastic sampling timings", "",
         "Times are microseconds per full controller call (per batch for eight rows).",
         "100 fresh-key calls per scenario; compilation and explicit restore/reset calls excluded.",
         "Rolled and unrolled variants use identical checkpoints and sampler logic.", "",
         "| Client | Policy | Batch | Scenario | Replay rows | Rolled median / p95 | Unrolled median / p95 | Median speedup |",
         "|---|---|---:|---|---:|---:|---:|---:|"]
for client in ["host", "cuda"]:
    manifest = json.loads((root / f"{client}_manifest.json").read_text())
    assert manifest["hashes_verified_after"] and manifest["priming_controllers_exact"]
    for arch in ["min_gru", "gru", "mamba", "mamba2"]:
        for rows in [1, 8]:
            reports = [json.loads((root / f"{arch}_{client}_{rows}_{kernel}.json").read_text())
                       for kernel in ["rolled", "unrolled"]]
            assert reports[0]["priming_controller_sha256"] == reports[1]["priming_controller_sha256"]
            assert len(reports[0]["results"]) == len(reports[1]["results"])
            for old, new in zip(reports[0]["results"], reports[1]["results"]):
                assert (old["kind"], old["rows"]) == (new["kind"], new["rows"])
                speedup = old["median_us"] / new["median_us"]
                lines.append(f"| {client} | {arch} | {rows} | {old['kind']} | {old['rows']} | "
                             f"{old['median_us']} / {old['p95_us']} | "
                             f"{new['median_us']} / {new['p95_us']} | {speedup:.2f}x |")
(root / "TABLES.md").write_text("\n".join(lines) + "\n")
