#!/usr/bin/env python3
"""Descriptive style comparison, not an automatic R3 promotion gate."""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATURES = ("jump_x_ratio", "cstick_aerial_frac", "short_hop_frac", "lightshield_frac",
            "aerial_per_min", "grab_per_min", "roll_forward_per_min", "roll_backward_per_min",
            "spotdodge_per_min", "dashdance_per_min", "wavedash_per_min", "airdodge_per_min")


def quantile(values, fraction):
    xs = sorted(values)
    if not xs:
        return None
    position = (len(xs) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (position - lo)


def make_report(folder):
    rows = [json.loads(line) for line in (folder / "fingerprints.jsonl").read_text().splitlines()]
    humans = {key: [] for key in FEATURES}
    sources = [ROOT / "eval_runs/0917_style_identity" / name for name in ("erickfm_fox.jsonl", "yeti_fox.jsonl")]
    for source in sources:
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("character") != "Fox":
                    continue
                for key in FEATURES:
                    value = row.get("features", {}).get(key)
                    if isinstance(value, (float, int)):
                        humans[key].append(value)
    result = {"evaluation": str(folder.relative_to(ROOT)), "status": "descriptive_only",
              "caveat": "Sim fingerprints cover only the first 1800 in-game frames; human corpus covers full games with different opponents/stages. Quantiles below are descriptive, not preregistered acceptance thresholds.",
              "human_sources": [str(p.relative_to(ROOT)) for p in sources], "features": {}}
    for key in FEATURES:
        means = {}
        for role in ("candidate", "prior"):
            values = [r["features"][key] for r in rows if r["role"] == role and key in r["features"]]
            means[role] = {"n": len(values), "mean": statistics.mean(values) if values else None}
        result["features"][key] = {**means, "human_n": len(humans[key]),
            "human_median": quantile(humans[key], 0.5), "human_p05": quantile(humans[key], 0.05),
            "human_p95": quantile(humans[key], 0.95)}
    (folder / "style_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1]).resolve()
    make_report(folder)
    print(f"Wrote {folder / 'style_comparison.json'}")
