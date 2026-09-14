"""Assemble existing multishine evidence without loading a model or launching Mix."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAND = re.compile(r"^ep(\d+): ([\d.eE+-]+)/min chain (\d+)$")
CPU = re.compile(r"^ep(\d+)\s+(?:\[[^]]+\]\s+)?r(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+(\d+)$")


def capture(path, root):
    resolved = (root / path).resolve()
    evidence = {"path": str(resolved)}
    try:
        before = resolved.stat()
        content = resolved.read_bytes()
        after = resolved.stat()
        signature = lambda stat: (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature(before) != signature(after):
            return {**evidence, "status": "changing"}, None
        return {**evidence, "status": "present", "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}, content
    except FileNotFoundError:
        return {**evidence, "status": "pending"}, None
    except OSError as error:
        return {**evidence, "status": "unreadable", "error": str(error)}, None


def summarize(kind, content, checkpoint, root):
    text = ANSI.sub("", content.decode("utf-8"))
    if kind in ("stand", "cpu"):
        epoch = checkpoint.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise ValueError("Gate evidence requires an explicit integer epoch")
        selected = []
        for line in text.splitlines():
            match = (STAND if kind == "stand" else CPU).fullmatch(line.strip())
            if not match or int(match[1]) != epoch:
                continue
            if kind == "stand":
                row = {"reported_rate_per_minute": float(match[2]), "max_chain": int(match[3])}
            else:
                row = {"run": int(match[2]), "frames": int(match[3]), "shines": int(match[4]),
                       "self_shines": int(match[5]), "hit_associated_shines": int(match[6]),
                       "self_per_minute": float(match[7]), "max_chain": int(match[8])}
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError("Non-finite gate metric")
            selected.append(row)
        if not selected:
            return {"status": "no_matching_epoch", "rows": []}
        duplicate = len(selected) > 1 if kind == "stand" else len({row["run"] for row in selected}) < len(selected)
        return {"status": "ambiguous" if duplicate else "parsed", "rows": selected}

    if kind in ("audit", "execution"):
        pattern = r"shift|conflict|ambig|audit" if kind == "audit" else r"latency|reaction.delay|handoff|warm|observe|diverg|error"
        lines = [line for line in text.splitlines() if re.search(pattern, line, re.I)]
        return {"status": "text_evidence", "lines": lines[-80:],
                "verification": "Log excerpts only; completion and correctness are not inferred"}

    data = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    policy = data.get("policy")
    expected = (root / checkpoint["policy"]).resolve()
    if policy is not None and (root / policy).resolve() != expected:
        return {"status": "policy_mismatch", "reported_policy": policy}
    if kind == "coverage":
        if policy is None:
            raise ValueError("Coverage map has no policy identity")
        baseline = [row for row in data["rows"] if row["axis"] == "baseline"]
        if len(baseline) != 1:
            raise ValueError("Coverage map must contain one baseline")
        worst = sorted(data["rows"], key=lambda row: row["mean"])[:8]
        return {"status": "parsed", "delay_id": data.get("delay_id"),
                "label_offset": data.get("label_offset"), "temperature": data.get("temperature"),
                "baseline": baseline[0], "weakest_cells": worst}
    if kind == "corrections":
        if policy is None:
            raise ValueError("Correction scoreboard has no policy identity")
        return {"status": "parsed", "summary": data["summary"],
                "diverged_runs": data["diverged_runs"], "errored_runs": data["errored_runs"],
                "input_offset": data.get("input_offset"), "response_delay": data.get("response_delay")}
    if kind == "recovery":
        runs = [run for run in data["runs"] if (root / run["policy"]).resolve() == expected]
        return {"status": "parsed" if runs else "no_matching_policy",
                "metric": data.get("metric"),
                "runs": [{"id": run["id"], "scenario": run["scenario"],
                          "recovery": run["metrics"]["recovery"]} for run in runs]}
    raise ValueError(f"Unknown evidence kind: {kind}")


def build_report(manifest, root):
    if manifest.get("version") != 1 or not manifest.get("checkpoints"):
        raise ValueError("Expected version 1 and nonempty checkpoints")
    ids = [checkpoint["id"] for checkpoint in manifest["checkpoints"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate checkpoint IDs")
    checkpoints = []
    for checkpoint in manifest["checkpoints"]:
        identity, _content = capture(checkpoint["policy"], root)
        evidence = {}
        for kind in ("audit", "coverage", "stand", "cpu", "corrections", "recovery", "execution"):
            path = checkpoint.get(kind)
            if path is None:
                evidence[kind] = {"status": "not_supplied"}
                continue
            item, content = capture(path, root)
            if content is not None:
                try:
                    item.update(summarize(kind, content, checkpoint, root))
                except (ValueError, KeyError, TypeError, AttributeError, UnicodeError) as error:
                    item.update(status="invalid", error=str(error))
            evidence[kind] = item
        checkpoints.append({"id": checkpoint["id"], "epoch": checkpoint.get("epoch"),
                            "policy": identity, "evidence": evidence})
    return {"version": 1, "checkpoints": checkpoints,
            "interpretation": "Evidence inventory only. No ranking or promotion. Historical policy associations and protocol settings are declared; missing evidence is not a passing result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    content = args.manifest.read_bytes()
    report = build_report(json.loads(content), args.root.resolve())
    report["manifest_sha256"] = hashlib.sha256(content).hexdigest()
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    for checkpoint in report["checkpoints"]:
        states = ", ".join(f"{kind}={item['status']}" for kind, item in checkpoint["evidence"].items())
        print(f"{checkpoint['id']}: {states}")


if __name__ == "__main__":
    main()
