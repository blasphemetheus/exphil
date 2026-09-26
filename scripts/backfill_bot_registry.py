#!/usr/bin/env python3
"""Backfill recent saved bots into the existing v1 registry, with evidence.

prepare -> read-only Elixir metadata extraction -> apply. Existing entries
are preserved; content-identical new policy files become aliases. All added
metadata lives in training_config.provenance, which Registry preserves.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / "eval_runs/0923_registry_backfill"
REGISTRY = ROOT / "checkpoints/registry.json"
MANIFEST = ROOT / "docs/reference/RECENT_BOTS.json"
CATALOG = ROOT / "docs/reference/RECENT_BOTS.md"
CUTOFF = dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc).timestamp()
FOX_CUTOFF = dt.datetime(2026, 8, 25, tzinfo=dt.timezone.utc).timestamp()
PRIOR = "checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin"
HANDOFF = "docs/planning/HANDOFF_2026-09-18.md"
OVERRIDES = {
    "checkpoints/fox_v3_1_20260919_022008_ep3": (
        "checkpoints/fox_v3_1_20260919_022008/model_epoch2.axon", HANDOFF + " (epochs 3–4 resume entry)"),
    "checkpoints/fox_v3_1_20260919_022008": (
        "checkpoints/fox_v3_20260917_050901_resume3/model_batch215000.axon", HANDOFF + " (V3.1 launch entry)"),
}


def rel(path):
    p = pathlib.Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    return str(p)


def digest(path):
    with pathlib.Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def sidecar(path):
    p = ROOT / path
    if p.name.endswith("_policy.bin"):
        exact = p.with_name(p.name.replace("_policy.bin", "_config.json"))
        generic = p.parent / "model_config.json"
        # Flat training exports often share the run's config with the best snapshot.
        run = p.with_name(re.sub(r"_best_policy\.bin$", "_config.json", p.name))
        for candidate in (exact, generic, run):
            if candidate.suffix == ".json" and candidate.exists():
                return rel(candidate), read_json(candidate)
    args = p.parent / "train_args.json"
    if args.exists():
        values = read_json(args)
        if isinstance(values, list):
            cfg = {}
            for i, word in enumerate(values):
                if isinstance(word, str) and word.startswith("--"):
                    cfg[word[2:].replace("-", "_")] = values[i + 1] if i + 1 < len(values) and not str(values[i + 1]).startswith("--") else True
            return rel(args), cfg
        if isinstance(values, dict):
            return rel(args), values
    return None, {}


def parent_evidence(path, meta):
    if meta.get("parent_path"):
        return rel(meta["parent_path"]), path + " embedded policy field", []
    p = pathlib.Path(path)
    if str(p.parent) in OVERRIDES:
        parent, evidence = OVERRIDES[str(p.parent)]
        return parent, evidence, ["Copied sidecar is not authoritative for this run; parent recovered from dated handoff."]
    source, cfg = sidecar(path)
    declared = cfg.get("checkpoint_path") or cfg.get("checkpoint")
    if declared and pathlib.Path(rel(declared)).parent != p.parent:
        return None, None, [f"Sidecar {source} describes a different directory ({rel(declared)}); not used for lineage."]
    for field in ("resume", "init_policy", "init", "initial_policy"):
        value = cfg.get(field)
        if isinstance(value, str) and value and (ROOT / rel(value)).is_file():
            return rel(value), f"{source} field {field}", []
    return None, None, ["Parent not established from available records; no generation-number inference."]


def discover():
    paths = set()
    for p in (ROOT / "checkpoints").rglob("*.bin"):
        modified = p.stat().st_mtime
        if p.name.endswith("_policy.bin") and modified >= FOX_CUTOFF:
            paths.add(rel(p))
        elif re.match(r"(?:ms_|mc_).+\.bin$", p.name) and modified >= CUTOFF:
            paths.add(rel(p))
    for name in ("ms_g19_ep4.bin", "ms_g15_oppmask_full.bin", "mc_g1_mdq_ss.bin"):
        if (ROOT / "checkpoints" / name).exists():
            paths.add("checkpoints/" + name)
    for folder in (ROOT / "eval_runs").glob("09*"):
        for p in folder.rglob("*.bin"):
            if p.name in ("candidate.bin", "candidate_policy.bin") or re.fullmatch(r"head_iter\d+\.bin", p.name):
                paths.add(rel(p))
    # Bring in exact resume ancestors, even if no policy export was retained.
    pending = list(paths)
    while pending:
        path = pending.pop()
        parent, _, _ = parent_evidence(path, {})
        if parent and parent not in paths and (ROOT / parent).is_file():
            paths.add(parent)
            pending.append(parent)
    return sorted(paths)


def artifact(path, role):
    p = ROOT / path
    return {"path": path, "role": role, "sha256": digest(p), "bytes": p.stat().st_size,
            "file_mtime": dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat()}


def family(path):
    p = pathlib.Path(path)
    if p.parts[0] == "eval_runs" or p.parent != pathlib.Path("checkpoints"):
        return str(p.parent)
    return "checkpoints/" + re.sub(r"_(?:ep\d+|latest|best_policy|policy)$", "", p.stem)


def validate(registry):
    ids = [m["id"] for m in registry["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model IDs")
    models = {m["id"]: m for m in registry["models"]}
    for model in models.values():
        seen = {model["id"]}
        parent = model.get("parent_id")
        while parent:
            if parent not in models:
                raise ValueError(f"Missing parent {parent}")
            if parent in seen:
                raise ValueError(f"Lineage cycle at {parent}")
            seen.add(parent)
            parent = models[parent].get("parent_id")


def apply_backfill():
    original_bytes = REGISTRY.read_bytes()
    registry = json.loads(original_bytes)
    old = json.loads(original_bytes)
    metadata = [json.loads(line) for line in (WORK / "metadata.jsonl").read_text().splitlines()]
    errors = [m for m in metadata if "error" in m]
    metadata = [m for m in metadata if "error" not in m]
    by_path = {m["path"]: m for m in metadata}
    existing_paths = {rel(m[k]): m for m in registry["models"] for k in ("checkpoint_path", "policy_path") if m.get(k)}
    for m in registry["models"]:
        for a in m.get("training_config", {}).get("provenance", {}).get("artifacts", []):
            existing_paths[a["path"]] = m
    groups = collections.defaultdict(list)
    for meta in metadata:
        a = artifact(meta["path"], meta["kind"])
        groups[a["sha256"]].append((meta, a))
    additions = []
    registrations = {}
    for sha, group in sorted(groups.items()):
        group.sort(key=lambda pair: ("latest" in pair[0]["path"], pair[0]["path"]))
        meta, primary = group[0]
        path = meta["path"]
        known = next((existing_paths[m["path"]] for m, _ in group if m["path"] in existing_paths), None)
        if known:
            for m, _ in group:
                registrations[m["path"]] = known
            continue  # Never rewrite legacy entries during this backfill.
        cfg = dict(meta.get("config", {}))
        source, side = sidecar(path)
        parent, evidence, notes = parent_evidence(path, meta)
        char = cfg.get("train_character") or side.get("train_character")
        if not char:
            char = "mewtwo" if "mewtwo" in path or "/mc_" in path else "fox" if any(s in path for s in ("fox", "/ms_", "ppo", "local_zero")) else "unknown"
        character_source = "embedded/sidecar train_character" if cfg.get("train_character") or side.get("train_character") else "artifact family convention; verify before cross-character transfer"
        method = "ppo" if meta["kind"] == "ppo_head" else "search_distillation" if "step8_mix" in path else "imitation" if "fox_" in path else "drill_imitation" if "/ms_" in path or "train_args" in str(source) else "unknown"
        if meta["kind"] == "ppo_head":
            cfg = dict(by_path.get(parent, {}).get("config", {}))
        # Embedded inference metadata wins; do not import stale sidecar fields.
        cfg.update({"train_character": char, "training_method": method})
        artifacts = [a for _, a in group]
        trainer = pathlib.Path(path).with_name(f"trainer_iter{meta.get('iteration')}.bin")
        if meta["kind"] == "ppo_head" and (ROOT / trainer).exists():
            artifacts.append(artifact(str(trainer), "trainer_state"))
        provenance = {"backfill": "recent-bots-2026-09-23", "family": family(path), "kind": meta["kind"],
            "artifacts": artifacts, "metadata_source": path, "sidecar_source": source,
            "character_source": character_source, "parent_path": parent, "parent_evidence": evidence,
            "lineage_status": "pending_resolution" if parent else "unknown", "notes": notes,
            "created_at_source": "file mtime; not a claimed training-start timestamp"}
        if "iteration" in meta:
            provenance["iteration"] = meta["iteration"]
        cfg["provenance"] = provenance
        name = path.removesuffix(".bin").removesuffix(".axon").replace("checkpoints/", "").replace("eval_runs/", "").replace("/", ":")
        if meta["kind"] == "ppo_head":
            name = f"fox-gru-ppo-{pathlib.Path(path).parent.name}-i{meta['iteration']}"
        entry = {"id": "bot_" + sha[:20], "name": name, "checkpoint_path": path,
            "policy_path": path if meta["kind"] == "policy" else None, "config_path": source,
            "created_at": primary["file_mtime"], "training_config": cfg, "metrics": {},
            "tags": [char, str(cfg.get("backbone", "unknown")), method, "backfilled", meta["kind"]], "parent_id": None}
        additions.append(entry)
        for m, _ in group:
            registrations[m["path"]] = entry
    registry["models"].extend(additions)
    for entry in additions:
        prov = entry["training_config"]["provenance"]
        parent = registrations.get(prov["parent_path"]) or existing_paths.get(prov["parent_path"])
        if parent and parent["id"] != entry["id"]:
            entry["parent_id"] = parent["id"]
            prov["lineage_status"] = "documented"
        elif prov["parent_path"]:
            prov["lineage_status"] = "unresolved"
    # Attach full exported policies to the precise head named in their eval summary.
    # They retain their own entries/file hashes; the parent link records materialization.
    for entry in additions:
        path = entry["policy_path"]
        if not path or pathlib.Path(path).name != "candidate_policy.bin":
            continue
        summary_path = ROOT / pathlib.Path(path).parent / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        parent = registrations.get(summary.get("head"))
        if parent:
            entry["parent_id"] = parent["id"]
            cfg = entry["training_config"]
            cfg["training_method"] = "ppo_export"
            cfg["provenance"].update(parent_path=summary["head"], parent_evidence=rel(summary_path), lineage_status="documented", notes=["Full policy materialized from the saved PPO head plus its frozen prior trunk."])
            entry["tags"] = ["fox", "gru", "ppo", "ppo_export", "backfilled", "policy"]
            if "outcomes" in summary:
                entry["metrics"]["evaluation"] = summary
                entry["metrics"]["evaluation_source"] = rel(summary_path)
    latest = registrations.get("eval_runs/0923_ppo/v1_fixed/head_iter200.bin")
    if latest and latest in additions:
        latest["metrics"]["evaluations"] = [
            {"source": p, "results": {k: v for k, v in read_json(ROOT / p).items() if k != "rows"}}
            for p in ("eval_runs/0923_ppo/eval_candidate/summary.json", "eval_runs/0923_ppo/eval_iter200/summary.json") if (ROOT / p).exists()]
        latest["metrics"]["human_report"] = {"source": "user report in Codex conversation, 2026-09-23", "consecutive_wins_reported": 4, "replay_verified": False, "quote": "best version I ever played"}
        latest["metrics"]["promotion_status"] = "not_promoted; style/transfer validation incomplete"
    validate(registry)
    assert registry["models"][:len(old["models"])] == old["models"], "Legacy entry changed"
    if REGISTRY.read_bytes() != original_bytes:
        raise RuntimeError("Registry changed concurrently; rerun against the latest copy")
    backup = WORK / "registry.before.json"
    if not backup.exists():
        backup.write_bytes(original_bytes)
    tmp = REGISTRY.with_suffix(".json.backfill.tmp")
    tmp.write_text(json.dumps(registry, indent=2) + "\n")
    tmp.replace(REGISTRY)
    backfilled = [m for m in registry["models"] if m.get("training_config", {}).get("provenance", {}).get("backfill") == "recent-bots-2026-09-23"]
    snapshot = {"schema": "existing-registry-v1-backfill", "scope": "Policy exports since Aug 25; drill checkpoints since Sep 10; September eval candidates/PPO heads; documented ancestors and current champions.",
                "models": backfilled, "unreadable_artifacts": errors}
    MANIFEST.write_text(json.dumps(snapshot, indent=2) + "\n")
    render_catalog(snapshot)
    print(json.dumps({"added": len(additions), "total": len(registry["models"]), "backfilled": len(backfilled), "unreadable": len(errors), "linked": sum(bool(m["parent_id"]) for m in backfilled)}, indent=2))


def render_catalog(snapshot):
    models = snapshot["models"]
    groups = collections.defaultdict(list)
    for m in models:
        groups[m["training_config"]["provenance"]["family"]].append(m)
    lines = ["# Recent bot catalog", "", "Generated by `scripts/backfill_bot_registry.py`. Live registry: `checkpoints/registry.json`.",
        "Tracked snapshot with IDs, hashes, exact paths, lineage evidence and evaluation results: [RECENT_BOTS.json](RECENT_BOTS.json).", "",
        snapshot["scope"], "", "GRU is an architecture; PPO is a training method. No bot is promoted by this inventory.",
        "Dates below come from file mtimes. Missing parents remain unknown; version numbers do not establish ancestry.", "",
        f"{len(models)} backfilled records across {len(groups)} families. Identical files are aliases within a record.", "",
        "## The Fox PPO bot played by the user", "",
        "Playable export: `eval_runs/0923_ppo/eval_candidate/candidate_policy.bin`.",
        "Training head: `eval_runs/0923_ppo/v1_fixed/head_iter200.bin`; frozen prior: `checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin`.",
        "Two simulator evaluations: 180W/17L/3D and 182W/18L in 200 games each. User reported four consecutive losses; not replay-verified.",
        "Style gate remains incomplete. This is not a production promotion.", "", "## Families", "",
        "| Family | Character | Architecture | Method | Saved versions | Documented parents |", "| --- | --- | --- | --- | ---: | ---: |"]
    for fam, rows in sorted(groups.items()):
        cfgs = [m["training_config"] for m in rows]
        vals = lambda key: ", ".join(sorted({str(c.get(key, "unknown")) for c in cfgs}))
        lines.append(f"| `{fam}` | {vals('train_character')} | {vals('backbone')} | {vals('training_method')} | {len(rows)} | {sum(bool(m['parent_id']) for m in rows)} |")
    lines += ["", "## Versions", "", "| Name | ID | Parent ID | Artifact |", "| --- | --- | --- | --- |"]
    for m in sorted(models, key=lambda m: m["name"]):
        lines.append(f"| `{m['name']}` | `{m['id']}` | {m['parent_id'] or 'unknown'} | `{m['policy_path'] or m['checkpoint_path']}` |")
    lines += ["", "## Rebuild / extend", "", "```bash", "python3 scripts/backfill_bot_registry.py prepare",
        "devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/registry_metadata.exs eval_runs/0923_registry_backfill/paths.json eval_runs/0923_registry_backfill/metadata.jsonl",
        "python3 scripts/backfill_bot_registry.py apply", "```", "",
        "Re-running preserves existing entries and IDs. The original registry backup is in `eval_runs/0923_registry_backfill/registry.before.json`.",
        "Model weights are never moved or edited. Fatal batches, critic-only weights, replay data and optimizer files are not presented as playable bots."]
    CATALOG.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "apply"])
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        paths = discover()
        (WORK / "paths.json").write_text(json.dumps(paths, indent=2) + "\n")
        print(f"Selected {len(paths)} artifacts for read-only inspection")
    else:
        apply_backfill()
