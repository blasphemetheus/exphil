#!/usr/bin/env python3
"""
Fast Slippi replay scanner - extracts character metadata without full parsing.

Usage:
    python replay_scanner.py /path/to/replays
    python replay_scanner.py /path/to/replays --max-files 1000

Outputs JSON with character counts and matchup data.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    from slippi import Game
    HAS_SLIPPI = True
except ImportError:
    HAS_SLIPPI = False


def scan_replay(path: str) -> dict | None:
    """Extract character info from a single replay file."""
    try:
        game = Game(path)

        players = []
        for i, player in enumerate(game.start.players):
            if player is not None:
                players.append({
                    "port": i,
                    "character": player.character.value if player.character else None,
                    "type": player.type.value if player.type else None
                })

        # Only return human vs human games (type 0 = human)
        human_players = [p for p in players if p.get("type") == 0]
        if len(human_players) < 2:
            return None

        return {
            "path": path,
            "characters": [p["character"] for p in human_players if p["character"] is not None],
            "stage": game.start.stage.value if game.start.stage else None,
            "duration": len(game.frames) if hasattr(game, 'frames') else 0
        }
    except Exception as e:
        return None


def scan_directory(dir_path: str, max_files: int | None = None) -> dict:
    """Scan a directory of replay files."""
    results = {
        "scanned_files": 0,
        "error_count": 0,
        "character_counts": defaultdict(int),
        "matchup_counts": defaultdict(int),
        "stage_counts": defaultdict(int)
    }

    # Find all .slp files
    files = list(Path(dir_path).rglob("*.slp"))
    if max_files:
        files = files[:max_files]

    total = len(files)

    for i, path in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"Scanning: {i + 1}/{total}", file=sys.stderr)

        result = scan_replay(str(path))
        if result:
            results["scanned_files"] += 1

            # Count each character appearance
            for char_id in result["characters"]:
                results["character_counts"][char_id] += 1

            # Count matchups (sorted tuple for consistency)
            if len(result["characters"]) == 2:
                matchup = tuple(sorted(result["characters"]))
                results["matchup_counts"][f"{matchup[0]}-{matchup[1]}"] += 1

            # Count stages
            if result["stage"]:
                results["stage_counts"][result["stage"]] += 1
        else:
            results["error_count"] += 1

    # Convert defaultdicts to regular dicts for JSON
    results["character_counts"] = dict(results["character_counts"])
    results["matchup_counts"] = dict(results["matchup_counts"])
    results["stage_counts"] = dict(results["stage_counts"])

    return results


def main():
    parser = argparse.ArgumentParser(description="Scan Slippi replays for character data")
    parser.add_argument("directory", help="Directory containing .slp files")
    parser.add_argument("--max-files", type=int, help="Maximum files to scan")
    args = parser.parse_args()

    if not HAS_SLIPPI:
        print(json.dumps({
            "error": "py-slippi not installed. Run: pip install py-slippi",
            "scanned_files": 0,
            "character_counts": {},
            "matchup_counts": {}
        }))
        sys.exit(0)

    if not os.path.isdir(args.directory):
        print(json.dumps({
            "error": f"Directory not found: {args.directory}",
            "scanned_files": 0,
            "character_counts": {},
            "matchup_counts": {}
        }))
        sys.exit(1)

    results = scan_directory(args.directory, args.max_files)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
