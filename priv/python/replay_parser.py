#!/usr/bin/env python3
"""
ExPhil Replay Parser

Parses Slippi replay files (.slp) and converts them to training data format.
Uses slippi-py library for parsing.

Usage:
    # Parse single file
    python replay_parser.py parse replay.slp output.json

    # Parse directory
    python replay_parser.py parse_dir ./replays ./parsed

    # Parse with filters
    python replay_parser.py parse_dir ./replays ./parsed --character mewtwo --min-length 60

Protocol (when used as stdin/stdout pipe):
    Request:  {"cmd": "parse", "path": "replay.slp"}
    Response: {"ok": true, "frames": [...], "metadata": {...}}

Security Note:
    Output files use JSON format for safe serialization.
    The training Data module loads these via :erlang.binary_to_term after
    conversion in the Elixir layer.
"""

import json
import sys
import os
import gzip
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[replay_parser] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    from slippi import Game
    from slippi.id import InGameCharacter, Stage
except ImportError:
    logger.error("slippi-py not installed. Run: pip install py-slippi")
    sys.exit(1)


# Character ID mapping (slippi enum to int)
CHARACTER_IDS = {
    InGameCharacter.CAPTAIN_FALCON: 0,
    InGameCharacter.DONKEY_KONG: 1,
    InGameCharacter.FOX: 2,
    InGameCharacter.GAME_AND_WATCH: 3,
    InGameCharacter.KIRBY: 4,
    InGameCharacter.BOWSER: 5,
    InGameCharacter.LINK: 6,
    InGameCharacter.LUIGI: 7,
    InGameCharacter.MARIO: 8,
    InGameCharacter.MARTH: 9,
    InGameCharacter.MEWTWO: 10,
    InGameCharacter.NESS: 11,
    InGameCharacter.PEACH: 12,
    InGameCharacter.PIKACHU: 13,
    InGameCharacter.ICE_CLIMBERS: 14,
    InGameCharacter.JIGGLYPUFF: 15,
    InGameCharacter.SAMUS: 16,
    InGameCharacter.YOSHI: 17,
    InGameCharacter.ZELDA: 18,
    InGameCharacter.SHEIK: 19,
    InGameCharacter.FALCO: 20,
    InGameCharacter.YOUNG_LINK: 21,
    InGameCharacter.DR_MARIO: 22,
    InGameCharacter.ROY: 23,
    InGameCharacter.PICHU: 24,
    InGameCharacter.GANONDORF: 25,
}

# Reverse mapping for filtering
CHARACTER_NAMES = {v: k.name.lower() for k, v in CHARACTER_IDS.items()}
NAME_TO_CHARACTER = {name: char for char, name in CHARACTER_NAMES.items()}

# Stage ID mapping
STAGE_IDS = {
    Stage.FOUNTAIN_OF_DREAMS: 2,
    Stage.POKEMON_STADIUM: 3,
    Stage.YOSHIS_STORY: 8,
    Stage.DREAM_LAND_N64: 28,
    Stage.BATTLEFIELD: 31,
    Stage.FINAL_DESTINATION: 32,
}


def get_character_id(char: InGameCharacter) -> int:
    """Convert slippi character enum to numeric ID."""
    return CHARACTER_IDS.get(char, -1)


def get_stage_id(stage: Stage) -> int:
    """Convert slippi stage enum to numeric ID."""
    return STAGE_IDS.get(stage, 0)


def serialize_player(player_frame, port: int) -> Dict[str, Any]:
    """Convert slippi player frame to serializable dict."""
    if player_frame is None:
        return None

    pre = player_frame.pre
    post = player_frame.post

    return {
        "character": get_character_id(post.character) if post.character else -1,
        "x": float(post.position.x) if post.position else 0.0,
        "y": float(post.position.y) if post.position else 0.0,
        "percent": float(post.percent) if post.percent is not None else 0.0,
        "stock": post.stocks_remaining if post.stocks_remaining is not None else 0,
        "facing": 1 if post.facing and post.facing.value > 0 else -1,
        "action": post.state.value if post.state else 0,
        "action_frame": post.state_frame if post.state_frame is not None else 0,
        "invulnerable": post.is_invulnerable if post.is_invulnerable is not None else False,
        "jumps_left": post.jumps_remaining if post.jumps_remaining is not None else 0,
        "on_ground": post.is_grounded if post.is_grounded is not None else True,
        "shield_strength": float(post.shield) if post.shield is not None else 60.0,
        "hitstun_frames_left": post.hitstun_remaining if post.hitstun_remaining is not None else 0,
        "speed_air_x_self": float(post.speed_air_x_self) if post.speed_air_x_self is not None else 0.0,
        "speed_ground_x_self": float(post.speed_ground_x_self) if post.speed_ground_x_self is not None else 0.0,
        "speed_y_self": float(post.speed_y_self) if post.speed_y_self is not None else 0.0,
        "speed_x_attack": float(post.speed_x_attack) if post.speed_x_attack is not None else 0.0,
        "speed_y_attack": float(post.speed_y_attack) if post.speed_y_attack is not None else 0.0,
        "nana": None,  # TODO: Handle Ice Climbers
        "controller_state": serialize_controller(pre) if pre else None,
    }


def serialize_controller(pre_frame) -> Dict[str, Any]:
    """Extract controller state from pre-frame data."""
    if pre_frame is None:
        return None

    # Pre-frame contains the inputs that led to this frame
    joystick = pre_frame.joystick if pre_frame.joystick else (0.5, 0.5)
    cstick = pre_frame.cstick if pre_frame.cstick else (0.5, 0.5)
    triggers = pre_frame.triggers if pre_frame.triggers else (0.0, 0.0)
    buttons = pre_frame.buttons if pre_frame.buttons else None

    # Normalize joystick from [-1, 1] to [0, 1]
    main_x = (joystick.x + 1.0) / 2.0 if hasattr(joystick, 'x') else (joystick[0] + 1.0) / 2.0
    main_y = (joystick.y + 1.0) / 2.0 if hasattr(joystick, 'y') else (joystick[1] + 1.0) / 2.0
    c_x = (cstick.x + 1.0) / 2.0 if hasattr(cstick, 'x') else (cstick[0] + 1.0) / 2.0
    c_y = (cstick.y + 1.0) / 2.0 if hasattr(cstick, 'y') else (cstick[1] + 1.0) / 2.0

    # Get trigger values
    l_trigger = triggers.logical.l if hasattr(triggers, 'logical') else triggers[0]
    r_trigger = triggers.logical.r if hasattr(triggers, 'logical') else triggers[1]

    # Button states
    button_state = {
        "a": False, "b": False, "x": False, "y": False,
        "z": False, "l": False, "r": False, "d_up": False
    }

    if buttons:
        physical = buttons.physical if hasattr(buttons, 'physical') else buttons
        if hasattr(physical, 'a'):
            button_state["a"] = bool(physical.a)
            button_state["b"] = bool(physical.b)
            button_state["x"] = bool(physical.x)
            button_state["y"] = bool(physical.y)
            button_state["z"] = bool(physical.z)
            button_state["l"] = bool(physical.l)
            button_state["r"] = bool(physical.r)
            button_state["d_up"] = bool(getattr(physical, 'd_up', False))

    return {
        "main_stick": {"x": float(main_x), "y": float(main_y)},
        "c_stick": {"x": float(c_x), "y": float(c_y)},
        "l_shoulder": float(l_trigger) if l_trigger else 0.0,
        "r_shoulder": float(r_trigger) if r_trigger else 0.0,
        "button_a": button_state["a"],
        "button_b": button_state["b"],
        "button_x": button_state["x"],
        "button_y": button_state["y"],
        "button_z": button_state["z"],
        "button_l": button_state["l"],
        "button_r": button_state["r"],
        "button_d_up": button_state["d_up"],
    }


def serialize_item(item, player_port: int, opponent_port: int) -> Dict[str, Any]:
    """
    Convert slippi item/projectile to serializable dict.

    Items in Slippi include both regular items and projectiles (Fox lasers,
    Sheik needles, Link arrows, etc.). We convert them to a unified format.

    Args:
        item: Slippi Item object from frame.items
        player_port: The player whose perspective we're training from
        opponent_port: The opponent's port

    Returns:
        Dict with projectile data, or None if invalid
    """
    if item is None:
        return None

    try:
        # Get position
        x = float(item.position.x) if item.position else 0.0
        y = float(item.position.y) if item.position else 0.0

        # Get velocity
        speed_x = float(item.velocity.x) if item.velocity else 0.0
        speed_y = float(item.velocity.y) if item.velocity else 0.0

        # Get item type - convert enum to int
        item_type = item.type.value if hasattr(item.type, 'value') else int(item.type) if item.type else 0

        # Determine owner: spawn_id can help, but we need to infer from context
        # For projectiles, the spawn_id often correlates with the spawning player
        # However, the most reliable way is to check if the item was spawned by
        # a character - which requires tracking spawn frames
        # For now, we use a heuristic: lower spawn_ids tend to be player 1's projectiles
        # This isn't perfect but provides a starting point
        spawn_id = item.spawn_id if item.spawn_id is not None else 0

        # Note: Proper owner detection would require tracking which character
        # spawned each projectile, which is complex. For now we set owner=0
        # and let the embedding handle it with positional context
        owner = 0  # Will be improved in future versions

        return {
            "owner": owner,
            "x": x,
            "y": y,
            "type": item_type,
            "subtype": item.state if item.state is not None else 0,  # Action state as subtype
            "speed_x": speed_x,
            "speed_y": speed_y,
            "spawn_id": spawn_id,
            "timer": item.timer if item.timer is not None else 0,
            "damage": item.damage if item.damage is not None else 0,
        }
    except Exception as e:
        logger.debug(f"Failed to serialize item: {e}")
        return None


def parse_replay(path: str, player_port: Optional[int] = None) -> Dict[str, Any]:
    """
    Parse a single .slp replay file.

    Args:
        path: Path to .slp file
        player_port: Which player's perspective (1-4), or None for all

    Returns:
        Dict with 'frames', 'metadata', and 'success' keys
    """
    try:
        game = Game(path)
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
        return {"success": False, "error": str(e), "frames": [], "metadata": {}}

    if not game.frames:
        return {"success": False, "error": "No frames in replay", "frames": [], "metadata": {}}

    # Get metadata
    metadata = {
        "path": str(path),
        "stage": get_stage_id(game.start.stage) if game.start and game.start.stage else 0,
        "duration_frames": len(game.frames),
        "players": {},
    }

    # Get player info
    ports = []
    if game.start and game.start.players:
        for port_idx, player in enumerate(game.start.players):
            if player:
                port = port_idx + 1  # 1-indexed
                ports.append(port)
                metadata["players"][port] = {
                    "character": get_character_id(player.character) if player.character else -1,
                    "character_name": player.character.name if player.character else "unknown",
                    "tag": player.tag if hasattr(player, 'tag') else None,
                }

    if len(ports) < 2:
        return {"success": False, "error": "Need at least 2 players", "frames": [], "metadata": metadata}

    # Default to port 1 if not specified
    if player_port is None:
        player_port = ports[0]

    # Find opponent port
    opponent_port = [p for p in ports if p != player_port][0] if len(ports) > 1 else None

    metadata["player_port"] = player_port
    metadata["opponent_port"] = opponent_port

    # Parse frames
    frames = []
    for frame_num, frame in enumerate(game.frames):
        if frame is None:
            continue

        # Get player data
        player_data = None
        opponent_data = None

        for port in frame.ports:
            if port is None:
                continue
            port_num = port.port.value + 1 if hasattr(port, 'port') else ports[0]

            if port_num == player_port and port.leader:
                player_data = serialize_player(port.leader, port_num)
            elif port_num == opponent_port and port.leader:
                opponent_data = serialize_player(port.leader, port_num)

        if player_data is None:
            continue

        # Parse items/projectiles from frame
        projectiles = []
        if hasattr(frame, 'items') and frame.items:
            for item in frame.items:
                serialized = serialize_item(item, player_port, opponent_port)
                if serialized:
                    projectiles.append(serialized)

        # Create game state
        game_state = {
            "frame": frame_num,
            "stage": metadata["stage"],
            "menu_state": 2,  # IN_GAME
            "players": {
                str(player_port): player_data,  # JSON keys must be strings
            },
            "projectiles": projectiles,
            "distance": 0.0,
        }

        if opponent_data:
            game_state["players"][str(opponent_port)] = opponent_data
            # Calculate distance
            dx = player_data["x"] - opponent_data["x"]
            dy = player_data["y"] - opponent_data["y"]
            game_state["distance"] = (dx*dx + dy*dy) ** 0.5

        # Controller state (from the player we're learning from)
        controller = player_data.get("controller_state")

        frames.append({
            "game_state": game_state,
            "controller": controller,
            "metadata": {
                "player_port": player_port,
                "character": metadata["players"].get(player_port, {}).get("character", -1),
            }
        })

    return {
        "success": True,
        "frames": frames,
        "metadata": metadata,
    }


def save_parsed(result: Dict[str, Any], output_path: str, compress: bool = True):
    """Save parsed replay data to file using JSON (gzipped by default)."""
    if compress:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(result, f)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f)


def load_parsed(path: str) -> Dict[str, Any]:
    """Load parsed replay data from file."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def parse_directory(
    input_dir: str,
    output_dir: str,
    character: Optional[str] = None,
    min_length: int = 0,
    max_files: Optional[int] = None,
    player_port: Optional[int] = None,
    workers: int = 4,
    compress: bool = True,
) -> Dict[str, Any]:
    """
    Parse all .slp files in a directory.

    Args:
        input_dir: Directory containing .slp files
        output_dir: Directory to write parsed .json.gz files
        character: Filter by character name (e.g., 'mewtwo')
        min_length: Minimum game length in seconds
        max_files: Maximum number of files to process
        player_port: Which player's perspective to use
        workers: Number of parallel workers
        compress: Whether to gzip output files

    Returns:
        Summary statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .slp files
    slp_files = list(input_path.glob("**/*.slp"))
    if max_files:
        slp_files = slp_files[:max_files]

    logger.info(f"Found {len(slp_files)} .slp files")

    # Convert character filter
    char_filter = None
    if character:
        char_lower = character.lower()
        if char_lower in NAME_TO_CHARACTER:
            char_filter = NAME_TO_CHARACTER[char_lower]
        else:
            logger.warning(f"Unknown character: {character}")

    min_frames = min_length * 60  # 60 fps

    stats = {
        "total_files": len(slp_files),
        "parsed_files": 0,
        "skipped_files": 0,
        "total_frames": 0,
        "errors": [],
    }

    def process_file(slp_path: Path) -> Tuple[Optional[Path], Dict]:
        """Process a single file."""
        result = parse_replay(str(slp_path), player_port)

        if not result["success"]:
            return None, {"skipped": True, "reason": result.get("error", "unknown")}

        # Apply filters
        if len(result["frames"]) < min_frames:
            return None, {"skipped": True, "reason": "too_short"}

        if char_filter is not None:
            player_char = result["metadata"].get("players", {}).get(
                result["metadata"].get("player_port", 1), {}
            ).get("character", -1)
            if player_char != char_filter:
                return None, {"skipped": True, "reason": "wrong_character"}

        # Write output
        rel_path = slp_path.relative_to(input_path)
        suffix = ".json.gz" if compress else ".json"
        out_file = output_path / rel_path.with_suffix(suffix)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        save_parsed(result, str(out_file), compress=compress)

        return out_file, {"frames": len(result["frames"])}

    # Process files (parallel for speed)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_file, f): f for f in slp_files}
            for future in as_completed(futures):
                slp_path = futures[future]
                try:
                    out_path, info = future.result()
                    if out_path:
                        stats["parsed_files"] += 1
                        stats["total_frames"] += info.get("frames", 0)
                        logger.debug(f"Parsed: {slp_path.name} -> {out_path.name}")
                    else:
                        stats["skipped_files"] += 1
                except Exception as e:
                    stats["errors"].append({"file": str(slp_path), "error": str(e)})
                    logger.warning(f"Error processing {slp_path}: {e}")
    else:
        # Sequential processing
        for slp_path in slp_files:
            try:
                out_path, info = process_file(slp_path)
                if out_path:
                    stats["parsed_files"] += 1
                    stats["total_frames"] += info.get("frames", 0)
                else:
                    stats["skipped_files"] += 1
            except Exception as e:
                stats["errors"].append({"file": str(slp_path), "error": str(e)})

    logger.info(f"Parsed {stats['parsed_files']}/{stats['total_files']} files, "
                f"{stats['total_frames']} total frames")

    return stats


def handle_stdin():
    """Handle JSON commands from stdin (for Elixir Port communication)."""
    for line in sys.stdin:
        try:
            cmd = json.loads(line.strip())
            command = cmd.get("cmd")

            if command == "parse":
                path = cmd.get("path")
                port = cmd.get("player_port")
                result = parse_replay(path, port)
                print(json.dumps({"ok": True, **result}), flush=True)

            elif command == "parse_dir":
                result = parse_directory(
                    cmd.get("input_dir"),
                    cmd.get("output_dir"),
                    character=cmd.get("character"),
                    min_length=cmd.get("min_length", 0),
                    max_files=cmd.get("max_files"),
                    player_port=cmd.get("player_port"),
                    workers=cmd.get("workers", 1),  # Single worker for pipe mode
                )
                print(json.dumps({"ok": True, **result}), flush=True)

            else:
                print(json.dumps({"error": f"Unknown command: {command}"}), flush=True)

        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Parse Slippi replay files")
    subparsers = parser.add_subparsers(dest="command")

    # Parse single file
    parse_cmd = subparsers.add_parser("parse", help="Parse single replay file")
    parse_cmd.add_argument("input", help="Input .slp file")
    parse_cmd.add_argument("output", help="Output .json or .json.gz file")
    parse_cmd.add_argument("--port", type=int, help="Player port (1-4)")
    parse_cmd.add_argument("--no-compress", action="store_true", help="Don't gzip output")

    # Parse directory
    dir_cmd = subparsers.add_parser("parse_dir", help="Parse directory of replays")
    dir_cmd.add_argument("input", help="Input directory")
    dir_cmd.add_argument("output", help="Output directory")
    dir_cmd.add_argument("--character", "-c", help="Filter by character name")
    dir_cmd.add_argument("--min-length", "-l", type=int, default=0,
                        help="Minimum game length in seconds")
    dir_cmd.add_argument("--max-files", "-n", type=int, help="Maximum files to process")
    dir_cmd.add_argument("--port", type=int, help="Player port (1-4)")
    dir_cmd.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel workers")
    dir_cmd.add_argument("--no-compress", action="store_true", help="Don't gzip output")

    # Pipe mode (stdin/stdout)
    subparsers.add_parser("pipe", help="Run in pipe mode (stdin/stdout JSON)")

    # List characters
    subparsers.add_parser("characters", help="List available character names")

    args = parser.parse_args()

    if args.command == "parse":
        result = parse_replay(args.input, args.port)
        if result["success"]:
            compress = not args.no_compress
            save_parsed(result, args.output, compress=compress)
            print(f"Parsed {len(result['frames'])} frames -> {args.output}")
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "parse_dir":
        stats = parse_directory(
            args.input,
            args.output,
            character=args.character,
            min_length=args.min_length,
            max_files=args.max_files,
            player_port=args.port,
            workers=args.workers,
            compress=not args.no_compress,
        )
        print(json.dumps(stats, indent=2))

    elif args.command == "pipe":
        handle_stdin()

    elif args.command == "characters":
        print("Available characters:")
        for name in sorted(NAME_TO_CHARACTER.keys()):
            print(f"  {name}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
