#!/usr/bin/env python3
"""
Filter low-tier character replays from a 7z archive.

Extracts files in batches, checks for target characters, keeps only relevant
replays organized by character. Designed for disk-constrained environments.

Usage:
    python filter_replays_from_archive.py <archive.7z> <output_dir> [--batch-size 100]

Target characters: Mewtwo, Ganondorf, Link, Zelda, Mr. Game & Watch
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Try to import slippi parser
try:
    from slippi import Game
    HAS_SLIPPI = True
except ImportError:
    HAS_SLIPPI = False
    print("Warning: py-slippi not installed. Install with: pip install py-slippi")

# Target characters (internal IDs from Melee)
# Reference: https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md
TARGET_CHARACTERS = {
    10: "mewtwo",
    22: "ganondorf", 
    5: "link",
    18: "zelda",
    24: "game_and_watch",
}

# Also check external character IDs (some parsers use these)
TARGET_CHARACTERS_EXTERNAL = {
    0x10: "mewtwo",       # 16
    0x19: "ganondorf",    # 25
    0x06: "link",         # 6
    0x13: "zelda",        # 19
    0x18: "game_and_watch", # 24
}

def get_characters_from_replay(filepath):
    """
    Extract character IDs from a replay file.
    Returns list of (port, character_name) for target characters found.
    """
    if not HAS_SLIPPI:
        return []
    
    try:
        game = Game(str(filepath))
        found = []
        
        for port, player in enumerate(game.metadata.players or []):
            if player is None:
                continue
            
            # Get character from player metadata
            char_id = None
            if hasattr(player, 'characters') and player.characters:
                # characters is a dict of {char_id: frames_played}
                char_id = max(player.characters.keys(), key=lambda k: player.characters[k])
            
            if char_id is not None:
                # Check both internal and external IDs
                if char_id in TARGET_CHARACTERS:
                    found.append((port, TARGET_CHARACTERS[char_id]))
                elif char_id in TARGET_CHARACTERS_EXTERNAL:
                    found.append((port, TARGET_CHARACTERS_EXTERNAL[char_id]))
        
        return found
    except Exception as e:
        # Corrupted or unreadable replay
        return []

def list_archive_files(archive_path):
    """List all .slp files in the 7z archive."""
    result = subprocess.run(
        ["7z", "l", "-slt", archive_path],
        capture_output=True,
        text=True
    )
    
    files = []
    current_path = None
    
    for line in result.stdout.split('\n'):
        if line.startswith('Path = '):
            path = line[7:]
            if path.endswith('.slp'):
                files.append(path)
    
    return files

def extract_files(archive_path, files, output_dir):
    """Extract specific files from the archive."""
    if not files:
        return
    
    # Write file list to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for filepath in files:
            f.write(filepath + '\n')
        listfile = f.name
    
    try:
        subprocess.run(
            ["7z", "x", "-y", f"-o{output_dir}", archive_path, f"@{listfile}"],
            capture_output=True,
            check=True
        )
    finally:
        os.unlink(listfile)

def process_batch(archive_path, files, temp_dir, output_base):
    """
    Process a batch of files:
    1. Extract to temp dir
    2. Check each for target characters
    3. Move matching files to character folders
    4. Delete non-matching files
    
    Returns dict of {character: count} for files kept.
    """
    if not files:
        return {}
    
    # Extract batch
    extract_files(archive_path, files, temp_dir)
    
    kept = {}
    
    for filepath in files:
        full_path = Path(temp_dir) / filepath
        if not full_path.exists():
            continue
        
        # Check for target characters
        found_chars = get_characters_from_replay(full_path)
        
        if found_chars:
            # Move to character-specific folder(s)
            for port, char_name in found_chars:
                char_dir = Path(output_base) / char_name
                char_dir.mkdir(parents=True, exist_ok=True)
                
                dest = char_dir / full_path.name
                # Handle duplicates
                if dest.exists():
                    base = dest.stem
                    suffix = dest.suffix
                    i = 1
                    while dest.exists():
                        dest = char_dir / f"{base}_{i}{suffix}"
                        i += 1
                
                shutil.copy2(full_path, dest)
                kept[char_name] = kept.get(char_name, 0) + 1
        
        # Delete from temp regardless
        full_path.unlink()
    
    return kept

def main():
    parser = argparse.ArgumentParser(description="Filter low-tier replays from 7z archive")
    parser.add_argument("archive", help="Path to 7z archive")
    parser.add_argument("output", help="Output directory for filtered replays")
    parser.add_argument("--batch-size", type=int, default=200, help="Files to process per batch")
    parser.add_argument("--start-at", type=int, default=0, help="Start at file index (for resuming)")
    parser.add_argument("--dry-run", action="store_true", help="List files without extracting")
    args = parser.parse_args()
    
    if not HAS_SLIPPI:
        print("Error: py-slippi is required. Install with: pip install py-slippi")
        sys.exit(1)
    
    archive = Path(args.archive)
    if not archive.exists():
        print(f"Error: Archive not found: {archive}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Archive: {archive}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Target characters: {', '.join(TARGET_CHARACTERS.values())}")
    print()
    
    # List all files in archive
    print("Listing archive contents...")
    all_files = list_archive_files(archive)
    total_files = len(all_files)
    print(f"Found {total_files} replay files")
    
    if args.dry_run:
        print("\nDry run - not extracting")
        return
    
    # Process in batches
    total_kept = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(args.start_at, total_files, args.batch_size):
            batch = all_files[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (total_files + args.batch_size - 1) // args.batch_size
            
            print(f"\rBatch {batch_num}/{total_batches} ({i}/{total_files} files)...", end="", flush=True)
            
            kept = process_batch(archive, batch, temp_dir, output_dir)
            
            for char, count in kept.items():
                total_kept[char] = total_kept.get(char, 0) + count
            
            # Progress update with counts
            if kept:
                kept_str = ", ".join(f"{c}: {n}" for c, n in kept.items())
                print(f" kept {sum(kept.values())} ({kept_str})")
            else:
                print(" kept 0")
    
    # Summary
    print("\n" + "=" * 50)
    print("COMPLETE!")
    print("=" * 50)
    print(f"Processed: {total_files} files")
    print(f"Kept: {sum(total_kept.values())} files")
    print("\nBy character:")
    for char, count in sorted(total_kept.items()):
        print(f"  {char}: {count}")
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
