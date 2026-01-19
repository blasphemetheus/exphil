#!/usr/bin/env python3
"""
Cloud Replay Filter - Download and filter low-tier replays from Google Drive archives.

Designed for RunPod or similar cloud instances with ample disk space.

Usage:
    # Single archive
    python cloud_filter_replays.py --url "https://drive.google.com/..." --output /workspace/lowtier
    
    # Multiple archives from file
    python cloud_filter_replays.py --urls-file links.txt --output /workspace/lowtier
    
    # With cleanup (delete archives after processing)
    python cloud_filter_replays.py --urls-file links.txt --output /workspace/lowtier --cleanup

Setup on RunPod:
    pip install gdown py-slippi
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Target characters
TARGET_CHARACTERS = {
    # Internal IDs
    10: "mewtwo",
    22: "ganondorf", 
    5: "link",
    18: "zelda",
    24: "game_and_watch",
    # External IDs (some parsers use these)
    16: "mewtwo",
    25: "ganondorf",
    6: "link",
    19: "zelda",
}

def install_dependencies():
    """Install required packages if missing."""
    packages = ["gdown", "py-slippi"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

def download_from_gdrive(url, output_dir):
    """Download file from Google Drive using gdown."""
    import gdown
    
    print(f"Downloading: {url}")
    
    # Handle different URL formats
    if "drive.google.com" in url:
        if "/file/d/" in url:
            # Format: https://drive.google.com/file/d/FILE_ID/view
            file_id = url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            # Format: https://drive.google.com/uc?id=FILE_ID
            file_id = url.split("id=")[1].split("&")[0]
        else:
            file_id = url
    else:
        file_id = url
    
    output_path = os.path.join(output_dir, f"{file_id}.7z")
    
    try:
        gdown.download(id=file_id, output=output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as e:
        print(f"gdown failed: {e}")
    
    # Fallback: try direct URL
    try:
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(direct_url, output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    return None

def get_characters_from_replay(filepath):
    """Extract character IDs from a replay file."""
    try:
        from slippi import Game
        game = Game(str(filepath))
        found = []
        
        for port, player in enumerate(game.metadata.players or []):
            if player is None:
                continue
            
            char_id = None
            if hasattr(player, 'characters') and player.characters:
                char_id = max(player.characters.keys(), key=lambda k: player.characters[k])
            
            if char_id is not None and char_id in TARGET_CHARACTERS:
                found.append((port, TARGET_CHARACTERS[char_id]))
        
        return found
    except Exception:
        return []

def extract_archive(archive_path, output_dir):
    """Extract entire 7z archive."""
    print(f"Extracting: {archive_path}")
    start = time.time()
    
    result = subprocess.run(
        ["7z", "x", "-y", f"-o{output_dir}", archive_path],
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start
    print(f"Extraction complete in {elapsed:.1f}s")
    
    return result.returncode == 0

def process_replay(filepath, output_base):
    """Check single replay and move if it has target characters."""
    found = get_characters_from_replay(filepath)
    
    if found:
        for port, char_name in found:
            char_dir = Path(output_base) / char_name
            char_dir.mkdir(parents=True, exist_ok=True)
            
            dest = char_dir / filepath.name
            counter = 1
            while dest.exists():
                dest = char_dir / f"{filepath.stem}_{counter}{filepath.suffix}"
                counter += 1
            
            shutil.copy2(filepath, dest)
        
        return found[0][1]  # Return first character found
    
    return None

def filter_replays(extract_dir, output_dir, num_workers=8):
    """Filter extracted replays for target characters."""
    print(f"Scanning for .slp files in {extract_dir}...")
    
    slp_files = list(Path(extract_dir).rglob("*.slp"))
    total = len(slp_files)
    print(f"Found {total} replay files")
    
    if total == 0:
        return {}
    
    counts = {}
    processed = 0
    start = time.time()
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_replay, f, output_dir): f for f in slp_files}
        
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            
            if result:
                counts[result] = counts.get(result, 0) + 1
            
            # Progress every 1000 files
            if processed % 1000 == 0 or processed == total:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                kept = sum(counts.values())
                print(f"  {processed}/{total} ({100*processed/total:.1f}%) | "
                      f"kept: {kept} | {rate:.0f} files/s | ETA: {eta:.0f}s")
    
    return counts

def process_archive(url, download_dir, output_dir, cleanup=False):
    """Download, extract, filter, and optionally cleanup a single archive."""
    print(f"\n{'='*60}")
    print(f"Processing: {url[:80]}...")
    print(f"{'='*60}")
    
    # Download
    archive_path = download_from_gdrive(url, download_dir)
    if not archive_path:
        print(f"ERROR: Failed to download {url}")
        return {}
    
    archive_size = os.path.getsize(archive_path) / (1024**3)
    print(f"Downloaded: {archive_path} ({archive_size:.1f} GB)")
    
    # Extract to temp location
    extract_dir = os.path.join(download_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    if not extract_archive(archive_path, extract_dir):
        print(f"ERROR: Failed to extract {archive_path}")
        if cleanup:
            os.remove(archive_path)
        return {}
    
    # Delete archive to free space (we have the extracted files)
    if cleanup:
        print(f"Removing archive to free space...")
        os.remove(archive_path)
    
    # Filter replays
    counts = filter_replays(extract_dir, output_dir)
    
    # Cleanup extracted files
    print(f"Cleaning up extracted files...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    
    return counts

def main():
    parser = argparse.ArgumentParser(
        description="Download and filter low-tier replays from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single URL
    python cloud_filter_replays.py --url "https://drive.google.com/file/d/ABC123/view" --output ./lowtier
    
    # Multiple URLs from file (one per line)
    python cloud_filter_replays.py --urls-file links.txt --output ./lowtier --cleanup
    
    # On RunPod with workspace volume
    python cloud_filter_replays.py --urls-file /workspace/links.txt --output /workspace/lowtier --cleanup

Target characters: mewtwo, ganondorf, link, zelda, game_and_watch
        """
    )
    parser.add_argument("--url", help="Single Google Drive URL")
    parser.add_argument("--urls-file", help="File containing URLs (one per line)")
    parser.add_argument("--output", required=True, help="Output directory for filtered replays")
    parser.add_argument("--download-dir", default="/tmp/replay_downloads", help="Temp dir for downloads")
    parser.add_argument("--cleanup", action="store_true", help="Delete archives after processing")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for filtering")
    args = parser.parse_args()
    
    if not args.url and not args.urls_file:
        parser.error("Must provide --url or --urls-file")
    
    # Install dependencies
    install_dependencies()
    
    # Collect URLs
    urls = []
    if args.url:
        urls.append(args.url)
    if args.urls_file:
        with open(args.urls_file) as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    
    print(f"URLs to process: {len(urls)}")
    print(f"Output directory: {args.output}")
    print(f"Download directory: {args.download_dir}")
    print(f"Cleanup after processing: {args.cleanup}")
    print(f"Target characters: {', '.join(set(TARGET_CHARACTERS.values()))}")
    
    # Create directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)
    
    # Check disk space
    total, used, free = shutil.disk_usage(args.download_dir)
    print(f"\nDisk space: {free/1024**3:.1f} GB free of {total/1024**3:.1f} GB")
    
    if free < 100 * 1024**3:  # Less than 100GB
        print("WARNING: Less than 100GB free. Consider using --cleanup flag.")
    
    # Process each archive
    total_counts = {}
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing archive...")
        counts = process_archive(url, args.download_dir, args.output, args.cleanup)
        
        for char, count in counts.items():
            total_counts[char] = total_counts.get(char, 0) + count
        
        # Show running totals
        if total_counts:
            print(f"\nRunning totals:")
            for char, count in sorted(total_counts.items()):
                print(f"  {char}: {count}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Archives processed: {len(urls)}")
    print(f"Total replays kept: {sum(total_counts.values())}")
    print(f"\nBy character:")
    for char, count in sorted(total_counts.items()):
        print(f"  {char}: {count}")
    print(f"\nOutput: {args.output}")
    
    # List output structure
    print(f"\nOutput structure:")
    for char_dir in sorted(Path(args.output).iterdir()):
        if char_dir.is_dir():
            count = len(list(char_dir.glob("*.slp")))
            size = sum(f.stat().st_size for f in char_dir.glob("*.slp")) / 1024**2
            print(f"  {char_dir.name}/: {count} files ({size:.1f} MB)")

if __name__ == "__main__":
    main()
