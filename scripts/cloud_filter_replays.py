#!/usr/bin/env python3
"""
Cloud Replay Filter - Download and filter low-tier replays from Google Drive archives.

Designed for RunPod or similar cloud instances with ample disk space.

Usage:
    # Process already-downloaded 7z files
    python cloud_filter_replays.py --local /workspace/downloads --output /workspace/lowtier

    # Streaming mode - extract in batches to save disk space
    python cloud_filter_replays.py --local /workspace/downloads --output /workspace/lowtier --streaming

    # Single archive from URL
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


def list_archive_files(archive_path):
    """List all .slp files in a 7z archive."""
    result = subprocess.run(
        ["7z", "l", "-ba", archive_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return []

    files = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        # 7z -ba output format: "attr size compressed date time name"
        # The filename is the last part after the datetime
        parts = line.split()
        if len(parts) >= 6:
            # Reconstruct filename (may contain spaces)
            filename = ' '.join(parts[5:])
            if filename.lower().endswith('.slp'):
                files.append(filename)

    return files


def extract_specific_files(archive_path, files, output_dir):
    """Extract specific files from a 7z archive."""
    if not files:
        return True

    # Create a temporary file list for 7z
    list_file = os.path.join(output_dir, "_filelist.txt")
    with open(list_file, 'w') as f:
        for filepath in files:
            f.write(f"{filepath}\n")

    # 7z uses -i@listfile syntax for include list
    result = subprocess.run(
        ["7z", "x", "-y", f"-o{output_dir}", archive_path, f"-i@{list_file}"],
        capture_output=True,
        text=True
    )

    # Cleanup list file
    if os.path.exists(list_file):
        os.remove(list_file)

    if result.returncode != 0:
        print(f"7z error: {result.stderr}")

    return result.returncode == 0


def process_archive_streaming(archive_path, output_dir, batch_size=200, cleanup=False):
    """Process archive in streaming mode - extract and filter in batches."""
    print(f"\n{'='*60}")
    print(f"Processing (streaming): {os.path.basename(archive_path)}")
    print(f"{'='*60}")

    archive_size = os.path.getsize(archive_path) / (1024**3)
    print(f"Archive size: {archive_size:.1f} GB")
    print(f"Batch size: {batch_size} files")

    # List all .slp files in archive
    print("Listing archive contents...")
    all_files = list_archive_files(archive_path)
    total_files = len(all_files)
    print(f"Found {total_files} .slp files in archive")

    if total_files == 0:
        return {}

    # Process in batches
    counts = {}
    processed = 0
    kept = 0
    start = time.time()

    extract_dir = os.path.join(os.path.dirname(archive_path), "extracted_batch")

    num_batches = (total_files + batch_size - 1) // batch_size
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = all_files[batch_start:batch_end]

        print(f"  Batch {batch_num + 1}/{num_batches}: Extracting {len(batch_files)} files...", flush=True)

        # Clean extract dir
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)

        # Extract batch
        extract_start = time.time()
        if not extract_specific_files(archive_path, batch_files, extract_dir):
            print(f"  Warning: Failed to extract batch {batch_num + 1}")
            continue
        extract_time = time.time() - extract_start
        print(f"  Batch {batch_num + 1}/{num_batches}: Extracted in {extract_time:.1f}s, filtering...", flush=True)

        # Process extracted files
        batch_kept = 0
        for slp_file in Path(extract_dir).rglob("*.slp"):
            result = process_replay(slp_file, output_dir)
            if result:
                counts[result] = counts.get(result, 0) + 1
                kept += 1
                batch_kept += 1
            processed += 1

        # Progress
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total_files - processed) / rate if rate > 0 else 0
        print(f"  Batch {batch_num + 1}/{num_batches}: Done | {processed}/{total_files} ({100*processed/total_files:.1f}%) | "
              f"batch kept: {batch_kept} | total kept: {kept} | {rate:.0f} files/s | ETA: {eta:.0f}s", flush=True)

    # Final cleanup
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    if cleanup:
        print(f"Removing archive...")
        os.remove(archive_path)

    return counts

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

def process_local_archive(archive_path, output_dir, cleanup=False):
    """Extract, filter, and optionally cleanup a local archive."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(archive_path)}")
    print(f"{'='*60}")

    archive_size = os.path.getsize(archive_path) / (1024**3)
    print(f"Archive size: {archive_size:.1f} GB")

    # Extract to temp location next to archive
    extract_dir = os.path.join(os.path.dirname(archive_path), "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    if not extract_archive(archive_path, extract_dir):
        print(f"ERROR: Failed to extract {archive_path}")
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
    # Process local 7z files (already downloaded)
    python cloud_filter_replays.py --local /workspace/downloads --output ./lowtier

    # Streaming mode - extract in batches (uses less disk space)
    python cloud_filter_replays.py --local /workspace/downloads --output ./lowtier --streaming

    # Streaming with custom batch size
    python cloud_filter_replays.py --local /workspace/downloads --output ./lowtier --streaming --batch-size 500

    # Single URL
    python cloud_filter_replays.py --url "https://drive.google.com/file/d/ABC123/view" --output ./lowtier

    # Multiple URLs from file (one per line)
    python cloud_filter_replays.py --urls-file links.txt --output ./lowtier --cleanup

Target characters: mewtwo, ganondorf, link, zelda, game_and_watch
        """
    )
    parser.add_argument("--local", help="Directory containing already-downloaded 7z files")
    parser.add_argument("--url", help="Single Google Drive URL")
    parser.add_argument("--urls-file", help="File containing URLs (one per line)")
    parser.add_argument("--output", required=True, help="Output directory for filtered replays")
    parser.add_argument("--download-dir", default="/tmp/replay_downloads", help="Temp dir for downloads")
    parser.add_argument("--cleanup", action="store_true", help="Delete archives after processing")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for filtering")
    parser.add_argument("--streaming", action="store_true", help="Extract in batches to save disk space (slower but uses less space)")
    parser.add_argument("--batch-size", type=int, default=200, help="Files per batch in streaming mode (default: 200)")
    args = parser.parse_args()

    if not args.url and not args.urls_file and not args.local:
        parser.error("Must provide --local, --url, or --urls-file")

    # Install dependencies
    install_dependencies()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    total_counts = {}
    archives_processed = 0

    # Handle local archives
    if args.local:
        local_dir = Path(args.local)
        archives = list(local_dir.glob("*.7z"))
        if not archives:
            print(f"No .7z files found in {args.local}")
            return

        print(f"Local archives to process: {len(archives)}")
        print(f"Output directory: {args.output}")
        print(f"Cleanup after processing: {args.cleanup}")
        print(f"Streaming mode: {args.streaming}")
        if args.streaming:
            print(f"Batch size: {args.batch_size}")
        print(f"Target characters: {', '.join(set(TARGET_CHARACTERS.values()))}")

        # Check disk space
        total, used, free = shutil.disk_usage(args.local)
        print(f"\nDisk space: {free/1024**3:.1f} GB free of {total/1024**3:.1f} GB")

        for i, archive in enumerate(sorted(archives), 1):
            print(f"\n[{i}/{len(archives)}] Processing archive...")
            if args.streaming:
                counts = process_archive_streaming(str(archive), args.output, args.batch_size, args.cleanup)
            else:
                counts = process_local_archive(str(archive), args.output, args.cleanup)

            archives_processed += 1
            for char, count in counts.items():
                total_counts[char] = total_counts.get(char, 0) + count

            if total_counts:
                print(f"\nRunning totals:")
                for char, count in sorted(total_counts.items()):
                    print(f"  {char}: {count}")
    else:
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
        os.makedirs(args.download_dir, exist_ok=True)

        # Check disk space
        total, used, free = shutil.disk_usage(args.download_dir)
        print(f"\nDisk space: {free/1024**3:.1f} GB free of {total/1024**3:.1f} GB")

        if free < 100 * 1024**3:  # Less than 100GB
            print("WARNING: Less than 100GB free. Consider using --cleanup flag.")

        # Process each archive
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Processing archive...")
            counts = process_archive(url, args.download_dir, args.output, args.cleanup)

            archives_processed += 1
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
    print(f"Archives processed: {archives_processed}")
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
