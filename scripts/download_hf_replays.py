#!/usr/bin/env python3
"""Download .slp replay files from HuggingFace slippi-public-dataset-v3.7.

Usage:
    python3 scripts/download_hf_replays.py              # Download 20 files
    python3 scripts/download_hf_replays.py --count 100  # Download 100 files
    python3 scripts/download_hf_replays.py --all         # Download all ~10k files
"""
import argparse
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "erickfm/slippi-public-dataset-v3.7"
LOCAL_DIR = "replays/huggingface"

parser = argparse.ArgumentParser(description="Download Slippi replays from HuggingFace")
parser.add_argument("--count", type=int, default=20, help="Number of files to download (default: 20)")
parser.add_argument("--all", action="store_true", help="Download all files")
parser.add_argument("--dir", default=LOCAL_DIR, help=f"Local directory (default: {LOCAL_DIR})")
args = parser.parse_args()

print(f"Listing files in {REPO}...")
files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".slp")]
print(f"Found {len(files)} .slp files")

count = len(files) if args.all else min(args.count, len(files))
print(f"Downloading {count} files to {args.dir}/\n")

for i, f in enumerate(files[:count]):
    path = hf_hub_download(REPO, f, repo_type="dataset", local_dir=args.dir)
    print(f"  [{i+1}/{count}] {f}")

print(f"\nDone! {count} files in {args.dir}/")
