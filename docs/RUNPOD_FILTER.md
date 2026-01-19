# Filtering Replays on RunPod

Quick guide to extract low-tier character replays from large 7z archives using RunPod.

## Why RunPod?

- **Disk space**: Get 100GB+ container disk
- **Bandwidth**: Fast downloads from Google Drive
- **CPU**: Multi-core parallel processing
- **No GPU needed**: This is I/O bound, use cheapest instance

## Setup

### 1. Create RunPod Instance

- **Template**: Ubuntu 22.04 (or any Linux)
- **GPU**: None needed (CPU only) - use cheapest option
- **Container Disk**: 200GB+ (need space for archives + extraction)
- **Volume**: 50GB at `/workspace` (to persist results)

### 2. Connect and Setup

```bash
# SSH into your pod
ssh root@<pod-ip> -p <port>

# Install dependencies
apt-get update && apt-get install -y p7zip-full python3-pip
pip3 install gdown py-slippi

# Get the filter script
curl -O https://raw.githubusercontent.com/blasphemetheus/exphil/main/scripts/cloud_filter_replays.py
chmod +x cloud_filter_replays.py
```

### 3. Create Links File

Create a file with your Google Drive links (one per line):

```bash
cat > /workspace/links.txt << 'EOF'
https://drive.google.com/file/d/XXXXXXX/view
https://drive.google.com/file/d/YYYYYYY/view
https://drive.google.com/file/d/ZZZZZZZ/view
EOF
```

### 4. Run the Filter

```bash
# Process all archives, cleanup after each to save space
python3 cloud_filter_replays.py \
  --urls-file /workspace/links.txt \
  --output /workspace/lowtier \
  --cleanup

# Or run in tmux so it survives disconnects
tmux new -s filter
python3 cloud_filter_replays.py --urls-file /workspace/links.txt --output /workspace/lowtier --cleanup
# Ctrl+B, D to detach
# tmux attach -t filter to reattach
```

### 5. Download Results

From your local machine:

```bash
rsync -avz --progress -e "ssh -p PORT" \
  root@IP:/workspace/lowtier/ \
  ~/git/melee/replays/lowtier/
```

## Expected Output Structure

```
/workspace/lowtier/
├── mewtwo/
│   ├── game1.slp
│   ├── game2.slp
│   └── ...
├── ganondorf/
│   └── ...
├── link/
│   └── ...
├── zelda/
│   └── ...
└── game_and_watch/
    └── ...
```

## Cost Estimate

| Archive Size | Extract Time | Filter Time | Total |
|--------------|--------------|-------------|-------|
| 70GB (100K files) | ~10 min | ~15 min | ~25 min |
| 100GB (150K files) | ~15 min | ~20 min | ~35 min |

With a $0.10/hr CPU instance, processing 5 archives costs ~$0.30.

## Troubleshooting

### "Quota exceeded" on Google Drive
- Try again later, or use `gdown --fuzzy` flag
- Split downloads across time

### Out of disk space
- Use `--cleanup` flag to delete archives after extraction
- Check space: `df -h`

### Slow downloads
- Google Drive can throttle; downloads vary from 10-100 MB/s
- Large files (100GB+) may take 20-60 min to download

## Target Characters

The script filters for these characters:
- Mewtwo
- Ganondorf  
- Link
- Zelda
- Mr. Game & Watch
