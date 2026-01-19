# Filtering Replays on RunPod

Quick guide to extract low-tier character replays from large 7z archives using RunPod.

## Why RunPod?

- **Disk space**: Network volumes with hundreds of GB
- **Bandwidth**: Fast downloads from Google Drive
- **CPU**: Multi-core parallel processing
- **No GPU needed**: This is I/O bound, use cheapest instance

## Setup

### 1. Create RunPod Instance

- **Template**: Ubuntu 22.04 (or any Linux)
- **GPU**: None needed (CPU only) - use cheapest option
- **Container Disk**: 5GB is fine (tools only, data goes to volume)
- **Volume**: 200GB+ at `/workspace` (persists across restarts)

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

### "Quota exceeded" / "Too many users" on Google Drive

gdown often fails on popular files. **Use rclone instead** - see [RCLONE_GDRIVE.md](RCLONE_GDRIVE.md) for full setup.

Quick version:
```bash
# Install rclone
curl https://rclone.org/install.sh | bash

# Configure with your own Google API credentials (bypasses rate limits)
rclone config

# Open each file link in your browser first (adds to "Shared with me")
# Then download:
rclone copy --progress --drive-shared-with-me "gdrive:" /workspace/downloads/ --include "*.7z"
```

### Out of disk space
- Use `--cleanup` flag to delete archives after extraction
- Check space: `df -h`

### Slow downloads
- Google Drive can throttle; downloads vary from 10-100 MB/s
- Large files (100GB+) may take 20-60 min to download

## Resuming After Pod Restart

Container filesystems are ephemeral - you lose installed tools but `/workspace` persists.

### Quick Resume Checklist

```bash
# 1. Install tools
apt-get update && apt-get install -y p7zip-full python3-pip
pip3 install --upgrade pip setuptools wheel
pip3 install gdown py-slippi
curl https://rclone.org/install.sh | bash

# 2. Setup directories and get filter script
mkdir -p /workspace/tmp /workspace/downloads /workspace/lowtier
curl -o /workspace/cloud_filter_replays.py https://raw.githubusercontent.com/blasphemetheus/exphil/main/scripts/cloud_filter_replays.py

# 3. Reconfigure rclone (need your Google API credentials)
rclone config
# n → gdrive → drive → [client_id] → [client_secret] → 1 → blank → blank → n → n
# Run authorize command on local machine, paste token back
# n → y → q

# 4. Check what's already downloaded
ls -lh /workspace/downloads/

# 5. Resume downloads (skips completed files)
TMPDIR=/workspace/tmp rclone copy --progress --drive-shared-with-me "gdrive:" /workspace/downloads/ --include "*.7z"

# 6. After downloads complete, run filter
python3 /workspace/cloud_filter_replays.py \
  --download-dir /workspace/downloads \
  --output /workspace/lowtier \
  --cleanup
```

### Your Google API Credentials

Store these somewhere safe - you'll need them after each restart:
- **Client ID**: (from Google Cloud Console)
- **Client Secret**: (from Google Cloud Console)

See [RCLONE_GDRIVE.md](RCLONE_GDRIVE.md) for full setup instructions.

## Target Characters

The script filters for these characters:
- Mewtwo
- Ganondorf
- Link
- Zelda
- Mr. Game & Watch
