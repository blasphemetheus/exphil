#!/bin/bash
# RunPod entrypoint script - auto-configures services from environment variables

set -e

echo "=== ExPhil RunPod Entrypoint ==="

# Generate unique pod identifier for checkpoint organization
# Format: hostname_YYYYMMDD (e.g., fd9c9e74e6d5_20260124)
POD_ID="${HOSTNAME:-$(hostname)}_$(date +%Y%m%d)"
export POD_ID
echo "Pod ID: $POD_ID"

# Auto-configure rclone from env vars
if [ -n "$B2_KEY_ID" ] && [ -n "$B2_APP_KEY" ]; then
  mkdir -p ~/.config/rclone
  cat > ~/.config/rclone/rclone.conf << EOF
[b2]
type = b2
account = $B2_KEY_ID
key = $B2_APP_KEY
EOF
  echo "✓ Rclone configured for B2"

  # Set default bucket if not specified
  B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"
  export B2_BUCKET

  # Optionally auto-sync replays if SYNC_REPLAYS is set
  if [ "$SYNC_REPLAYS" = "true" ]; then
    echo "Syncing replays from b2:$B2_BUCKET..."
    mkdir -p /workspace/replays
    rclone copy "b2:$B2_BUCKET/mewtwo" /workspace/replays/mewtwo/ --progress 2>/dev/null || \
      rclone copy "b2:$B2_BUCKET" /workspace/replays/ --progress
    echo "✓ Replays synced"
  fi

  # List existing checkpoints on B2 (informational)
  echo ""
  echo "Existing checkpoints on B2:"
  rclone lsd "b2:$B2_BUCKET/checkpoints/" 2>/dev/null || echo "  (none or checkpoints/ doesn't exist yet)"
  echo ""

  # Optionally pull checkpoints from B2 on startup
  if [ "$SYNC_CHECKPOINTS_DOWN" = "true" ]; then
    echo "Downloading checkpoints from B2..."
    mkdir -p /workspace/checkpoints
    rclone copy "b2:$B2_BUCKET/checkpoints/" /workspace/checkpoints/ --progress
    echo "✓ Checkpoints downloaded"
  fi
else
  echo "⚠ B2_KEY_ID or B2_APP_KEY not set - rclone not configured"
  echo "  Set these environment variables or run 'rclone config' manually"
fi

# Create workspace directories if they don't exist
mkdir -p /workspace/checkpoints /workspace/logs /workspace/replays

# Link checkpoint/logs dirs to workspace for persistence
# Must remove existing directories first (ln -sf can't replace dirs)
if [ -d "/app" ]; then
  # Checkpoints: move any existing files to workspace, then symlink
  if [ -d "/app/checkpoints" ] && [ ! -L "/app/checkpoints" ]; then
    # Real directory exists - move contents to workspace
    if [ "$(ls -A /app/checkpoints 2>/dev/null)" ]; then
      echo "Moving existing checkpoints to /workspace/checkpoints/"
      mv /app/checkpoints/* /workspace/checkpoints/ 2>/dev/null || true
    fi
    rmdir /app/checkpoints 2>/dev/null || rm -rf /app/checkpoints
  fi
  ln -sf /workspace/checkpoints /app/checkpoints 2>/dev/null || true

  # Logs: same process
  if [ -d "/app/logs" ] && [ ! -L "/app/logs" ]; then
    if [ "$(ls -A /app/logs 2>/dev/null)" ]; then
      echo "Moving existing logs to /workspace/logs/"
      mv /app/logs/* /workspace/logs/ 2>/dev/null || true
    fi
    rmdir /app/logs 2>/dev/null || rm -rf /app/logs
  fi
  ln -sf /workspace/logs /app/logs 2>/dev/null || true

  echo "✓ /app/checkpoints -> /workspace/checkpoints (persistent)"
  echo "✓ /app/logs -> /workspace/logs (persistent)"
fi

# Create helper scripts for checkpoint management
cat > /usr/local/bin/sync-checkpoints-up << 'SCRIPT'
#!/bin/bash
# Upload local checkpoints to B2 (organized by date)
# Usage: sync-checkpoints-up [--today | --date YYYY-MM-DD | --flat]
#
# Default: uploads to checkpoints/YYYY-MM-DD/ based on each file's date
# --today: upload all to today's folder
# --date: upload all to specific date folder
# --flat: upload to root checkpoints/ folder (no date organization)

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"
TODAY=$(date +%Y-%m-%d)

# Find checkpoint directory (prefer /app/checkpoints, fallback to /workspace)
if [ -d "/app/checkpoints" ] && [ "$(ls -A /app/checkpoints 2>/dev/null)" ]; then
  CKPT_DIR="/app/checkpoints"
elif [ -d "/workspace/checkpoints" ] && [ "$(ls -A /workspace/checkpoints 2>/dev/null)" ]; then
  CKPT_DIR="/workspace/checkpoints"
else
  echo "No checkpoints found in /app/checkpoints or /workspace/checkpoints"
  exit 1
fi
echo "Using checkpoint directory: $CKPT_DIR"

if [ "$1" = "--flat" ]; then
  echo "Uploading checkpoints to b2:$B2_BUCKET/checkpoints/ (flat)..."
  rclone copy "$CKPT_DIR/" "b2:$B2_BUCKET/checkpoints/" --progress
elif [ "$1" = "--date" ] && [ -n "$2" ]; then
  echo "Uploading checkpoints to b2:$B2_BUCKET/checkpoints/$2/..."
  rclone copy "$CKPT_DIR/" "b2:$B2_BUCKET/checkpoints/$2/" --progress
elif [ "$1" = "--today" ] || [ -z "$1" ]; then
  # Default: upload to today's date folder
  echo "Uploading checkpoints to b2:$B2_BUCKET/checkpoints/$TODAY/..."
  rclone copy "$CKPT_DIR/" "b2:$B2_BUCKET/checkpoints/$TODAY/" --progress
else
  echo "Usage: sync-checkpoints-up [--today | --date YYYY-MM-DD | --flat]"
  exit 1
fi
echo "✓ Checkpoints uploaded"
SCRIPT
chmod +x /usr/local/bin/sync-checkpoints-up

cat > /usr/local/bin/sync-checkpoints-down << 'SCRIPT'
#!/bin/bash
# Download checkpoints from B2
# Usage: sync-checkpoints-down [YYYY-MM-DD | --all | --latest]
#
# YYYY-MM-DD: download from specific date folder
# --all: download everything (all dates)
# --latest: download from most recent date folder

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"

if [ "$1" = "--all" ]; then
  echo "Downloading ALL checkpoints from b2:$B2_BUCKET/checkpoints/..."
  rclone copy "b2:$B2_BUCKET/checkpoints/" /workspace/checkpoints/ --progress
elif [ "$1" = "--latest" ]; then
  # Find most recent date folder
  LATEST=$(rclone lsd "b2:$B2_BUCKET/checkpoints/" 2>/dev/null | awk '{print $NF}' | sort -r | head -1)
  if [ -z "$LATEST" ]; then
    echo "No checkpoint folders found on B2"
    exit 1
  fi
  echo "Downloading checkpoints from b2:$B2_BUCKET/checkpoints/$LATEST/ (most recent)..."
  rclone copy "b2:$B2_BUCKET/checkpoints/$LATEST/" /workspace/checkpoints/ --progress
elif [ -n "$1" ]; then
  # Download from specific date folder
  echo "Downloading checkpoints from b2:$B2_BUCKET/checkpoints/$1/..."
  rclone copy "b2:$B2_BUCKET/checkpoints/$1/" /workspace/checkpoints/ --progress
else
  echo "Usage: sync-checkpoints-down [YYYY-MM-DD | --all | --latest]"
  echo ""
  echo "Available dates on B2:"
  rclone lsd "b2:$B2_BUCKET/checkpoints/" 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
  exit 0
fi
echo "✓ Checkpoints downloaded"
SCRIPT
chmod +x /usr/local/bin/sync-checkpoints-down

cat > /usr/local/bin/list-checkpoints << 'SCRIPT'
#!/bin/bash
# List checkpoints on B2 or locally
# Usage: list-checkpoints [--local | YYYY-MM-DD]

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"

if [ "$1" = "--local" ]; then
  echo "Local checkpoints:"
  echo ""
  if [ -L "/app/checkpoints" ]; then
    echo "/app/checkpoints -> $(readlink /app/checkpoints) (symlink)"
  elif [ -d "/app/checkpoints" ]; then
    echo "/app/checkpoints/ (real directory):"
    ls -lah /app/checkpoints/*.axon 2>/dev/null || echo "  (no .axon files)"
    ls -lah /app/checkpoints/*.bin 2>/dev/null || echo "  (no .bin files)"
  fi
  echo ""
  echo "/workspace/checkpoints/:"
  ls -lah /workspace/checkpoints/*.axon 2>/dev/null || echo "  (no .axon files)"
  ls -lah /workspace/checkpoints/*.bin 2>/dev/null || echo "  (no .bin files)"
elif [ -n "$1" ]; then
  # List specific date folder
  echo "Checkpoints on B2 for $1:"
  rclone ls "b2:$B2_BUCKET/checkpoints/$1/" 2>/dev/null || echo "  (none or folder doesn't exist)"
else
  echo "Checkpoint dates on B2 (b2:$B2_BUCKET/checkpoints/):"
  echo ""
  rclone lsd "b2:$B2_BUCKET/checkpoints/" 2>/dev/null | while read line; do
    folder=$(echo "$line" | awk '{print $NF}')
    count=$(rclone ls "b2:$B2_BUCKET/checkpoints/$folder/" 2>/dev/null | wc -l)
    echo "  $folder/ ($count files)"
  done || echo "  (none)"
  echo ""
  echo "Use 'list-checkpoints YYYY-MM-DD' to see files in a specific date"
  echo "Use 'list-checkpoints --local' to see local checkpoints"
fi
SCRIPT
chmod +x /usr/local/bin/list-checkpoints

echo "=== Entrypoint complete ==="
echo ""
echo "Checkpoint sync commands available:"
echo "  sync-checkpoints-up           # Upload to B2 (today's date folder)"
echo "  sync-checkpoints-up --date YYYY-MM-DD  # Upload to specific date"
echo "  sync-checkpoints-up --flat    # Upload to root (no date folders)"
echo ""
echo "  sync-checkpoints-down         # Show available dates"
echo "  sync-checkpoints-down YYYY-MM-DD  # Download from specific date"
echo "  sync-checkpoints-down --latest    # Download most recent date"
echo "  sync-checkpoints-down --all       # Download everything"
echo ""
echo "  list-checkpoints              # List dates on B2"
echo "  list-checkpoints YYYY-MM-DD   # List files in date folder"
echo "  list-checkpoints --local      # List local checkpoints"
echo ""

# Run whatever command was passed (or default to bash)
exec "$@"
