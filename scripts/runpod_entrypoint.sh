#!/bin/bash
# RunPod entrypoint script - auto-configures services from environment variables

set -e

echo "=== ExPhil RunPod Entrypoint ==="

# Clean up files that cause git pull conflicts
# These files are in git but may have stale copies from Docker build
# Removing them allows git pull to succeed without conflicts
if [ -d "/app/.git" ]; then
  echo "Cleaning up potential git conflicts..."
  # List of files that frequently cause conflicts between Docker build and git
  CONFLICT_FILES=(
    "scripts/analyze_replays.exs"
  )
  for f in "${CONFLICT_FILES[@]}"; do
    if [ -f "/app/$f" ]; then
      # Check if file differs from git
      if ! git -C /app diff --quiet HEAD -- "$f" 2>/dev/null; then
        echo "  Removing stale $f (will be restored by git pull)"
        rm -f "/app/$f"
      fi
    fi
  done
fi

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
mkdir -p /workspace/checkpoints /workspace/logs /workspace/replays /workspace/cache

# Fix any circular symlinks from previous buggy runs
# (Bug: ln -sf on existing symlink-to-dir creates symlink inside the dir)
if [ -L "/workspace/checkpoints/checkpoints" ]; then
  echo "⚠ Removing circular symlink /workspace/checkpoints/checkpoints"
  rm /workspace/checkpoints/checkpoints
fi
if [ -L "/workspace/logs/logs" ]; then
  echo "⚠ Removing circular symlink /workspace/logs/logs"
  rm /workspace/logs/logs
fi

# Link checkpoint/logs dirs to workspace for persistence
# Must remove existing directories first (ln -sf can't replace dirs)
# IMPORTANT: Check if symlink already exists to avoid circular symlinks on re-run
if [ -d "/app" ]; then
  # Checkpoints: move any existing files to workspace, then symlink
  if [ -L "/app/checkpoints" ]; then
    # Already a symlink - check if it points to the right place
    if [ "$(readlink /app/checkpoints)" = "/workspace/checkpoints" ]; then
      echo "✓ /app/checkpoints symlink already correct"
    else
      echo "Fixing /app/checkpoints symlink..."
      rm /app/checkpoints
      ln -s /workspace/checkpoints /app/checkpoints
    fi
  elif [ -d "/app/checkpoints" ]; then
    # Real directory exists - move contents to workspace, then symlink
    if [ "$(ls -A /app/checkpoints 2>/dev/null)" ]; then
      echo "Moving existing checkpoints to /workspace/checkpoints/"
      mv /app/checkpoints/* /workspace/checkpoints/ 2>/dev/null || true
    fi
    rm -rf /app/checkpoints
    ln -s /workspace/checkpoints /app/checkpoints
    echo "✓ Created /app/checkpoints -> /workspace/checkpoints"
  else
    # Doesn't exist - create symlink
    ln -s /workspace/checkpoints /app/checkpoints
    echo "✓ Created /app/checkpoints -> /workspace/checkpoints"
  fi

  # Logs: same process
  if [ -L "/app/logs" ]; then
    # Already a symlink - check if it points to the right place
    if [ "$(readlink /app/logs)" = "/workspace/logs" ]; then
      echo "✓ /app/logs symlink already correct"
    else
      echo "Fixing /app/logs symlink..."
      rm /app/logs
      ln -s /workspace/logs /app/logs
    fi
  elif [ -d "/app/logs" ]; then
    # Real directory exists - move contents to workspace, then symlink
    if [ "$(ls -A /app/logs 2>/dev/null)" ]; then
      echo "Moving existing logs to /workspace/logs/"
      mv /app/logs/* /workspace/logs/ 2>/dev/null || true
    fi
    rm -rf /app/logs
    ln -s /workspace/logs /app/logs
    echo "✓ Created /app/logs -> /workspace/logs"
  else
    # Doesn't exist - create symlink
    ln -s /workspace/logs /app/logs
    echo "✓ Created /app/logs -> /workspace/logs"
  fi
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
  echo "No checkpoints found in /app/checkpoints or /workspace/checkpoints (nothing to upload)"
  exit 0  # Not an error, just nothing to do
fi
echo "Using checkpoint directory: $CKPT_DIR"

if [ "$1" = "--flat" ]; then
  echo "Uploading checkpoints to b2:$B2_BUCKET/checkpoints/ (flat)..."
  rclone copy "$CKPT_DIR/" "b2:$B2_BUCKET/checkpoints/" --copy-links --progress
elif [ "$1" = "--date" ] && [ -n "$2" ]; then
  echo "Uploading checkpoints to b2:$B2_BUCKET/checkpoints/$2/..."
  rclone copy "$CKPT_DIR/" "b2:$B2_BUCKET/checkpoints/$2/" --copy-links --progress
elif [ "$1" = "--today" ] || [ -z "$1" ]; then
  # Default: upload to today's date folder
  echo "Uploading checkpoints to b2:$B2_BUCKET/checkpoints/$TODAY/..."
  rclone copy "$CKPT_DIR/" "b2:$B2_BUCKET/checkpoints/$TODAY/" --copy-links --progress
else
  echo "Usage: sync-checkpoints-up [--today | --date YYYY-MM-DD | --flat]"
  exit 1
fi
echo "✓ Checkpoints uploaded"
SCRIPT
chmod +x /usr/local/bin/sync-checkpoints-up

cat > /usr/local/bin/sync-logs-up << 'SCRIPT'
#!/bin/bash
# Upload local logs to B2 (organized by date)
# Usage: sync-logs-up [--today | --date YYYY-MM-DD | --flat]

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"
TODAY=$(date +%Y-%m-%d)

# Find logs directory
if [ -d "/app/logs" ] && [ "$(ls -A /app/logs 2>/dev/null)" ]; then
  LOGS_DIR="/app/logs"
elif [ -d "/workspace/logs" ] && [ "$(ls -A /workspace/logs 2>/dev/null)" ]; then
  LOGS_DIR="/workspace/logs"
else
  echo "No logs found in /app/logs or /workspace/logs (nothing to upload)"
  exit 0  # Not an error, just nothing to do
fi
echo "Using logs directory: $LOGS_DIR"

if [ "$1" = "--flat" ]; then
  echo "Uploading logs to b2:$B2_BUCKET/logs/ (flat)..."
  rclone copy "$LOGS_DIR/" "b2:$B2_BUCKET/logs/" --copy-links --progress
elif [ "$1" = "--date" ] && [ -n "$2" ]; then
  echo "Uploading logs to b2:$B2_BUCKET/logs/$2/..."
  rclone copy "$LOGS_DIR/" "b2:$B2_BUCKET/logs/$2/" --copy-links --progress
elif [ "$1" = "--today" ] || [ -z "$1" ]; then
  echo "Uploading logs to b2:$B2_BUCKET/logs/$TODAY/..."
  rclone copy "$LOGS_DIR/" "b2:$B2_BUCKET/logs/$TODAY/" --copy-links --progress
else
  echo "Usage: sync-logs-up [--today | --date YYYY-MM-DD | --flat]"
  exit 1
fi
echo "✓ Logs uploaded"
SCRIPT
chmod +x /usr/local/bin/sync-logs-up

cat > /usr/local/bin/sync-cache-up << 'SCRIPT'
#!/bin/bash
# Upload local cache to B2 (embeddings, kmeans centers, etc.)
# Usage: sync-cache-up [--today | --date YYYY-MM-DD | --flat]
#
# Cache includes:
#   - Precomputed embeddings (saves ~1hr on re-runs)
#   - K-means cluster centers
#   - Any other cached data

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"
TODAY=$(date +%Y-%m-%d)

# Find cache directory
if [ -d "/workspace/cache" ] && [ "$(ls -A /workspace/cache 2>/dev/null)" ]; then
  CACHE_DIR="/workspace/cache"
else
  echo "No cache found in /workspace/cache (nothing to upload)"
  exit 0  # Not an error, just nothing to do
fi
echo "Using cache directory: $CACHE_DIR"

# Show what will be uploaded
echo "Cache contents:"
du -sh "$CACHE_DIR"/* 2>/dev/null | head -10

if [ "$1" = "--flat" ]; then
  echo "Uploading cache to b2:$B2_BUCKET/cache/ (flat)..."
  rclone copy "$CACHE_DIR/" "b2:$B2_BUCKET/cache/" --copy-links --progress
elif [ "$1" = "--date" ] && [ -n "$2" ]; then
  echo "Uploading cache to b2:$B2_BUCKET/cache/$2/..."
  rclone copy "$CACHE_DIR/" "b2:$B2_BUCKET/cache/$2/" --copy-links --progress
elif [ "$1" = "--today" ] || [ -z "$1" ]; then
  echo "Uploading cache to b2:$B2_BUCKET/cache/$TODAY/..."
  rclone copy "$CACHE_DIR/" "b2:$B2_BUCKET/cache/$TODAY/" --copy-links --progress
else
  echo "Usage: sync-cache-up [--today | --date YYYY-MM-DD | --flat]"
  exit 1
fi
echo "✓ Cache uploaded"
SCRIPT
chmod +x /usr/local/bin/sync-cache-up

cat > /usr/local/bin/sync-cache-down << 'SCRIPT'
#!/bin/bash
# Download cache from B2
# Usage: sync-cache-down [YYYY-MM-DD | --all | --latest | --flat]
#
# YYYY-MM-DD: download from specific date folder
# --all: download everything (all dates merged)
# --latest: download from most recent date folder
# --flat: download from root cache/ folder (no date organization)

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"
mkdir -p /workspace/cache

if [ "$1" = "--all" ]; then
  echo "Downloading ALL cache from b2:$B2_BUCKET/cache/..."
  rclone copy "b2:$B2_BUCKET/cache/" /workspace/cache/ --progress
elif [ "$1" = "--flat" ]; then
  echo "Downloading cache from b2:$B2_BUCKET/cache/ (flat structure)..."
  rclone copy "b2:$B2_BUCKET/cache/" /workspace/cache/ --progress --max-depth 1
elif [ "$1" = "--latest" ]; then
  # Find most recent date folder
  LATEST=$(rclone lsd "b2:$B2_BUCKET/cache/" 2>/dev/null | awk '{print $NF}' | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' | sort -r | head -1)
  if [ -z "$LATEST" ]; then
    echo "No dated cache folders found on B2, trying flat structure..."
    rclone copy "b2:$B2_BUCKET/cache/" /workspace/cache/ --progress
  else
    echo "Downloading cache from b2:$B2_BUCKET/cache/$LATEST/ (most recent)..."
    rclone copy "b2:$B2_BUCKET/cache/$LATEST/" /workspace/cache/ --progress
  fi
elif [ -n "$1" ]; then
  # Download from specific date folder
  echo "Downloading cache from b2:$B2_BUCKET/cache/$1/..."
  rclone copy "b2:$B2_BUCKET/cache/$1/" /workspace/cache/ --progress
else
  echo "Usage: sync-cache-down [YYYY-MM-DD | --all | --latest | --flat]"
  echo ""
  echo "Available dates on B2:"
  rclone lsd "b2:$B2_BUCKET/cache/" 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
  exit 0
fi
echo "✓ Cache downloaded"
SCRIPT
chmod +x /usr/local/bin/sync-cache-down

cat > /usr/local/bin/sync-all-up << 'SCRIPT'
#!/bin/bash
# Upload checkpoints, logs, and cache to B2
# Usage: sync-all-up [--today | --date YYYY-MM-DD | --flat]

echo "=== Syncing checkpoints ==="
sync-checkpoints-up "$@"

echo ""
echo "=== Syncing logs ==="
sync-logs-up "$@"

echo ""
echo "=== Syncing cache ==="
sync-cache-up "$@"

echo ""
echo "✓ All synced"
SCRIPT
chmod +x /usr/local/bin/sync-all-up

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
echo "Sync commands available:"
echo "  sync-all-up                   # Upload checkpoints + logs + cache to B2"
echo ""
echo "  sync-checkpoints-up           # Upload checkpoints (today's date folder)"
echo "  sync-checkpoints-down --latest    # Download most recent checkpoints"
echo ""
echo "  sync-logs-up                  # Upload logs (today's date folder)"
echo ""
echo "  sync-cache-up                 # Upload cache (embeddings, kmeans centers)"
echo "  sync-cache-down --latest      # Download most recent cache"
echo ""
echo "  list-checkpoints              # List checkpoint dates on B2"
echo ""
echo "Options for all sync commands: --today, --date YYYY-MM-DD, --flat"
echo ""

# Run whatever command was passed (or default to bash)
exec "$@"
