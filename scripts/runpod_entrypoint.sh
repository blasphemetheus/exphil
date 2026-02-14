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

  # Set default buckets if not specified
  # B2_REPLAYS = source replay files (read mostly)
  # B2_ARTIFACTS = checkpoints, logs, cache (write frequently)
  B2_REPLAYS="${B2_REPLAYS:-your-replays-bucket}"
  B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"
  export B2_REPLAYS B2_ARTIFACTS

  echo "  Replays bucket:   b2:$B2_REPLAYS"
  echo "  Artifacts bucket: b2:$B2_ARTIFACTS"

  # Optionally auto-sync replays if SYNC_REPLAYS is set
  if [ "$SYNC_REPLAYS" = "true" ]; then
    echo "Syncing replays from b2:$B2_REPLAYS..."
    mkdir -p /workspace/replays
    rclone copy "b2:$B2_REPLAYS" /workspace/replays/ --progress
    echo "✓ Replays synced"
  fi

  # List existing checkpoints on B2 (informational)
  echo ""
  echo "Existing checkpoints on B2:"
  rclone ls "b2:$B2_ARTIFACTS/checkpoints/" 2>/dev/null | head -5 || echo "  (none yet)"
  echo ""

  # Optionally pull checkpoints from B2 on startup
  if [ "$SYNC_CHECKPOINTS_DOWN" = "true" ]; then
    echo "Downloading checkpoints from B2..."
    mkdir -p /workspace/checkpoints
    rclone copy "b2:$B2_ARTIFACTS/checkpoints/" /workspace/checkpoints/ --progress
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
if [ -L "/workspace/cache/cache" ]; then
  echo "⚠ Removing circular symlink /workspace/cache/cache"
  rm /workspace/cache/cache
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

  # Cache: same process (embeddings, kmeans centers, etc.)
  if [ -L "/app/cache" ]; then
    # Already a symlink - check if it points to the right place
    if [ "$(readlink /app/cache)" = "/workspace/cache" ]; then
      echo "✓ /app/cache symlink already correct"
    else
      echo "Fixing /app/cache symlink..."
      rm /app/cache
      ln -s /workspace/cache /app/cache
    fi
  elif [ -d "/app/cache" ]; then
    # Real directory exists - move contents to workspace, then symlink
    if [ "$(ls -A /app/cache 2>/dev/null)" ]; then
      echo "Moving existing cache to /workspace/cache/"
      mv /app/cache/* /workspace/cache/ 2>/dev/null || true
    fi
    rm -rf /app/cache
    ln -s /workspace/cache /app/cache
    echo "✓ Created /app/cache -> /workspace/cache"
  else
    # Doesn't exist - create symlink
    ln -s /workspace/cache /app/cache
    echo "✓ Created /app/cache -> /workspace/cache"
  fi
fi

# Create helper scripts for checkpoint management
cat > /usr/local/bin/sync-checkpoints-up << 'SCRIPT'
#!/bin/bash
# Upload local checkpoints to B2 (flat structure)
# rclone skips identical files automatically.
# Use sync-snapshot for dated backups.

B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"

if [ -d "/app/checkpoints" ] && [ "$(ls -A /app/checkpoints 2>/dev/null)" ]; then
  CKPT_DIR="/app/checkpoints"
elif [ -d "/workspace/checkpoints" ] && [ "$(ls -A /workspace/checkpoints 2>/dev/null)" ]; then
  CKPT_DIR="/workspace/checkpoints"
else
  echo "No checkpoints found (nothing to upload)"
  exit 0
fi

echo "Uploading checkpoints to b2:$B2_ARTIFACTS/checkpoints/..."
rclone copy "$CKPT_DIR/" "b2:$B2_ARTIFACTS/checkpoints/" --copy-links --progress
echo "✓ Checkpoints synced"
SCRIPT
chmod +x /usr/local/bin/sync-checkpoints-up

cat > /usr/local/bin/sync-logs-up << 'SCRIPT'
#!/bin/bash
# Upload local logs to B2 (flat structure)

B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"

if [ -d "/app/logs" ] && [ "$(ls -A /app/logs 2>/dev/null)" ]; then
  LOGS_DIR="/app/logs"
elif [ -d "/workspace/logs" ] && [ "$(ls -A /workspace/logs 2>/dev/null)" ]; then
  LOGS_DIR="/workspace/logs"
else
  echo "No logs found (nothing to upload)"
  exit 0
fi

echo "Uploading logs to b2:$B2_ARTIFACTS/logs/..."
rclone copy "$LOGS_DIR/" "b2:$B2_ARTIFACTS/logs/" --copy-links --progress
echo "✓ Logs synced"
SCRIPT
chmod +x /usr/local/bin/sync-logs-up

cat > /usr/local/bin/sync-cache-up << 'SCRIPT'
#!/bin/bash
# Upload local cache to B2 (flat structure)

B2_ARTIFACTS="${B2_ARTIFACTS:-${B2_BUCKET:-your-artifacts-bucket}}"

if [ -d "/workspace/cache" ] && [ "$(ls -A /workspace/cache 2>/dev/null)" ]; then
  CACHE_DIR="/workspace/cache"
else
  echo "No cache found (nothing to upload)"
  exit 0
fi

echo "Cache contents:"
du -sh "$CACHE_DIR"/* 2>/dev/null | head -5

echo "Uploading cache to b2:$B2_ARTIFACTS/cache/..."
rclone copy "$CACHE_DIR/" "b2:$B2_ARTIFACTS/cache/" --copy-links --progress
echo "✓ Cache synced"
SCRIPT
chmod +x /usr/local/bin/sync-cache-up

cat > /usr/local/bin/sync-cache-down << 'SCRIPT'
#!/bin/bash
# Download cache from B2
# Usage: sync-cache-down [--snapshot YYYY-MM-DD]

B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"
mkdir -p /workspace/cache

if [ "$1" = "--snapshot" ] && [ -n "$2" ]; then
  echo "Downloading cache from snapshot $2..."
  rclone copy "b2:$B2_ARTIFACTS/snapshots/$2/cache/" /workspace/cache/ --progress
else
  echo "Downloading cache from b2:$B2_ARTIFACTS/cache/..."
  rclone copy "b2:$B2_ARTIFACTS/cache/" /workspace/cache/ --progress
fi
echo "✓ Cache downloaded"
SCRIPT
chmod +x /usr/local/bin/sync-cache-down

cat > /usr/local/bin/sync-all-up << 'SCRIPT'
#!/bin/bash
# Upload checkpoints, logs, and cache to B2 (flat structure)
# rclone skips identical files automatically.

echo "=== Syncing checkpoints ==="
sync-checkpoints-up

echo ""
echo "=== Syncing logs ==="
sync-logs-up

echo ""
echo "=== Syncing cache ==="
sync-cache-up

echo ""
echo "✓ All synced to B2"
SCRIPT
chmod +x /usr/local/bin/sync-all-up

# Snapshot command for dated backups
cat > /usr/local/bin/sync-snapshot << 'SCRIPT'
#!/bin/bash
# Create a dated snapshot of checkpoints, logs, and cache
# Usage: sync-snapshot [YYYY-MM-DD]
#
# Creates point-in-time backups in snapshots/YYYY-MM-DD/

B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"
DATE="${1:-$(date +%Y-%m-%d)}"

echo "Creating snapshot for $DATE..."

if [ -d "/workspace/checkpoints" ] && [ "$(ls -A /workspace/checkpoints 2>/dev/null)" ]; then
  echo "  Snapshotting checkpoints..."
  rclone copy /workspace/checkpoints/ "b2:$B2_ARTIFACTS/snapshots/$DATE/checkpoints/" --copy-links --progress
fi

if [ -d "/workspace/logs" ] && [ "$(ls -A /workspace/logs 2>/dev/null)" ]; then
  echo "  Snapshotting logs..."
  rclone copy /workspace/logs/ "b2:$B2_ARTIFACTS/snapshots/$DATE/logs/" --copy-links --progress
fi

if [ -d "/workspace/cache" ] && [ "$(ls -A /workspace/cache 2>/dev/null)" ]; then
  echo "  Snapshotting cache..."
  rclone copy /workspace/cache/ "b2:$B2_ARTIFACTS/snapshots/$DATE/cache/" --copy-links --progress
fi

echo "✓ Snapshot created at b2:$B2_ARTIFACTS/snapshots/$DATE/"
SCRIPT
chmod +x /usr/local/bin/sync-snapshot

# List snapshots
cat > /usr/local/bin/list-snapshots << 'SCRIPT'
#!/bin/bash
B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"
echo "Snapshots on B2:"
rclone lsd "b2:$B2_ARTIFACTS/snapshots/" 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
SCRIPT
chmod +x /usr/local/bin/list-snapshots
chmod +x /usr/local/bin/sync-all-up

cat > /usr/local/bin/sync-checkpoints-down << 'SCRIPT'
#!/bin/bash
# Download checkpoints from B2
# Usage: sync-checkpoints-down [--snapshot YYYY-MM-DD]
#
# Default: download from checkpoints/ (flat)
# --snapshot YYYY-MM-DD: download from a specific snapshot

B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"
mkdir -p /workspace/checkpoints

if [ "$1" = "--snapshot" ] && [ -n "$2" ]; then
  echo "Downloading checkpoints from snapshot $2..."
  rclone copy "b2:$B2_ARTIFACTS/snapshots/$2/checkpoints/" /workspace/checkpoints/ --progress
else
  echo "Downloading checkpoints from b2:$B2_ARTIFACTS/checkpoints/..."
  rclone copy "b2:$B2_ARTIFACTS/checkpoints/" /workspace/checkpoints/ --progress
fi
echo "✓ Checkpoints downloaded"
SCRIPT
chmod +x /usr/local/bin/sync-checkpoints-down

cat > /usr/local/bin/list-checkpoints << 'SCRIPT'
#!/bin/bash
# List checkpoints on B2 or locally
# Usage: list-checkpoints [--local | --remote]

B2_ARTIFACTS="${B2_ARTIFACTS:-your-artifacts-bucket}"

if [ "$1" = "--local" ]; then
  echo "Local checkpoints (/workspace/checkpoints/):"
  ls -lh /workspace/checkpoints/ 2>/dev/null || echo "  (empty)"
elif [ "$1" = "--remote" ] || [ -z "$1" ]; then
  echo "Checkpoints on B2 (b2:$B2_ARTIFACTS/checkpoints/):"
  rclone ls "b2:$B2_ARTIFACTS/checkpoints/" 2>/dev/null || echo "  (none)"
else
  echo "Usage: list-checkpoints [--local | --remote]"
fi
SCRIPT
chmod +x /usr/local/bin/list-checkpoints

echo "=== Entrypoint complete ==="
echo ""
echo "Sync commands (flat structure, rclone skips identical files):"
echo "  sync-all-up                   # Upload checkpoints + logs + cache"
echo "  sync-checkpoints-up           # Upload checkpoints"
echo "  sync-checkpoints-down         # Download checkpoints"
echo "  sync-logs-up                  # Upload logs"
echo "  sync-cache-up                 # Upload cache"
echo "  sync-cache-down               # Download cache"
echo ""
echo "Snapshots (dated backups):"
echo "  sync-snapshot                 # Create snapshot with today's date"
echo "  sync-snapshot 2026-02-01      # Create snapshot with specific date"
echo "  list-snapshots                # List available snapshots"
echo "  sync-checkpoints-down --snapshot 2026-02-01  # Restore from snapshot"
echo ""
echo "  list-checkpoints              # List checkpoints on B2"
echo "  list-checkpoints --local      # List local checkpoints"
echo ""
echo "Buckets: B2_REPLAYS=${B2_REPLAYS:-your-replays-bucket} B2_ARTIFACTS=${B2_ARTIFACTS:-your-artifacts-bucket}"
echo ""

# Run whatever command was passed (or default to bash)
exec "$@"
