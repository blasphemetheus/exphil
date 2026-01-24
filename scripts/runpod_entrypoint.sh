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

# Link checkpoint dirs if not already linked
if [ -d "/app" ]; then
  ln -sf /workspace/checkpoints /app/checkpoints 2>/dev/null || true
  ln -sf /workspace/logs /app/logs 2>/dev/null || true
fi

# Create helper scripts for checkpoint management
cat > /usr/local/bin/sync-checkpoints-up << 'SCRIPT'
#!/bin/bash
# Upload local checkpoints to B2 (organized by pod ID)
# Usage: sync-checkpoints-up [--all]

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"
POD_ID="${POD_ID:-$(hostname)_$(date +%Y%m%d)}"

if [ "$1" = "--all" ]; then
  # Upload to shared checkpoints folder (flattened)
  echo "Uploading all checkpoints to b2:$B2_BUCKET/checkpoints/..."
  rclone copy /workspace/checkpoints/ "b2:$B2_BUCKET/checkpoints/" --progress
else
  # Upload to pod-specific folder (avoids collisions)
  echo "Uploading checkpoints to b2:$B2_BUCKET/checkpoints/$POD_ID/..."
  rclone copy /workspace/checkpoints/ "b2:$B2_BUCKET/checkpoints/$POD_ID/" --progress
fi
echo "✓ Checkpoints uploaded"
SCRIPT
chmod +x /usr/local/bin/sync-checkpoints-up

cat > /usr/local/bin/sync-checkpoints-down << 'SCRIPT'
#!/bin/bash
# Download checkpoints from B2
# Usage: sync-checkpoints-down [pod_id]

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"

if [ -n "$1" ]; then
  # Download from specific pod folder
  echo "Downloading checkpoints from b2:$B2_BUCKET/checkpoints/$1/..."
  rclone copy "b2:$B2_BUCKET/checkpoints/$1/" /workspace/checkpoints/ --progress
else
  # Download all checkpoints (flattened)
  echo "Downloading all checkpoints from b2:$B2_BUCKET/checkpoints/..."
  rclone copy "b2:$B2_BUCKET/checkpoints/" /workspace/checkpoints/ --progress
fi
echo "✓ Checkpoints downloaded"
SCRIPT
chmod +x /usr/local/bin/sync-checkpoints-down

cat > /usr/local/bin/list-checkpoints << 'SCRIPT'
#!/bin/bash
# List checkpoints on B2
# Usage: list-checkpoints [--local]

B2_BUCKET="${B2_BUCKET:-exphil-replays-blewfargs}"

if [ "$1" = "--local" ]; then
  echo "Local checkpoints in /workspace/checkpoints/:"
  ls -lah /workspace/checkpoints/ 2>/dev/null || echo "  (none)"
else
  echo "Checkpoints on B2 (b2:$B2_BUCKET/checkpoints/):"
  echo ""
  echo "Pod folders:"
  rclone lsd "b2:$B2_BUCKET/checkpoints/" 2>/dev/null || echo "  (none)"
  echo ""
  echo "Files in root:"
  rclone ls "b2:$B2_BUCKET/checkpoints/" --max-depth 1 2>/dev/null | head -20 || echo "  (none)"
fi
SCRIPT
chmod +x /usr/local/bin/list-checkpoints

echo "=== Entrypoint complete ==="
echo ""
echo "Checkpoint sync commands available:"
echo "  sync-checkpoints-up      # Upload to B2 (pod-specific folder)"
echo "  sync-checkpoints-up --all # Upload to B2 (shared folder)"
echo "  sync-checkpoints-down    # Download all from B2"
echo "  sync-checkpoints-down <pod_id>  # Download from specific pod"
echo "  list-checkpoints         # List checkpoints on B2"
echo "  list-checkpoints --local # List local checkpoints"
echo ""

# Run whatever command was passed (or default to bash)
exec "$@"
