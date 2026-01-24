#!/bin/bash
# RunPod entrypoint script - auto-configures services from environment variables

set -e

echo "=== ExPhil RunPod Entrypoint ==="

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

  # Optionally auto-sync replays if SYNC_REPLAYS is set
  if [ "$SYNC_REPLAYS" = "true" ] && [ -n "$B2_BUCKET" ]; then
    echo "Syncing replays from $B2_BUCKET..."
    mkdir -p /workspace/replays
    rclone copy "b2:$B2_BUCKET" /workspace/replays/ --progress
    echo "✓ Replays synced"
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

echo "=== Entrypoint complete ==="
echo ""

# Run whatever command was passed (or default to bash)
exec "$@"
