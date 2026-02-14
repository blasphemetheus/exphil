#!/bin/bash
# Auto-sync checkpoints to cloud storage
# Run in background during training to prevent data loss
#
# Usage:
#   ./scripts/auto_sync_checkpoints.sh &              # Background
#   ./scripts/auto_sync_checkpoints.sh --interval 300 # Every 5 minutes
#   ./scripts/auto_sync_checkpoints.sh --once         # Single sync, then exit

set -e

# Configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"  # 10 minutes default
LOG_FILE="${LOG_FILE:-/workspace/logs/sync.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] $1" | tee -a "$LOG_FILE"
}

check_rclone_config() {
    # Check if B2 is configured
    if [ -n "$B2_KEY_ID" ] && [ -n "$B2_APP_KEY" ] && [ -n "$B2_BUCKET" ]; then
        # Configure rclone on-the-fly if not already configured
        if ! rclone listremotes | grep -q "^b2:$"; then
            log "Configuring rclone for B2..."
            rclone config create b2 b2 account="$B2_KEY_ID" key="$B2_APP_KEY" > /dev/null 2>&1
        fi
        REMOTE="b2:$B2_BUCKET/checkpoints"
        return 0
    fi

    # Check if R2 is configured
    if [ -n "$R2_ACCESS_KEY" ] && [ -n "$R2_SECRET_KEY" ] && [ -n "$R2_ENDPOINT" ] && [ -n "$R2_BUCKET" ]; then
        if ! rclone listremotes | grep -q "^r2:$"; then
            log "Configuring rclone for R2..."
            rclone config create r2 s3 provider=Cloudflare access_key_id="$R2_ACCESS_KEY" secret_access_key="$R2_SECRET_KEY" endpoint="$R2_ENDPOINT" > /dev/null 2>&1
        fi
        REMOTE="r2:$R2_BUCKET/checkpoints"
        return 0
    fi

    # Check if rclone already has a remote configured
    if rclone listremotes | grep -q "^b2:$"; then
        REMOTE="b2:${B2_BUCKET:-your-replays-bucket}/checkpoints"
        return 0
    fi

    return 1
}

do_sync() {
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        log "${YELLOW}Warning: $CHECKPOINT_DIR does not exist, skipping sync${NC}"
        return 0
    fi

    local file_count=$(find "$CHECKPOINT_DIR" -name "*.axon" -o -name "*.bin" -o -name "*.json" 2>/dev/null | wc -l)
    if [ "$file_count" -eq 0 ]; then
        log "No checkpoint files found, skipping sync"
        return 0
    fi

    log "Syncing $file_count checkpoint files to $REMOTE..."

    if rclone sync "$CHECKPOINT_DIR" "$REMOTE" --progress 2>&1 | tee -a "$LOG_FILE"; then
        log "${GREEN}Sync complete${NC}"
        return 0
    else
        log "${RED}Sync failed${NC}"
        return 1
    fi
}

show_help() {
    cat << EOF
Auto-sync checkpoints to cloud storage (B2 or R2)

Usage: $0 [OPTIONS]

Options:
    --interval SECONDS   Sync interval in seconds (default: 600)
    --once               Sync once and exit
    --checkpoint-dir DIR Directory to sync (default: /workspace/checkpoints)
    --help               Show this help message

Environment Variables:
    B2_KEY_ID, B2_APP_KEY, B2_BUCKET    Backblaze B2 credentials
    R2_ACCESS_KEY, R2_SECRET_KEY,       Cloudflare R2 credentials
    R2_ENDPOINT, R2_BUCKET
    CHECKPOINT_DIR                       Override checkpoint directory
    SYNC_INTERVAL                        Override sync interval
    LOG_FILE                             Override log file path

Examples:
    # Run in background with default 10-minute interval
    $0 &

    # Sync every 5 minutes
    $0 --interval 300 &

    # Single sync (useful for end-of-training)
    $0 --once

    # In a tmux pane
    tmux split-window -h '$0'
EOF
}

# Parse arguments
ONCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            SYNC_INTERVAL="$2"
            shift 2
            ;;
        --once)
            ONCE=true
            shift
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Check rclone configuration
if ! check_rclone_config; then
    log "${RED}Error: No cloud storage configured${NC}"
    log "Set B2_KEY_ID, B2_APP_KEY, B2_BUCKET for Backblaze B2"
    log "Or R2_ACCESS_KEY, R2_SECRET_KEY, R2_ENDPOINT, R2_BUCKET for Cloudflare R2"
    exit 1
fi

log "Auto-sync started"
log "  Checkpoint dir: $CHECKPOINT_DIR"
log "  Remote: $REMOTE"
log "  Interval: ${SYNC_INTERVAL}s"
log "  Log file: $LOG_FILE"

if [ "$ONCE" = true ]; then
    do_sync
    exit $?
fi

# Trap to sync on exit
trap 'log "Received shutdown signal, performing final sync..."; do_sync; exit 0' SIGINT SIGTERM

# Main loop
while true; do
    do_sync
    log "Next sync in ${SYNC_INTERVAL}s (Ctrl+C to stop)"
    sleep "$SYNC_INTERVAL"
done
