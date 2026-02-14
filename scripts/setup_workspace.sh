#!/bin/bash
# First-time workspace setup for Network Volume
# Run this once after attaching a new network volume to a pod
#
# Usage:
#   /app/scripts/setup_workspace.sh
#   /app/scripts/setup_workspace.sh --skip-replays  # Skip replay sync

set -e

# Configuration
WORKSPACE="${WORKSPACE:-/workspace}"
REPLAY_DIR="${REPLAY_DIR:-$WORKSPACE/replays}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/logs}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} ${GREEN}$1${NC}"; }
warn() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} ${YELLOW}$1${NC}"; }
error() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} ${RED}$1${NC}"; }

banner() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          ExPhil Workspace Setup                            ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_cloud_storage() {
    # Check B2
    if [ -n "$B2_KEY_ID" ] && [ -n "$B2_APP_KEY" ] && [ -n "$B2_BUCKET" ]; then
        if ! rclone listremotes 2>/dev/null | grep -q "^b2:$"; then
            log "Configuring rclone for B2..."
            rclone config create b2 b2 account="$B2_KEY_ID" key="$B2_APP_KEY" > /dev/null 2>&1
        fi
        REMOTE="b2:$B2_BUCKET"
        REMOTE_TYPE="B2"
        return 0
    fi

    # Check R2
    if [ -n "$R2_ACCESS_KEY" ] && [ -n "$R2_SECRET_KEY" ] && [ -n "$R2_ENDPOINT" ] && [ -n "$R2_BUCKET" ]; then
        if ! rclone listremotes 2>/dev/null | grep -q "^r2:$"; then
            log "Configuring rclone for R2..."
            rclone config create r2 s3 provider=Cloudflare access_key_id="$R2_ACCESS_KEY" secret_access_key="$R2_SECRET_KEY" endpoint="$R2_ENDPOINT" > /dev/null 2>&1
        fi
        REMOTE="r2:$R2_BUCKET"
        REMOTE_TYPE="R2"
        return 0
    fi

    # Check existing rclone config
    if rclone listremotes 2>/dev/null | grep -q "^b2:$"; then
        REMOTE="b2:${B2_BUCKET:-your-replays-bucket}"
        REMOTE_TYPE="B2 (existing)"
        return 0
    fi

    return 1
}

# Parse arguments
SKIP_REPLAYS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-replays)
            SKIP_REPLAYS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-replays]"
            echo ""
            echo "Options:"
            echo "  --skip-replays    Skip syncing replays from cloud storage"
            echo ""
            echo "Environment Variables:"
            echo "  B2_KEY_ID, B2_APP_KEY, B2_BUCKET    Backblaze B2 credentials"
            echo "  R2_ACCESS_KEY, R2_SECRET_KEY,       Cloudflare R2 credentials"
            echo "  R2_ENDPOINT, R2_BUCKET"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

banner

# Step 1: Check workspace mount
log "Step 1/5: Checking workspace mount..."
if [ ! -d "$WORKSPACE" ]; then
    error "Workspace not found at $WORKSPACE"
    error "Make sure the Network Volume is attached at /workspace"
    exit 1
fi

# Check if it's a mounted network volume or just container storage
USING_NETWORK_VOLUME=false
if mountpoint -q "$WORKSPACE" 2>/dev/null; then
    USING_NETWORK_VOLUME=true
    success "Network Volume detected at $WORKSPACE"
elif [ -f "$WORKSPACE/.volume_marker" ]; then
    # Previously set up, check if it was a network volume
    if grep -q "network_volume=true" "$WORKSPACE/.volume_marker" 2>/dev/null; then
        USING_NETWORK_VOLUME=true
        success "Network Volume detected (from marker)"
    else
        warn "Container storage detected (data lost on pod terminate)"
        warn "Using cloud storage sync for persistence"
    fi
else
    warn "No Network Volume detected - using container storage"
    warn "This is normal for Community Cloud pods"
    warn "Data will sync to/from cloud storage for persistence"
fi

# Step 2: Create directory structure
log "Step 2/5: Creating directory structure..."
mkdir -p "$REPLAY_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

# Create marker files
if [ "$USING_NETWORK_VOLUME" = true ]; then
    echo "network_volume=true" > "$WORKSPACE/.volume_marker"
else
    echo "network_volume=false" > "$WORKSPACE/.volume_marker"
fi
echo "$(date -Iseconds)" > "$WORKSPACE/.setup_timestamp"

success "Created directories:"
echo "    $REPLAY_DIR"
echo "    $CHECKPOINT_DIR"
echo "    $LOG_DIR"

# Step 3: Check GPU
log "Step 3/5: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    success "GPU: $GPU_NAME ($GPU_MEM)"
else
    warn "nvidia-smi not found - GPU may not be available"
fi

# Step 4: Configure cloud storage
log "Step 4/5: Configuring cloud storage..."
if check_cloud_storage; then
    success "Cloud storage configured: $REMOTE_TYPE"

    # Check what's in the bucket
    REMOTE_REPLAYS=$(rclone size "$REMOTE" --json 2>/dev/null | grep -o '"count":[0-9]*' | cut -d: -f2 || echo "0")
    log "Remote has approximately $REMOTE_REPLAYS files"
else
    warn "No cloud storage configured"
    warn "Set B2_KEY_ID, B2_APP_KEY, B2_BUCKET in RunPod environment variables"
    warn "Skipping replay sync"
    SKIP_REPLAYS=true
fi

# Step 5: Sync replays
log "Step 5/5: Syncing replays..."
if [ "$SKIP_REPLAYS" = true ]; then
    warn "Skipping replay sync (--skip-replays flag)"
elif [ -n "$REMOTE" ]; then
    LOCAL_COUNT=$(find "$REPLAY_DIR" -name "*.slp" 2>/dev/null | wc -l)

    if [ "$LOCAL_COUNT" -gt 0 ]; then
        log "Found $LOCAL_COUNT local replay files"
        read -p "Sync from cloud anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            warn "Skipping replay sync"
        else
            log "Syncing replays from $REMOTE..."
            rclone sync "$REMOTE" "$REPLAY_DIR" --progress --exclude "checkpoints/**"
            success "Replay sync complete"
        fi
    else
        log "Syncing replays from $REMOTE..."
        rclone sync "$REMOTE" "$REPLAY_DIR" --progress --exclude "checkpoints/**"
        success "Replay sync complete"
    fi

    REPLAY_COUNT=$(find "$REPLAY_DIR" -name "*.slp" 2>/dev/null | wc -l)
    success "Total replays available: $REPLAY_COUNT"
fi

# Summary
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          Setup Complete                                    ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Workspace: $WORKSPACE"
echo "Replays:   $(find "$REPLAY_DIR" -name "*.slp" 2>/dev/null | wc -l) files"
echo "Storage:   ${REMOTE_TYPE:-Not configured}"
if [ "$USING_NETWORK_VOLUME" = true ]; then
    echo "Volume:    Network Volume (data persists on terminate)"
else
    echo "Volume:    Container storage (data lost on terminate - use auto-sync!)"
fi
echo ""
echo "Next steps:"
echo ""
echo "  1. Start a tmux session for training:"
echo "     ${CYAN}tmux new -s train${NC}"
echo ""
if [ "$USING_NETWORK_VOLUME" = false ]; then
    echo "  2. ${YELLOW}IMPORTANT:${NC} Start auto-sync in a split pane (Ctrl+B, %):"
    echo "     ${CYAN}/app/scripts/auto_sync_checkpoints.sh${NC}"
    echo "     This backs up checkpoints every 10 min so you don't lose work!"
    echo ""
    echo "  3. Run training:"
else
    echo "  2. Run training:"
fi
echo "     ${CYAN}cd /app${NC}"
echo "     ${CYAN}mix run scripts/train_from_replays.exs --preset gpu_quick --replays $REPLAY_DIR${NC}"
echo ""
if [ "$USING_NETWORK_VOLUME" = true ]; then
    echo "  3. (Optional) Start auto-sync for cloud backup:"
    echo "     ${CYAN}/app/scripts/auto_sync_checkpoints.sh${NC}"
    echo ""
fi
echo "  After training, sync checkpoints to cloud:"
echo "     ${CYAN}/app/scripts/auto_sync_checkpoints.sh --once${NC}"
echo ""
if [ "$USING_NETWORK_VOLUME" = false ]; then
    echo -e "${YELLOW}Remember: You're on Community Cloud - run auto-sync to avoid losing work!${NC}"
    echo ""
fi
