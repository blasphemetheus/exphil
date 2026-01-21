#!/bin/bash
# Fetch replays from cloud storage (Backblaze B2 or Cloudflare R2)
#
# Usage:
#   # Set env vars then run:
#   export B2_KEY_ID="..." B2_APP_KEY="..." B2_BUCKET="exphil-replays"
#   ./scripts/fetch_replays.sh
#
#   # Or with R2:
#   export R2_ACCESS_KEY="..." R2_SECRET_KEY="..." R2_ENDPOINT="..." R2_BUCKET="..."
#   ./scripts/fetch_replays.sh
#
# Environment Variables:
#   B2_KEY_ID, B2_APP_KEY, B2_BUCKET    - For Backblaze B2
#   R2_ACCESS_KEY, R2_SECRET_KEY, R2_ENDPOINT, R2_BUCKET - For Cloudflare R2
#   REPLAY_DIR                          - Target directory (default: /workspace/replays)
#   REPLAY_SUBDIR                       - Specific subdirectory to sync (e.g., "mewtwo")

set -e

REPLAY_DIR="${REPLAY_DIR:-/workspace/replays}"
RCLONE_CONFIG="/tmp/rclone.conf"

log() {
    echo "[fetch_replays] $(date '+%H:%M:%S') $1" >&2
}

error() {
    echo "[fetch_replays] ERROR: $1" >&2
    exit 1
}

# Install rclone if not present
install_rclone() {
    if command -v rclone &> /dev/null; then
        log "rclone already installed: $(rclone version | head -1)"
        return 0
    fi

    log "Installing rclone..."
    curl -s https://rclone.org/install.sh | bash

    if ! command -v rclone &> /dev/null; then
        error "Failed to install rclone"
    fi
    log "rclone installed: $(rclone version | head -1)"
}

# Configure rclone for Backblaze B2
configure_b2() {
    log "Configuring rclone for Backblaze B2..."
    cat > "$RCLONE_CONFIG" <<EOF
[b2]
type = b2
account = ${B2_KEY_ID}
key = ${B2_APP_KEY}
EOF
    echo "b2:${B2_BUCKET}"
}

# Configure rclone for Cloudflare R2
configure_r2() {
    log "Configuring rclone for Cloudflare R2..."
    cat > "$RCLONE_CONFIG" <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY}
secret_access_key = ${R2_SECRET_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF
    echo "r2:${R2_BUCKET}"
}

# Main
main() {
    log "Starting replay fetch..."

    # Determine storage provider
    REMOTE=""
    if [[ -n "$B2_KEY_ID" && -n "$B2_APP_KEY" && -n "$B2_BUCKET" ]]; then
        install_rclone
        REMOTE=$(configure_b2)
        log "Using Backblaze B2: $REMOTE"
    elif [[ -n "$R2_ACCESS_KEY" && -n "$R2_SECRET_KEY" && -n "$R2_ENDPOINT" && -n "$R2_BUCKET" ]]; then
        install_rclone
        REMOTE=$(configure_r2)
        log "Using Cloudflare R2: $REMOTE"
    else
        log "No cloud storage configured. Set B2_* or R2_* environment variables."
        log "See docs/REPLAY_STORAGE.md for setup instructions."
        log "Skipping replay fetch."
        exit 0
    fi

    # Add subdirectory if specified
    if [[ -n "$REPLAY_SUBDIR" ]]; then
        REMOTE="${REMOTE}/${REPLAY_SUBDIR}"
        REPLAY_DIR="${REPLAY_DIR}/${REPLAY_SUBDIR}"
    fi

    # Create target directory
    mkdir -p "$REPLAY_DIR"

    # Sync replays
    log "Syncing from $REMOTE to $REPLAY_DIR..."
    rclone sync "$REMOTE" "$REPLAY_DIR" \
        --config "$RCLONE_CONFIG" \
        --progress \
        --transfers 8 \
        --checkers 16

    # Report results
    REPLAY_COUNT=$(find "$REPLAY_DIR" -name "*.slp" 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh "$REPLAY_DIR" 2>/dev/null | cut -f1)

    log "Sync complete!"
    log "  Replays: $REPLAY_COUNT files"
    log "  Size: $TOTAL_SIZE"
    log "  Location: $REPLAY_DIR"

    # Cleanup
    rm -f "$RCLONE_CONFIG"
}

main "$@"
