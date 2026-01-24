#!/bin/bash
# ExPhil RunPod Sync Script
# Syncs checkpoints and logs between RunPod, B2, and local machine

set -e

# Configuration
B2_BUCKET="b2:exphil-replays-blewfargs"
LOCAL_DIR="${EXPHIL_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}"
RUNPOD_CHECKPOINTS="/workspace/checkpoints"
RUNPOD_LOGS="/workspace/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

usage() {
    echo "ExPhil RunPod Sync Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  push        Push checkpoints/logs from RunPod to B2"
    echo "  pull        Pull checkpoints/logs from B2 to local"
    echo "  status      Show what would be synced (dry-run)"
    echo "  list        List checkpoints on B2"
    echo ""
    echo "Options:"
    echo "  --checkpoints    Sync only checkpoints (default: both)"
    echo "  --logs           Sync only logs"
    echo "  --all            Sync everything including replays"
    echo "  --dry-run        Show what would be synced without syncing"
    echo ""
    echo "Examples:"
    echo "  # On RunPod after training:"
    echo "  ./scripts/sync_runpod.sh push"
    echo ""
    echo "  # On local machine to download:"
    echo "  ./scripts/sync_runpod.sh pull"
    echo ""
    echo "  # Check what's available:"
    echo "  ./scripts/sync_runpod.sh list"
}

check_rclone() {
    if ! command -v rclone &> /dev/null; then
        echo -e "${RED}Error: rclone is not installed${NC}"
        echo "Install with: curl https://rclone.org/install.sh | sudo bash"
        exit 1
    fi

    if ! rclone listremotes | grep -q "^b2:"; then
        echo -e "${RED}Error: B2 remote not configured in rclone${NC}"
        echo "Run: rclone config"
        exit 1
    fi
}

detect_environment() {
    if [[ -d "/workspace" ]]; then
        echo "runpod"
    else
        echo "local"
    fi
}

push_to_b2() {
    local dry_run=$1
    local what=$2

    echo -e "${CYAN}Pushing to B2...${NC}"

    local flags="--progress"
    [[ "$dry_run" == "true" ]] && flags="$flags --dry-run"

    if [[ "$what" == "all" || "$what" == "checkpoints" ]]; then
        echo -e "${YELLOW}Syncing checkpoints...${NC}"
        if [[ -d "$RUNPOD_CHECKPOINTS" ]]; then
            rclone copy "$RUNPOD_CHECKPOINTS/" "$B2_BUCKET/checkpoints/" $flags
            echo -e "${GREEN}Checkpoints synced${NC}"
        else
            echo -e "${YELLOW}No checkpoints directory found at $RUNPOD_CHECKPOINTS${NC}"
        fi
    fi

    if [[ "$what" == "all" || "$what" == "logs" ]]; then
        echo -e "${YELLOW}Syncing logs...${NC}"
        if [[ -d "$RUNPOD_LOGS" ]]; then
            rclone copy "$RUNPOD_LOGS/" "$B2_BUCKET/logs/" $flags
            echo -e "${GREEN}Logs synced${NC}"
        else
            echo -e "${YELLOW}No logs directory found at $RUNPOD_LOGS${NC}"
        fi
    fi

    echo -e "${GREEN}Push complete!${NC}"
}

pull_from_b2() {
    local dry_run=$1
    local what=$2

    echo -e "${CYAN}Pulling from B2 to $LOCAL_DIR...${NC}"

    local flags="--progress"
    [[ "$dry_run" == "true" ]] && flags="$flags --dry-run"

    if [[ "$what" == "all" || "$what" == "checkpoints" ]]; then
        echo -e "${YELLOW}Syncing checkpoints...${NC}"
        mkdir -p "$LOCAL_DIR/checkpoints"
        rclone copy "$B2_BUCKET/checkpoints/" "$LOCAL_DIR/checkpoints/" $flags
        echo -e "${GREEN}Checkpoints synced${NC}"
    fi

    if [[ "$what" == "all" || "$what" == "logs" ]]; then
        echo -e "${YELLOW}Syncing logs...${NC}"
        mkdir -p "$LOCAL_DIR/logs"
        rclone copy "$B2_BUCKET/logs/" "$LOCAL_DIR/logs/" $flags
        echo -e "${GREEN}Logs synced${NC}"
    fi

    echo -e "${GREEN}Pull complete!${NC}"

    # Show what was downloaded
    echo ""
    echo -e "${CYAN}Downloaded checkpoints:${NC}"
    ls -lh "$LOCAL_DIR/checkpoints/"*.bin 2>/dev/null | tail -10 || echo "  (none)"
}

list_b2() {
    echo -e "${CYAN}Checkpoints on B2:${NC}"
    rclone ls "$B2_BUCKET/checkpoints/" 2>/dev/null | grep -E "\.(axon|bin)$" | sort -k2 || echo "  (none)"

    echo ""
    echo -e "${CYAN}Logs on B2:${NC}"
    rclone ls "$B2_BUCKET/logs/" 2>/dev/null | tail -10 || echo "  (none)"

    echo ""
    echo -e "${CYAN}Total size:${NC}"
    rclone size "$B2_BUCKET/checkpoints/" 2>/dev/null || echo "  (unable to calculate)"
}

show_status() {
    local env=$(detect_environment)

    echo -e "${CYAN}Environment: ${env}${NC}"
    echo ""

    if [[ "$env" == "runpod" ]]; then
        echo -e "${YELLOW}Local checkpoints (RunPod):${NC}"
        ls -lh "$RUNPOD_CHECKPOINTS/"*.bin 2>/dev/null | tail -10 || echo "  (none)"
        echo ""
        echo -e "${YELLOW}Would push to B2:${NC}"
        rclone copy "$RUNPOD_CHECKPOINTS/" "$B2_BUCKET/checkpoints/" --dry-run 2>&1 | head -20
    else
        echo -e "${YELLOW}Local checkpoints:${NC}"
        ls -lh "$LOCAL_DIR/checkpoints/"*.bin 2>/dev/null | tail -10 || echo "  (none)"
        echo ""
        echo -e "${YELLOW}Would pull from B2:${NC}"
        rclone copy "$B2_BUCKET/checkpoints/" "$LOCAL_DIR/checkpoints/" --dry-run 2>&1 | head -20
    fi
}

# Main
check_rclone

COMMAND=${1:-help}
DRY_RUN="false"
WHAT="both"

shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoints)
            WHAT="checkpoints"
            shift
            ;;
        --logs)
            WHAT="logs"
            shift
            ;;
        --all)
            WHAT="all"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Default "both" means checkpoints + logs but not replays
[[ "$WHAT" == "both" ]] && WHAT="all"

case $COMMAND in
    push)
        ENV=$(detect_environment)
        if [[ "$ENV" != "runpod" ]]; then
            echo -e "${YELLOW}Warning: Not on RunPod. Pushing from local directory.${NC}"
            RUNPOD_CHECKPOINTS="$LOCAL_DIR/checkpoints"
            RUNPOD_LOGS="$LOCAL_DIR/logs"
        fi
        push_to_b2 "$DRY_RUN" "$WHAT"
        ;;
    pull)
        pull_from_b2 "$DRY_RUN" "$WHAT"
        ;;
    status)
        show_status
        ;;
    list)
        list_b2
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        usage
        exit 1
        ;;
esac
