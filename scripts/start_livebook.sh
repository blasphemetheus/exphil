#!/bin/bash
# Start Livebook with the evaluation dashboard
#
# Usage:
#   ./scripts/start_livebook.sh
#
# The token is fixed so you can bookmark the URL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NOTEBOOK="$PROJECT_DIR/notebooks/evaluation_dashboard.livemd"
PORT="${LIVEBOOK_PORT:-8080}"
TOKEN="exphil-dev-token"

# Find livebook executable
LIVEBOOK="${HOME}/.asdf/installs/elixir/1.18.4-otp-28/.mix/escripts/livebook"
if [ ! -f "$LIVEBOOK" ]; then
  LIVEBOOK="$(which livebook 2>/dev/null || echo "")"
fi

if [ -z "$LIVEBOOK" ] || [ ! -f "$LIVEBOOK" ]; then
  echo "Livebook not found. Install with:"
  echo "  mix escript.install hex livebook"
  exit 1
fi

# Kill any existing Livebook on this port
pkill -f "livebook.*--port $PORT" 2>/dev/null || true
sleep 1

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ExPhil Livebook Dashboard                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting Livebook..."
echo ""
echo "  URL: http://localhost:$PORT/?token=$TOKEN"
echo ""
echo "  Notebook: $NOTEBOOK"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start Livebook with fixed token
LIVEBOOK_TOKEN_ENABLED=true \
LIVEBOOK_TOKEN="$TOKEN" \
"$LIVEBOOK" server "$NOTEBOOK" --port "$PORT"
