#!/usr/bin/env bash
# Pull replays from the B2 archive into the local corpus (task #30 BC
# vocabulary + #33 five-char char corpora). Uses the existing `b2`
# rclone remote (~/.config/rclone/rclone.conf).
#
#   scripts/pull_replays.sh mewtwo                 # -> corpus/archive/mewtwo
#   scripts/pull_replays.sh greg/ganondorf         # -> corpus/archive/greg/ganondorf
#   scripts/pull_replays.sh --list                 # show archive tree
#
# corpus/ is gitignored. rclone is in devenv (pkgs.rclone) OR via
# `nix run nixpkgs#rclone`. Distinct from corpus/human_blewf/ (Bradley's
# OWN games — calibration); this archive is other players' replays for
# neutral-vocabulary BC and per-character corpora.

set -euo pipefail

BUCKET="b2:exphil-replays-blewfargs"
DEST_ROOT="$(cd "$(dirname "$0")/.." && pwd)/corpus/archive"

rc() { command -v rclone >/dev/null 2>&1 && rclone "$@" || nix run nixpkgs#rclone -- "$@"; }

if [ "${1:-}" = "--list" ]; then
  rc lsd "$BUCKET"
  echo "--- greg (by character) ---"
  rc lsd "$BUCKET/greg"
  exit 0
fi

SUB="${1:?usage: pull_replays.sh <bucket-subpath> | --list}"
DEST="$DEST_ROOT/$SUB"
mkdir -p "$DEST"

echo "[pull] $BUCKET/$SUB -> $DEST"
rc copy "$BUCKET/$SUB" "$DEST" --progress --transfers 8 --fast-list
echo "[pull] done: $(find "$DEST" -name '*.slp' | wc -l) .slp files"
