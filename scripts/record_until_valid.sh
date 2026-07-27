#!/usr/bin/env bash
# Record a live run and RETRY until the replay is actually parseable.
#
#   scripts/record_until_valid.sh <out.slp> <log.log> -- <play_dolphin_async args...>
#
# Why this exists: ~30% of runs produce a TRUNCATED .slp that peppi rejects
# ("failed to fill whole buffer"), losing the entire recording. Slippi only
# finalizes on game end, and the graceful-SD path that ends the game is racy —
# AsyncRunner.stop/1 does Process.exit(frame_loop_pid, :shutdown), which tears
# down the Python bridge ("Parent process died — force exiting", Dolphin exits
# code 9). GracefulSD then talks to a corpse. It wins the race often enough to
# look reliable, which is worse than failing outright.
#
# Fixing that properly means restructuring the runner's shutdown so SD happens
# INSIDE the frame loop rather than after it. Until then, a truncated replay is
# cheaply DETECTABLE, so verify and retry instead of silently losing a seed.
#
# Always cleans up Dolphins by matching [A]ppRun.wrapped (the mounted AppImage,
# not the launcher path) — an orphan keeps Slippi's fixed console port 51441
# and poisons the NEXT run, which presents as "stuck at MAIN_MENU forever".

set -uo pipefail

OUT_SLP="$1"; shift
OUT_LOG="$1"; shift
[ "${1:-}" = "--" ] && shift

MAX_ATTEMPTS="${RECORD_MAX_ATTEMPTS:-3}"

kill_dolphins() {
  ps -eo pid,args | grep '[A]ppRun.wrapped' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 2
  rm -rf /tmp/libmelee_* 2>/dev/null
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  kill_dolphins
  echo "[record] attempt $attempt/$MAX_ATTEMPTS $(date +%H:%M:%S)"

  XLA_TARGET=cpu timeout "${RECORD_TIMEOUT:-480}" mix run scripts/play_dolphin_async.exs "$@" \
    > "$OUT_LOG" 2>&1

  kill_dolphins

  newest="$(ls -t ~/Slippi/*.slp 2>/dev/null | head -1)"
  if [ -z "$newest" ]; then
    echo "[record]   no .slp produced"
    continue
  fi

  cp "$newest" "$OUT_SLP"

  # The verification that matters: py-slippi refuses a truncated file, same as
  # peppi. Cheap, and it is the exact failure we are guarding against.
  if .venv/bin/python -c "
from slippi import Game
import sys
try:
    g = Game(sys.argv[1])
    sys.exit(0 if len(g.frames) > 600 else 1)
except Exception:
    sys.exit(1)
" "$OUT_SLP" 2>/dev/null; then
    echo "[record]   OK — $(basename "$OUT_SLP") parses"
    exit 0
  fi

  echo "[record]   TRUNCATED, retrying"
  rm -f "$OUT_SLP"
done

echo "[record] FAILED after $MAX_ATTEMPTS attempts — no valid replay"
exit 1
