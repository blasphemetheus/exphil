#!/usr/bin/env bash
# Report-card-on-latest co-tenant tier (task #16, final piece).
#
# Watches the training repo's checkpoints/*_latest.bin (published
# atomically every 10 epochs by dagger_drill) and, whenever a new snapshot
# appears, plays ONE headless probe game with it and appends the tier-1
# report-card score to a CSV — a live mid-training quality curve, by
# gates instead of loss.
#
#   ./scripts/watch_latest.sh            # poll loop (Ctrl+C to stop)
#   ./scripts/watch_latest.sh --once     # score the current snapshot, exit
#
# CO-TENANCY DESIGN (safe to run WHILE a training loop is live):
#   - mix runs from the exphil-play WORKTREE (own _build): the training
#     loop's per-stage recompiles never race this process, and this
#     process never injects code into the loop (CLAUDE.md mid-loop rule).
#     Caveat: nothing may rebuild the exla NIF while either is running
#     (GOTCHAS #50/#59) — app-code recompiles are fine, dep bumps are not.
#   - probe uses EXLA_MEMORY_FRACTION=0.10 next to training's 0.75.
#   - the snapshot is COPIED before probing: _latest.bin is atomically
#     replaced every 10 epochs and must not swap mid-probe.
#   - headless ExiAI Dolphin (GOTCHAS #64): no window, no GC adapter, no
#     Slippi-launcher conflict, ~6-7x realtime.
#
# Env overrides:
#   TRAIN_REPO   training checkout (default ~/git/exphil)
#   PLAY_REPO    worktree to run mix from (default ~/git/exphil-play)
#   EXPHIL_DOLPHIN  dolphin (default ~/.local/share/slippi/exi-ai/dolphin-emu-headless)
#   EXPHIL_ISO      iso (default ~/isos/melee.iso)
#   POLL_SECS    poll interval (default 30)
#   PROBE_MEMFRAC   EXLA memory fraction for the probe (default 0.10)
#   PROBE_EXTRA_FLAGS  extra play_dolphin_async flags (e.g. "--stateful-step")
set -uo pipefail

TRAIN_REPO=${TRAIN_REPO:-$HOME/git/exphil}
PLAY_REPO=${PLAY_REPO:-$HOME/git/exphil-play}
DOLPHIN=${EXPHIL_DOLPHIN:-$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless}
ISO=${EXPHIL_ISO:-$HOME/isos/melee.iso}
POLL_SECS=${POLL_SECS:-30}
MEMFRAC=${PROBE_MEMFRAC:-0.10}
EXTRA_FLAGS=${PROBE_EXTRA_FLAGS:-}
SLIPPI_PORT=51490

CSV=$TRAIN_REPO/logs/latest_scorecards.csv
LOG=$TRAIN_REPO/logs/watch_latest.log
PROBE_DIR=$TRAIN_REPO/logs/latest_probes
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

mkdir -p "$PROBE_DIR" "$(dirname "$CSV")"
[ -f "$CSV" ] || echo "ts,checkpoint,sha,score,total,replay" >"$CSV"

for tool in make cargo; do
  command -v "$tool" >/dev/null || {
    echo "[watch] $tool not on PATH — run me from 'devenv shell' (GOTCHAS #60)"
    exit 1
  }
done

# HARD INTERLOCK (GOTCHAS #67): every mix compile RELINKS the shared
# local-exla libexla.so (nx/exla/cache — symlinked into every consumer's
# _build). A mix invocation here while ANY other BEAM has that .so mapped
# rewrites it under the mapping -> SIGBUS in the other process. This
# exact failure killed r11's first launch 2026-07-16 17:54 (the watcher's
# first probe rebuilt exla and bus-errored the training BEAM 11s later).
# Until per-consumer NIF isolation exists, this watcher REFUSES to run
# while any beam.smp is alive. Override only if you have verified no
# exla-sharing BEAM is running: WATCH_LATEST_UNSAFE=1
if [ -z "${WATCH_LATEST_UNSAFE:-}" ] && pgrep -x beam.smp >/dev/null; then
  echo "[watch] REFUSING: a beam.smp is live and mix compiles relink the"
  echo "[watch] shared exla NIF under it (SIGBUS — killed r11 launch #1)."
  echo "[watch] Co-tenant mode is DISABLED pending per-consumer NIF"
  echo "[watch] isolation. Use --once between runs, or see GOTCHAS #67."
  exit 1
fi
[ -d "$PLAY_REPO" ] || { echo "[watch] play worktree missing: $PLAY_REPO"; exit 1; }

log() { echo "[watch $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

newest_latest() {
  ls -t "$TRAIN_REPO"/checkpoints/*_latest.bin 2>/dev/null | head -1
}

LAST_SHA=""

score_snapshot() {
  local src=$1
  local sha ts snap rdir plog card score replay
  sha=$(sha256sum "$src" | cut -c1-12)
  if [ "$sha" = "$LAST_SHA" ]; then
    return 1
  fi
  ts=$(date +%Y%m%dT%H%M%S)
  snap=$PROBE_DIR/${ts}_$(basename "$src")
  rdir=$PROBE_DIR/${ts}_replays
  plog=$PROBE_DIR/${ts}_probe.log
  cp "$src" "$snap"
  mkdir -p "$rdir"
  log "new snapshot $(basename "$src") sha=$sha — probing..."

  # EXPHIL_PYTHON: worktrees have no .venv of their own (gitignored) and
  # MeleePort's bare-"python3" fallback can't spawn (spawn_executable
  # doesn't search PATH) — use the training checkout's venv (has libmelee).
  (cd "$PLAY_REPO" && EXPHIL_PYTHON=$TRAIN_REPO/.venv/bin/python3 \
    EXLA_MEMORY_FRACTION=$MEMFRAC timeout 600 \
    mix run scripts/play_dolphin_async.exs \
    --policy "$snap" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 0 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --deterministic --on-game-end stop \
    --headless --slippi-port $SLIPPI_PORT \
    --replay-dir "$rdir" $EXTRA_FLAGS) >"$plog" 2>&1
  # BEAM may segfault in EXLA teardown AFTER a clean game (GOTCHAS #64) —
  # judge by replay presence, not exit code.

  replay=$(ls -t "$rdir"/*.slp 2>/dev/null | head -1)
  if [ -z "$replay" ]; then
    log "probe produced NO replay (see $plog) — not scored"
    LAST_SHA=$sha # don't retry the same snapshot in a loop
    return 0
  fi

  card=$(cd "$PLAY_REPO" && mix run scripts/report_card.exs "$replay" 2>/dev/null)
  echo "$card" | tee -a "$LOG"
  score=$(grep -oE 'SCORE: [0-9]+/[0-9]+' <<<"$card" | head -1)
  if [ -n "$score" ]; then
    IFS=/ read -r passed total <<<"${score#SCORE: }"
    echo "$ts,$(basename "$src"),$sha,$passed,$total,$replay" >>"$CSV"
    log "scored $passed/$total -> $CSV"
  else
    log "report card produced no SCORE line (replay unparseable?) — see $LOG"
  fi
  LAST_SHA=$sha
  return 0
}

log "watching $TRAIN_REPO/checkpoints/*_latest.bin every ${POLL_SECS}s (mix from $PLAY_REPO)"
while :; do
  latest=$(newest_latest)
  if [ -n "$latest" ]; then
    score_snapshot "$latest" || true
  fi
  [ "$ONCE" = 1 ] && break
  sleep "$POLL_SECS"
done
log "done"
