#!/usr/bin/env bash
# Parallel probes (task #5): run N probe games CONCURRENTLY against one
# policy, each in its own Dolphin instance with a distinct slippi port and
# a private replay dir (unambiguous attribution — no mtime-scanning a
# shared ~/Slippi).
#
#   ./scripts/parallel_probes.sh POLICY N OUT_DIR [extra play flags...]
#
# Examples:
#   ./scripts/parallel_probes.sh checkpoints/r11_policy.bin 3 probes/r11
#   ./scripts/parallel_probes.sh checkpoints/r11_policy.bin 3 probes/r11 --jump-debounce 10
#
# Modes:
#   default   windowed netplay build (works today; windows appear, GC
#             adapter not needed — controllers are pipes)
#   headless  pass --headless in the extra flags AND point EXPHIL_DOLPHIN
#             at a build supporting Null video (mainline or ExiAI
#             Ishiiruka; the stock Ishiiruka netplay build will refuse)
#
# Env overrides:
#   EXPHIL_DOLPHIN  Dolphin dir  (default ~/.local/share/slippi/netplay)
#   EXPHIL_ISO      Melee ISO    (default ~/isos/melee.iso)
#   PROBE_TIMEOUT   per-probe timeout seconds (default 900)
#   PROBE_MEMFRAC   per-instance EXLA memory fraction (default 0.15 —
#                   inference needs far less than training's 0.75, and
#                   N instances must fit the GPU together)
#
# Per GOTCHAS #60: run under devenv (recompiles need make/cargo).
# Per GOTCHAS #63: cleanup kills by exact PID, never pkill -f self-matches.
set -uo pipefail

POLICY=${1:?usage: parallel_probes.sh POLICY N OUT_DIR [flags...]}
N=${2:?number of parallel probes}
OUT=${3:?output dir}
shift 3
EXTRA_FLAGS=("$@")

DOLPHIN=${EXPHIL_DOLPHIN:-$HOME/.local/share/slippi/netplay}
ISO=${EXPHIL_ISO:-$HOME/isos/melee.iso}
TIMEOUT=${PROBE_TIMEOUT:-900}
MEMFRAC=${PROBE_MEMFRAC:-0.15}
BASE_PORT=51460

cd "$(dirname "$0")/.."
mkdir -p "$OUT"

[ -f "$POLICY" ] || { echo "[probes] no such policy: $POLICY"; exit 1; }

for tool in make cargo; do
  command -v "$tool" >/dev/null || {
    echo "[probes] $tool not on PATH — run me from 'devenv shell' (GOTCHAS #60)"
    exit 1
  }
done

# Compile ONCE before fanning out: concurrent first-compiles race on _build.
echo "[probes] precompiling..."
mix compile 2>&1 | tail -2

PIDS=()
for i in $(seq 1 "$N"); do
  RDIR="$OUT/p$i"
  LOG="$OUT/p$i.log"
  mkdir -p "$RDIR"
  echo "[probes] launching p$i (slippi port $((BASE_PORT + i)), replays -> $RDIR)"
  EXLA_MEMORY_FRACTION=$MEMFRAC timeout "$TIMEOUT" \
    mix run scripts/play_dolphin_async.exs \
    --policy "$POLICY" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 0 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --no-audio \
    --deterministic --on-game-end stop \
    --slippi-port $((BASE_PORT + i)) \
    --replay-dir "$RDIR" \
    "${EXTRA_FLAGS[@]}" >"$LOG" 2>&1 &
  PIDS+=($!)
  # Stagger startup: simultaneous Dolphin boots contend on the ISO read
  # and (windowed mode) the compositor; 5s apart boots cleanly.
  sleep 5
done

FAIL=0
for idx in "${!PIDS[@]}"; do
  p=$((idx + 1))
  if wait "${PIDS[$idx]}"; then
    echo "[probes] p$p finished ok"
  else
    rc=$?
    # The BEAM can segfault during EXLA/CUDA teardown AFTER a clean game
    # (observed 2026-07-16: 2 of 3 headless probes, replays + report
    # cards fine). Judge the probe by its output, not its exit code.
    if compgen -G "$OUT/p$p/*.slp" >/dev/null; then
      echo "[probes] p$p rc=$rc (teardown crash) but produced a replay — ok"
    else
      echo "[probes] p$p exited rc=$rc with NO replay (see $OUT/p$p.log)"
      FAIL=1
    fi
  fi
done

# GOTCHAS #58: the last game's Dolphin can orphan at menus and record
# garbage. Kill any survivors by exact PID via pgrep (no -f self-match).
sleep 2
for pid in $(pgrep -f '/tmp/libmelee_' 2>/dev/null); do
  [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null
done

echo "[probes] replays:"
REPLAYS=()
for i in $(seq 1 "$N"); do
  for slp in "$OUT/p$i"/*.slp; do
    [ -e "$slp" ] || continue
    echo "  p$i: $slp ($(du -h "$slp" | cut -f1))"
    REPLAYS+=("$slp")
  done
done

if [ ${#REPLAYS[@]} -gt 0 ]; then
  echo "[probes] report cards:"
  mix run scripts/report_card.exs "${REPLAYS[@]}"
fi

exit $FAIL
