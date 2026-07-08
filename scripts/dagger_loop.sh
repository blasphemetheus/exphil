#!/usr/bin/env bash
# Automated DAgger loop for the multishine probe:
#   retrain on all rollouts so far -> play a solo game (self-terminating,
#   requires the on-game-end fix) -> score the new replay -> add it to the
#   rollout pool -> repeat.
#
# Usage:
#   scripts/dagger_loop.sh [iterations]           # default 3
#
# Env overrides:
#   ROLLOUTS    comma-separated seed replays (default: today's replays >500KB)
#   SLIPPI_DIR  replay directory        (default ~/Slippi)
#   DOLPHIN     dolphin path            (default ~/.local/share/slippi/netplay)
#   ISO         melee ISO               (default ~/isos/melee.iso)
#
# Each iteration takes ~5-10 min (train ~2-4 min + one full solo game).
# Progress lands in logs/dagger_loop_<stamp>.log; the per-game scoreboard is
# printed and appended there.
set -euo pipefail
cd "$(dirname "$0")/.."

ITERS=${1:-3}
SLIPPI_DIR=${SLIPPI_DIR:-$HOME/Slippi}
DOLPHIN=${DOLPHIN:-$HOME/.local/share/slippi/netplay}
ISO=${ISO:-$HOME/isos/melee.iso}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/dagger_loop_${STAMP}.log
mkdir -p logs checkpoints

# Seed rollout pool: today's non-stub replays (stubs from menu re-entries are
# tiny; a real game is multi-MB)
if [ -z "${ROLLOUTS:-}" ]; then
  ROLLOUTS=$(find "$SLIPPI_DIR" -name "Game_$(date +%Y%m%d)T*.slp" -size +500k | sort | paste -sd,)
fi

if [ -z "$ROLLOUTS" ]; then
  echo "No seed rollouts found (set ROLLOUTS=...)" >&2
  exit 1
fi

echo "[dagger_loop] $ITERS iterations, seed pool: $(echo "$ROLLOUTS" | tr ',' '\n' | wc -l) replays" | tee -a "$LOG"

for i in $(seq 1 "$ITERS"); do
  POLICY=checkpoints/multishine_daggerloop_${STAMP}_i${i}_policy.bin
  echo "[dagger_loop] === iteration $i/$ITERS: training -> $POLICY ===" | tee -a "$LOG"

  mix run scripts/dagger_multishine.exs \
    --rollouts "$ROLLOUTS" \
    --out "$POLICY" >>"$LOG" 2>&1

  if [ ! -f "$POLICY" ]; then
    echo "[dagger_loop] training produced no checkpoint (diverged?) — aborting; see $LOG" | tee -a "$LOG"
    exit 1
  fi

  LAST_REPLAY=$(ls -t "$SLIPPI_DIR"/*.slp 2>/dev/null | head -1 || true)
  echo "[dagger_loop] playing solo probe..." | tee -a "$LOG"

  # Self-terminates on game end; timeout is a backstop against a hung session
  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$POLICY" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character fox --stage final_destination \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[dagger_loop] play session failed or timed out — aborting; see $LOG" | tee -a "$LOG"
    exit 1
  }

  NEW_REPLAY=$(find "$SLIPPI_DIR" -name '*.slp' ${LAST_REPLAY:+-newer "$LAST_REPLAY"} -size +500k | sort | tail -1)
  if [ -z "$NEW_REPLAY" ]; then
    echo "[dagger_loop] no new replay found after play — aborting" | tee -a "$LOG"
    exit 1
  fi

  mix run scripts/trace_multishine.exs "$NEW_REPLAY" 2>/dev/null | tee -a "$LOG"
  ROLLOUTS="$ROLLOUTS,$NEW_REPLAY"
done

echo "[dagger_loop] done — final policy: $POLICY" | tee -a "$LOG"
echo "[dagger_loop] scoreboard:" | tee -a "$LOG"
command grep -E "JC=" "$LOG" | tail -"$ITERS"
