#!/usr/bin/env bash
# Automated DAgger loop for scripted-expert drills:
#   retrain on all rollouts so far -> play a solo game (self-terminating) ->
#   score the new replay -> add it to the rollout pool -> repeat.
#
# Usage:
#   scripts/dagger_loop.sh [iterations]           # default 3
#
# Env overrides:
#   EXPERT      drill expert: multishine | mewtwo_fair   (default multishine)
#   CHARACTER   character the policy plays               (default per expert)
#   TRACE       scoreboard script                        (default per expert)
#   ROLLOUTS    comma-separated seed replays (default: today's replays >500KB
#               — WARNING: that picks up ALL of today's games including ones
#               from other drills; pass ROLLOUTS explicitly on mixed days)
#   DUMMY       scripted port-2 dummy for probe games:
#               stand|shield|jump|walk|cpu|tech_random (default: none)
#   DUMMY_CPU   CPU level when DUMMY=cpu (default 3)
#   PRESS/RELEASE  button hysteresis thresholds for probes (default
#               0.45/0.3 — drill policies are systematically
#               under-confident; the flat 0.5 cut dropped most intended
#               presses. A/B: 43 -> 189 fairs/game, same policy.)
#   SLIPPI_DIR  replay directory        (default ~/Slippi)
#   DOLPHIN     dolphin path            (default ~/.local/share/slippi/netplay)
#   ISO         melee ISO               (default ~/isos/melee.iso)
#
# Each iteration takes ~10-30 min (training grows with the pool + one game).
set -euo pipefail

# Self-protect against hypridle's 40-minute idle suspend, which pauses
# training mid-run (observed: a 2-iteration loop wall-clocked 5 hours
# across suspends). Re-exec under a sleep inhibitor whose lifetime is the
# whole run — a fixed-duration side inhibitor expires mid-loop.
if [ -z "${DAGGER_LOOP_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export DAGGER_LOOP_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="dagger_loop" \
    --why="drill DAgger rounds in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

ITERS=${1:-3}
EXPERT=${EXPERT:-multishine}
SLIPPI_DIR=${SLIPPI_DIR:-$HOME/Slippi}
DOLPHIN=${DOLPHIN:-$HOME/.local/share/slippi/netplay}
ISO=${ISO:-$HOME/isos/melee.iso}

case "$EXPERT" in
  multishine)
    CHARACTER=${CHARACTER:-fox}
    TRACE=${TRACE:-scripts/trace_multishine.exs}
    ;;
  mewtwo_fair)
    CHARACTER=${CHARACTER:-mewtwo}
    TRACE=${TRACE:-scripts/trace_mewtwo_fair.exs}
    ;;
  fox_recovery)
    CHARACTER=${CHARACTER:-fox}
    TRACE=${TRACE:-scripts/trace_sd_postmortem.exs}
    # A solo recovery drill never goes offstage — needs pressure.
    DUMMY=${DUMMY:-cpu}
    ;;
  mewtwo_techchase)
    CHARACTER=${CHARACTER:-mewtwo}
    TRACE=${TRACE:-scripts/trace_tech_chase.exs}
    # Reaction drill needs randomized techs to read
    DUMMY=${DUMMY:-tech_random}
    ;;
  mewtwo_combo)
    CHARACTER=${CHARACTER:-mewtwo}
    TRACE=${TRACE:-scripts/trace_tech_chase.exs}
    DUMMY=${DUMMY:-tech_random}
    ;;
  *)
    echo "Unknown EXPERT=$EXPERT (multishine | mewtwo_fair | fox_recovery)" >&2
    exit 1
    ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/dagger_loop_${EXPERT}_${STAMP}.log
mkdir -p logs checkpoints

if [ -z "${ROLLOUTS:-}" ]; then
  ROLLOUTS=$(find "$SLIPPI_DIR" -name "Game_$(date +%Y%m%d)T*.slp" -size +500k | sort | paste -sd,)
fi

if [ -z "$ROLLOUTS" ]; then
  echo "No seed rollouts found (set ROLLOUTS=...)" >&2
  exit 1
fi

echo "[dagger_loop] expert=$EXPERT char=$CHARACTER, $ITERS iterations, seed pool: $(echo "$ROLLOUTS" | tr ',' '\n' | wc -l) replays" | tee -a "$LOG"

for i in $(seq 1 "$ITERS"); do
  POLICY=checkpoints/${EXPERT}_daggerloop_${STAMP}_i${i}_policy.bin
  echo "[dagger_loop] === iteration $i/$ITERS: training -> $POLICY ===" | tee -a "$LOG"

  mix run scripts/dagger_drill.exs \
    --expert "$EXPERT" \
    ${FIXTURE:+--fixture "$FIXTURE"} \
    --rollouts "$ROLLOUTS" \
    --out "$POLICY" >>"$LOG" 2>&1

  if [ ! -f "$POLICY" ]; then
    echo "[dagger_loop] training produced no checkpoint (diverged?) — aborting; see $LOG" | tee -a "$LOG"
    exit 1
  fi

  # Orphaned Dolphins from crashed sessions hold the slippi port — a new
  # session then connects to the corpse and sits at CHARACTER_SELECT
  # forever (observed 2026-07-11). Clear strays before every probe.
  pgrep -f "libmelee[_]" >/dev/null && { pkill -f "libmelee[_]"; sleep 2; }

  LAST_REPLAY=$(ls -t "$SLIPPI_DIR"/*.slp 2>/dev/null | head -1 || true)
  echo "[dagger_loop] playing solo probe..." | tee -a "$LOG"

  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$POLICY" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character "$CHARACTER" --stage final_destination \
    ${DUMMY:+--dummy "$DUMMY"} \
    ${DUMMY:+--dummy-cpu-level "${DUMMY_CPU:-3}"} \
    --press-threshold "${PRESS:-0.45}" --release-threshold "${RELEASE:-0.3}" \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[dagger_loop] play session failed or timed out — aborting; see $LOG" | tee -a "$LOG"
    exit 1
  }

  NEW_REPLAY=$(find "$SLIPPI_DIR" -name '*.slp' ${LAST_REPLAY:+-newer "$LAST_REPLAY"} -size +500k | sort | tail -1)
  if [ -z "$NEW_REPLAY" ]; then
    echo "[dagger_loop] no new replay found after play — aborting" | tee -a "$LOG"
    exit 1
  fi

  mix run "$TRACE" "$NEW_REPLAY" 2>/dev/null | tee -a "$LOG"
  ROLLOUTS="$ROLLOUTS,$NEW_REPLAY"
done

echo "[dagger_loop] done — final policy: $POLICY" | tee -a "$LOG"
