#!/usr/bin/env bash
# Checkpoint ladder v1 (task #19): round-robin policy-vs-policy on
# MAINLINE headless Dolphin, scored into Elo standings.
#
#   CHECKPOINTS="checkpoints/a_policy.bin checkpoints/b_policy.bin ..." \
#     bash scripts/checkpoint_ladder.sh
#
# Env:
#   CHECKPOINTS      space-separated policy .bin paths (required, >= 2)
#   GAMES_PER_PAIR   games per unordered pair, port order alternates (2)
#   LADDER_DIR       output root (probes/ladder/<timestamp>)
#   LADDER_DOLPHIN   mainline nix-ld wrapper (GOTCHAS #66-RESOLUTION)
#   GAME_TIMEOUT     seconds per game before the arm is killed (720)
#
# Engine: MAINLINE from day one — the ladder is a NEW eval track with no
# ExiAI history to stay comparable with, so it doubles as the mainline
# validation mileage while legacy probes migrate at a round boundary.
# Expect one benign xcb window per game (mainline bundles only the xcb
# Qt platform — cosmetic).
#
# Matches run SEQUENTIALLY: each game already runs TWO inference paths
# (port-2 policy infers inline in the frame loop — PolicyOpponent), and
# headless games pace to the loop (#69), so matches run sub-real-time.
# Parallel matches are a later scaling lever (compose with probe_fanout's
# memory-fraction guards).
#
# NO-MIX (#67): refuses to start while a training loop or probe BEAM is
# alive — ladder games need the GPU and the mix runs here use
# --no-compile against the existing build. PRECOMPILE FIRST (mix compile
# before launching, never during).

set -u

CHECKPOINTS=${CHECKPOINTS:?space-separated policy .bin paths}
GAMES_PER_PAIR=${GAMES_PER_PAIR:-2}
LADDER_DIR=${LADDER_DIR:-probes/ladder/$(date +%Y%m%d_%H%M)}
LADDER_DOLPHIN=${LADDER_DOLPHIN:-$HOME/.local/share/slippi/mainline/dolphin-emu-mainline}
ISO=$HOME/isos/melee.iso
GAME_TIMEOUT=${GAME_TIMEOUT:-720}
BASE_PORT=51600
LOG="$LADDER_DIR/ladder.log"

if pgrep -f "overnight_newera8|dagger_drill" >/dev/null 2>&1; then
  echo "[ladder] REFUSING: training/probe loop is alive (#67 + GPU contention)"
  exit 1
fi

[ -x "$LADDER_DOLPHIN" ] || {
  echo "[ladder] mainline wrapper missing: $LADDER_DOLPHIN (GOTCHAS #64/#66-RESOLUTION)"
  exit 1
}
# Same store-path GC guard as overnight_newera8 (#64)
for sp in $(grep -oE '/nix/store/[^/ :"]+' "$LADDER_DOLPHIN" | sort -u); do
  [ -e "$sp" ] || {
    echo "[ladder] wrapper store path GC'd: $sp — re-run the nix build from GOTCHAS #64"
    exit 1
  }
done

read -ra CKPTS <<<"$CHECKPOINTS"
n=${#CKPTS[@]}
[ "$n" -ge 2 ] || {
  echo "[ladder] need >= 2 checkpoints, got $n"
  exit 1
}
for c in "${CKPTS[@]}"; do
  [ -f "$c" ] || {
    echo "[ladder] missing checkpoint: $c"
    exit 1
  }
done

mkdir -p "$LADDER_DIR"
total=$((n * (n - 1) / 2 * GAMES_PER_PAIR))
echo "[ladder] $n checkpoints, $GAMES_PER_PAIR games/pair -> $total games, out: $LADDER_DIR" | tee -a "$LOG"

match=0
for ((i = 0; i < n; i++)); do
  for ((j = i + 1; j < n; j++)); do
    for ((g = 0; g < GAMES_PER_PAIR; g++)); do
      # Alternate port order across the pair's games — port asymmetries
      # exist (e.g. mainline respawn nuance), don't bake them into ratings
      if ((g % 2 == 0)); then
        A=${CKPTS[i]} B=${CKPTS[j]}
      else
        A=${CKPTS[j]} B=${CKPTS[i]}
      fi

      match=$((match + 1))
      an=$(basename "$A" .bin)
      bn=$(basename "$B" .bin)
      dir=$(printf '%s/m%03d_%s__vs__%s' "$LADDER_DIR" "$match" "$an" "$bn")
      mkdir -p "$dir"

      echo "[ladder] match $match/$total: $an (P1) vs $bn (P2)" | tee -a "$LOG"

      timeout "$GAME_TIMEOUT" \
        mix run --no-compile --no-deps-check scripts/play_dolphin_async.exs \
        --policy "$A" --p2-policy "$B" \
        --dolphin "$LADDER_DOLPHIN" --iso "$ISO" \
        --character mewtwo --dummy-character mewtwo \
        --stage final_destination \
        --no-audio --on-game-end stop --headless \
        --slippi-port $((BASE_PORT + match % 50)) \
        --replay-dir "$dir" \
        >>"$LOG" 2>&1

      rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "[ladder] match $match exited rc=$rc (timeout/crash) — scorer will skip if no replay" | tee -a "$LOG"
      fi
    done
  done
done

echo "[ladder] scoring..." | tee -a "$LOG"
mix run --no-compile --no-deps-check scripts/ladder_score.exs "$LADDER_DIR" | tee -a "$LOG"
