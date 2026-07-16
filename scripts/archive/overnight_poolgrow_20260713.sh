#!/usr/bin/env bash
# Pool-growth DAgger night (2026-07-13): the handoff's recommended next step.
# Recipe held fixed at the validated ~25-26% config (default LR 2e-4, no
# --target-loss, --max-epochs 150); the ONLY variable is pool size.
#   Round 1: 42 replays = 28 originals + 14 clean on-policy probe games
#            from 2026-07-12 (all real-time 60fps, flat ~/Slippi only —
#            the ~/Slippi/2026-07/ fast-forward poison is excluded).
#   Rounds 2-3: each round's 3 probe replays feed the next round's pool.
# 3 scored probes per round (single-game scores are noise; see handoff).
set -uo pipefail

if [ -z "${POOLGROW_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export POOLGROW_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="poolgrowth" \
    --why="pool-growth DAgger loop in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

SLIPPI_DIR=$HOME/Slippi
DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
LOG=logs/overnight_poolgrow_20260713.log
TRACE=scripts/trace_tech_chase.exs
ROUNDS=3

# NO-MIX guard: abort if any exphil beam or play session is already live.
if pgrep -x beam.smp >/dev/null; then
  echo "[poolgrow] BEAM ALREADY LIVE — refusing to start (NO-MIX rule)" | tee -a "$LOG"
  exit 1
fi

# 28 originals (the 215741 pool) + 14 clean 2026-07-12 probe replays.
POOL=(
  "$HOME/Slippi/Game_20260708T120149.slp"
  "$HOME/Slippi/Game_20260708T121033.slp"
  "$HOME/Slippi/Game_20260708T121650.slp"
  "$HOME/Slippi/Game_20260708T122851.slp"
  "$HOME/Slippi/Game_20260708T123626.slp"
  "$HOME/Slippi/Game_20260708T124312.slp"
  "$HOME/Slippi/Game_20260708T130416.slp"
  "$HOME/Slippi/Game_20260708T132008.slp"
  "$HOME/Slippi/Game_20260708T134241.slp"
  "$HOME/Slippi/Game_20260708T141640.slp"
  "$HOME/Slippi/Game_20260708T142043.slp"
  "$HOME/Slippi/Game_20260708T143817.slp"
  "$HOME/Slippi/Game_20260708T145131.slp"
  "$HOME/Slippi/Game_20260708T150809.slp"
  "$HOME/Slippi/Game_20260708T152012.slp"
  "$HOME/Slippi/Game_20260708T201557.slp"
  "$HOME/Slippi/Game_20260709T131805.slp"
  "$HOME/Slippi/Game_20260709T133915.slp"
  "$HOME/Slippi/Game_20260709T135833.slp"
  "$HOME/Slippi/Game_20260710T183936.slp"
  "$HOME/Slippi/Game_20260710T195825.slp"
  "$HOME/Slippi/Game_20260711T134504.slp"
  "$HOME/Slippi/Game_20260711T141746.slp"
  "$HOME/Slippi/Game_20260711T142553.slp"
  "$HOME/Slippi/Game_20260711T143400.slp"
  "$HOME/Slippi/Game_20260711T144205.slp"
  "$HOME/Slippi/Game_20260711T185758.slp"
  "$HOME/Slippi/Game_20260711T200721.slp"
  "$HOME/Slippi/Game_20260712T034257.slp"
  "$HOME/Slippi/Game_20260712T060102.slp"
  "$HOME/Slippi/Game_20260712T084659.slp"
  "$HOME/Slippi/Game_20260712T085533.slp"
  "$HOME/Slippi/Game_20260712T090234.slp"
  "$HOME/Slippi/Game_20260712T114741.slp"
  "$HOME/Slippi/Game_20260712T115317.slp"
  "$HOME/Slippi/Game_20260712T120034.slp"
  "$HOME/Slippi/Game_20260712T132309.slp"
  "$HOME/Slippi/Game_20260712T133126.slp"
  "$HOME/Slippi/Game_20260712T133942.slp"
  "$HOME/Slippi/Game_20260712T180617.slp"
  "$HOME/Slippi/Game_20260712T181433.slp"
  "$HOME/Slippi/Game_20260712T182250.slp"
)

probe() {
  local policy=$1 tag=$2
  pgrep -f "/tmp/libmelee_" >/dev/null && { pkill -f "/tmp/libmelee_"; sleep 2; }
  local last_replay
  last_replay=$(ls -t "$SLIPPI_DIR"/*.slp 2>/dev/null | head -1 || true)
  echo "[poolgrow] probing $tag..." | tee -a "$LOG"
  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$policy" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 3 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --no-audio \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[poolgrow] $tag probe failed or timed out" | tee -a "$LOG"
    return 1
  }
  local new_replay
  new_replay=$(find "$SLIPPI_DIR" -maxdepth 1 -name '*.slp' ${last_replay:+-newer "$last_replay"} -size +500k | sort | tail -1)
  if [ -z "$new_replay" ]; then
    echo "[poolgrow] $tag: no new replay" | tee -a "$LOG"
    return 1
  fi
  echo "[poolgrow] $tag replay: $new_replay" | tee -a "$LOG"
  mix run "$TRACE" "$new_replay" 2>/dev/null | tee -a "$LOG"
  PROBE_REPLAY=$new_replay
}

for R in $(seq 1 "$ROUNDS"); do
  POLICY=checkpoints/mewtwo_combo_poolgrow_r${R}_policy.bin
  ROLLOUTS=$(IFS=,; echo "${POOL[*]}")
  echo "[poolgrow] === round $R: training on ${#POOL[@]} replays -> $POLICY ===" | tee -a "$LOG"
  mix run scripts/dagger_drill.exs \
    --expert mewtwo_combo \
    --max-epochs 150 \
    --rollouts "$ROLLOUTS" \
    --out "$POLICY" >>"$LOG" 2>&1
  if [ ! -f "$POLICY" ]; then
    echo "[poolgrow] round $R produced no checkpoint — stopping" | tee -a "$LOG"
    exit 1
  fi
  for p in 1 2 3; do
    PROBE_REPLAY=""
    if probe "$POLICY" "r${R}_p${p}" && [ -n "$PROBE_REPLAY" ]; then
      POOL+=("$PROBE_REPLAY")
    fi
  done
done
echo "[poolgrow] done" | tee -a "$LOG"
