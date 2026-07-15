#!/usr/bin/env bash
# Control replication of the known-good 215741_i1 recipe (~26% pooled
# conversion): original 28-replay pool (NO salvage replay), default LR
# (no --lr => 2e-4; inferred — 215741's loss descended ~2x faster than
# tonight's 1e-4 runs and dagger_loop.sh only passes --lr when LR is set),
# --max-epochs 150 (215741's best epoch was 150 @ loss 0.039). 3 probes.
# Outcome reading:
#   ~26% pooled  -> recipe robust; tonight's LR/pool changes caused the 0%s
#   ~10% or less -> run-to-run training variance dominates the ladder
set -uo pipefail

if [ -z "${REPLICATE_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export REPLICATE_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="replicate215741" \
    --why="drill replication control in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

SLIPPI_DIR=$HOME/Slippi
DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
LOG=logs/overnight_replicate215741_20260712.log
TRACE=scripts/trace_tech_chase.exs

# The 215741 pool: 28 replays, WITHOUT Game_20260712T034257.slp
ROLLOUTS=$(printf '%s,' \
  "$HOME/Slippi/Game_20260708T120149.slp" \
  "$HOME/Slippi/Game_20260708T121033.slp" \
  "$HOME/Slippi/Game_20260708T121650.slp" \
  "$HOME/Slippi/Game_20260708T122851.slp" \
  "$HOME/Slippi/Game_20260708T123626.slp" \
  "$HOME/Slippi/Game_20260708T124312.slp" \
  "$HOME/Slippi/Game_20260708T130416.slp" \
  "$HOME/Slippi/Game_20260708T132008.slp" \
  "$HOME/Slippi/Game_20260708T134241.slp" \
  "$HOME/Slippi/Game_20260708T141640.slp" \
  "$HOME/Slippi/Game_20260708T142043.slp" \
  "$HOME/Slippi/Game_20260708T143817.slp" \
  "$HOME/Slippi/Game_20260708T145131.slp" \
  "$HOME/Slippi/Game_20260708T150809.slp" \
  "$HOME/Slippi/Game_20260708T152012.slp" \
  "$HOME/Slippi/Game_20260708T201557.slp" \
  "$HOME/Slippi/Game_20260709T131805.slp" \
  "$HOME/Slippi/Game_20260709T133915.slp" \
  "$HOME/Slippi/Game_20260709T135833.slp" \
  "$HOME/Slippi/Game_20260710T183936.slp" \
  "$HOME/Slippi/Game_20260710T195825.slp" \
  "$HOME/Slippi/Game_20260711T134504.slp" \
  "$HOME/Slippi/Game_20260711T141746.slp" \
  "$HOME/Slippi/Game_20260711T142553.slp" \
  "$HOME/Slippi/Game_20260711T143400.slp" \
  "$HOME/Slippi/Game_20260711T144205.slp" \
  "$HOME/Slippi/Game_20260711T185758.slp" \
  "$HOME/Slippi/Game_20260711T200721.slp")
ROLLOUTS=${ROLLOUTS%,}

probe() {
  local policy=$1 tag=$2
  pgrep -f "/tmp/libmelee_" >/dev/null && { pkill -f "/tmp/libmelee_"; sleep 2; }
  local last_replay
  last_replay=$(ls -t "$SLIPPI_DIR"/*.slp 2>/dev/null | head -1 || true)
  echo "[replicate] probing $tag..." | tee -a "$LOG"
  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$policy" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 3 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --no-audio \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[replicate] $tag probe failed or timed out" | tee -a "$LOG"
    return 1
  }
  local new_replay
  new_replay=$(find "$SLIPPI_DIR" -name '*.slp' ${last_replay:+-newer "$last_replay"} -size +500k | sort | tail -1)
  if [ -z "$new_replay" ]; then
    echo "[replicate] $tag: no new replay" | tee -a "$LOG"
    return 1
  fi
  echo "[replicate] $tag replay: $new_replay" | tee -a "$LOG"
  mix run "$TRACE" "$new_replay" 2>/dev/null | tee -a "$LOG"
}

POLICY=checkpoints/mewtwo_combo_replicate215741_policy.bin
echo "[replicate] === training (28-pool, default LR, max 150 epochs) -> $POLICY ===" | tee -a "$LOG"
mix run scripts/dagger_drill.exs \
  --expert mewtwo_combo \
  --max-epochs 150 \
  --rollouts "$ROLLOUTS" \
  --out "$POLICY" >>"$LOG" 2>&1
if [ ! -f "$POLICY" ]; then
  echo "[replicate] training produced no checkpoint — aborting" | tee -a "$LOG"
  exit 1
fi
for p in 1 2 3; do
  probe "$POLICY" "p${p}"
done
echo "[replicate] done" | tee -a "$LOG"
