#!/usr/bin/env bash
# Stopping-point A/B after both 035205 rounds scored 0%: hold the pool and
# LR fixed at exactly tonight's i1 recipe, vary ONLY the loss at which we
# stop training. Hypothesis (from 5 rounds of scoreboard history): live
# conversion peaks at moderate loss (0.16 -> 36.4%, 0.08 -> 23.1%) and
# collapses at the loss floor (0.122 post-NaN and 0.002 memorized -> 0%).
#   Run A: --target-loss 0.08   Run B: --target-loss 0.15
set -uo pipefail

# Same self-inhibit pattern as dagger_loop.sh (hypridle suspends at 40min)
if [ -z "${STOPPING_AB_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export STOPPING_AB_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="stopping_ab" \
    --why="drill stopping-point A/B in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

SLIPPI_DIR=$HOME/Slippi
DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
LOG=logs/overnight_stopping_ab_20260712.log
TRACE=scripts/trace_tech_chase.exs

# Tonight's exact 29-replay pool (the 035205 loop's seed: 28 originals +
# the 23.1% salvage-probe replay). Do NOT auto-seed — mixed day.
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
  "$HOME/Slippi/Game_20260711T200721.slp" \
  "$HOME/Slippi/Game_20260712T034257.slp")
ROLLOUTS=${ROLLOUTS%,}

probe() {
  local policy=$1 tag=$2
  pgrep -f "/tmp/libmelee_" >/dev/null && { pkill -f "/tmp/libmelee_"; sleep 2; }
  local last_replay
  last_replay=$(ls -t "$SLIPPI_DIR"/*.slp 2>/dev/null | head -1 || true)
  echo "[stopping_ab] probing $tag..." | tee -a "$LOG"
  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$policy" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 3 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --no-audio \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[stopping_ab] $tag probe failed or timed out" | tee -a "$LOG"
    return 1
  }
  local new_replay
  new_replay=$(find "$SLIPPI_DIR" -name '*.slp' ${last_replay:+-newer "$last_replay"} -size +500k | sort | tail -1)
  if [ -z "$new_replay" ]; then
    echo "[stopping_ab] $tag: no new replay" | tee -a "$LOG"
    return 1
  fi
  echo "[stopping_ab] $tag replay: $new_replay" | tee -a "$LOG"
  mix run "$TRACE" "$new_replay" 2>/dev/null | tee -a "$LOG"
}

for TL in 0.08 0.15; do
  TAG=tl${TL/0./}
  POLICY=checkpoints/mewtwo_combo_stopab_${TAG}_policy.bin
  echo "[stopping_ab] === run $TAG: training to target-loss $TL -> $POLICY ===" | tee -a "$LOG"
  mix run scripts/dagger_drill.exs \
    --lr 1.0e-4 \
    --expert mewtwo_combo \
    --target-loss "$TL" \
    --rollouts "$ROLLOUTS" \
    --out "$POLICY" >>"$LOG" 2>&1
  if [ ! -f "$POLICY" ]; then
    echo "[stopping_ab] $TAG produced no checkpoint — skipping probe" | tee -a "$LOG"
    continue
  fi
  # 3 probes per arm: single-game conversion % is too noisy to judge a
  # policy (observed 2026-07-12: same checkpoint scored 0/10 then 1/6).
  for p in 1 2 3; do
    probe "$POLICY" "${TAG}_p${p}"
  done
done
echo "[stopping_ab] done" | tee -a "$LOG"
