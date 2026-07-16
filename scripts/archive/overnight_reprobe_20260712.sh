#!/usr/bin/env bash
# Diagnostic re-probes after the 20260712_035205 combo loop scored 0.0% twice.
# Probe A: known-good 23.1% policy (215741_i1) — environment sanity + variance.
# Probe B: tonight's i2 (loss 0.00195, scored 0/10) — second sample.
# Mirrors dagger_loop.sh probe flags exactly.
set -uo pipefail
cd "$(dirname "$0")/.."

SLIPPI_DIR=$HOME/Slippi
DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
LOG=logs/overnight_reprobe_20260712.log
TRACE=scripts/trace_tech_chase.exs

probe() {
  local policy=$1 tag=$2
  echo "[reprobe] === $tag: $policy ===" | tee -a "$LOG"
  pgrep -f "/tmp/libmelee_" >/dev/null && { pkill -f "/tmp/libmelee_"; sleep 2; }
  local last_replay
  last_replay=$(ls -t "$SLIPPI_DIR"/*.slp 2>/dev/null | head -1 || true)
  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$policy" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 3 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --no-audio \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[reprobe] $tag probe failed or timed out" | tee -a "$LOG"
    return 1
  }
  local new_replay
  new_replay=$(find "$SLIPPI_DIR" -name '*.slp' ${last_replay:+-newer "$last_replay"} -size +500k | sort | tail -1)
  if [ -z "$new_replay" ]; then
    echo "[reprobe] $tag: no new replay found" | tee -a "$LOG"
    return 1
  fi
  echo "[reprobe] $tag replay: $new_replay" | tee -a "$LOG"
  mix run "$TRACE" "$new_replay" 2>/dev/null | tee -a "$LOG"
}

probe checkpoints/mewtwo_combo_daggerloop_20260711_215741_i1_policy.bin "A(known-good-23.1pct)"
probe checkpoints/mewtwo_combo_daggerloop_20260712_035205_i2_policy.bin "B(tonight-i2)"
echo "[reprobe] done" | tee -a "$LOG"
