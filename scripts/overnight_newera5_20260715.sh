#!/usr/bin/env bash
# NEW-ERA DAgger loop (2026-07-14): first training of the post-fix age.
#   - FIXED dummy: tech_random actually drives port 2 (cpu_level 0),
#     walks toward the bot, real random techs (GOTCHAS #57 fix)
#   - FIXED loss: logit clamp (min/max form — Nx.clip grads are broken
#     upstream) removes the exp-cliff NaN; no more divergence lottery
#   - ANTI-COPYCAT objectives (#19): --prev-action-dropout 0.4,
#     --transition-weight 2.0
# Seed pool: the 3 new-era games only (old-era data faced a lvl-3 CPU —
# different task, never mix). 2 rounds, 3 scored probes each, probes feed
# the pool. NEW BENCHMARK ERA: scores not comparable to anything earlier.
set -uo pipefail

if [ -z "${NEWERA5_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export NEWERA5_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="newera5" \
    --why="new-era DAgger loop in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

SLIPPI_DIR=$HOME/Slippi
DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
LOG=logs/overnight_newera5_20260715.log
TRACE=scripts/trace_tech_chase.exs
ROUNDS=1

if pgrep -x beam.smp >/dev/null; then
  echo "[newera] BEAM ALREADY LIVE — refusing to start (NO-MIX rule)" | tee -a "$LOG"
  exit 1
fi

POOL=(
  "$HOME/Slippi/Game_20260714T115716.slp"
  "$HOME/Slippi/Game_20260714T142354.slp"
  "$HOME/Slippi/Game_20260714T170156.slp"
  "$HOME/Slippi/Game_20260714T195416.slp"
  "$HOME/Slippi/Game_20260714T200239.slp"
  "$HOME/Slippi/Game_20260714T201103.slp"
  "$HOME/Slippi/Game_20260714T213744.slp"
  "$HOME/Slippi/Game_20260714T214601.slp"
  "$HOME/Slippi/Game_20260714T215417.slp"
  "$HOME/Slippi/Game_20260714T234927.slp"
  "$HOME/Slippi/Game_20260714T235750.slp"
  "$HOME/Slippi/Game_20260715T000607.slp"
  "$HOME/Slippi/Game_20260715T023011.slp"
  "$HOME/Slippi/Game_20260715T023834.slp"
  "$HOME/Slippi/Game_20260715T024658.slp"
  "$HOME/Slippi/Game_20260715T050120.slp"
  "$HOME/Slippi/Game_20260715T050944.slp"
  "$HOME/Slippi/Game_20260715T051807.slp"
  "$HOME/Slippi/Game_20260715T074615.slp"
  "$HOME/Slippi/Game_20260715T075438.slp"
  "$HOME/Slippi/Game_20260715T080302.slp"
  "$HOME/Slippi/Game_20260715T115334.slp"
  "$HOME/Slippi/Game_20260715T120157.slp"
  "$HOME/Slippi/Game_20260715T121021.slp"
)

probe() {
  local policy=$1 tag=$2
  pgrep -f "/tmp/libmelee_" >/dev/null && { pkill -f "/tmp/libmelee_"; sleep 2; }
  local last_replay
  last_replay=$(ls -t "$SLIPPI_DIR"/*.slp 2>/dev/null | head -1 || true)
  echo "[newera] probing $tag..." | tee -a "$LOG"
  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$policy" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 0 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --no-audio \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[newera] $tag probe failed or timed out" | tee -a "$LOG"
    return 1
  }
  local new_replay
  new_replay=$(find "$SLIPPI_DIR" -maxdepth 1 -name '*.slp' ${last_replay:+-newer "$last_replay"} -size +500k | sort | tail -1)
  if [ -z "$new_replay" ]; then
    echo "[newera] $tag: no new replay" | tee -a "$LOG"
    return 1
  fi
  echo "[newera] $tag replay: $new_replay" | tee -a "$LOG"
  mix run "$TRACE" "$new_replay" 2>/dev/null | tee -a "$LOG"
  PROBE_REPLAY=$new_replay
}

for R in $(seq 8 $((7 + ROUNDS))); do
  POLICY=checkpoints/mewtwo_combo_newera_r${R}_policy.bin
  ROLLOUTS=$(IFS=,; echo "${POOL[*]}")
  echo "[newera] === round $R: ${#POOL[@]} replays, dropout 0.4, transition 2.0 -> $POLICY ===" | tee -a "$LOG"
  mix run scripts/dagger_drill.exs \
    --expert mewtwo_combo \
    --max-epochs 150 \
    --prev-action-dropout 0.4 \
    --transition-weight 2.0 \
    --rollouts "$ROLLOUTS" \
    --out "$POLICY" >>"$LOG" 2>&1
  if [ ! -f "$POLICY" ]; then
    echo "[newera] round $R produced no checkpoint — stopping" | tee -a "$LOG"
    exit 1
  fi
  for p in 1 2 3; do
    PROBE_REPLAY=""
    if probe "$POLICY" "r${R}_p${p}" && [ -n "$PROBE_REPLAY" ]; then
      POOL+=("$PROBE_REPLAY")
    fi
  done
done
# GOTCHAS #58: kill the FINAL probe's Dolphin too — orphans record
# garbage >500k replays that poison size-globbed pools
sleep 3
pgrep -f "/tmp/libmelee_" >/dev/null && pkill -f "/tmp/libmelee_"
echo "[newera] done" | tee -a "$LOG"
