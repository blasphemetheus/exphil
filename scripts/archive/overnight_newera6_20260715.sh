#!/usr/bin/env bash
# NEW-ERA DAgger round 9 (2026-07-15 evening): first training with the
# REPAIRED TEACHER (pathology #4 fix, commit 3e53016):
#   - grounded defaults are side/spacing-aware: opponent behind -> turn
#     toward; wrong spacing -> approach; jump-restart only AT spacing
#     (the old jump default was 31% of all labels @ 36.8% jump)
#   - pool += Game_20260715T194348.slp (r8+debounce game: 24 knockdowns,
#     16.7% conversions — right-kind on-policy data, old task #20)
#   - memory_fraction 0.75 (config/dev.exs): co-tenant eval headroom
#   - PROBES A/B: p1/p2 plain, p3/p4 --jump-debounce 10 — does the fixed
#     teacher obsolete the decode band-aid?
# Stack: nx rebased to main (0.12.1+, clip fix carried), edifice with
# bot-runtime merge. Anti-copycat objectives unchanged (dropout 0.4,
# transition 2.0). Scores comparable to r1-r8 (same dummy/task).
set -uo pipefail

if [ -z "${NEWERA6_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export NEWERA6_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="newera6" \
    --why="new-era DAgger loop in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

SLIPPI_DIR=$HOME/Slippi
DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
LOG=logs/overnight_newera6_20260715.log
TRACE=scripts/trace_tech_chase.exs
ROUNDS=2

if pgrep -x beam.smp >/dev/null; then
  echo "[newera] BEAM ALREADY LIVE — refusing to start (NO-MIX rule)" | tee -a "$LOG"
  exit 1
fi

# Recompiles need the devenv toolchain: launching from a bare shell fails
# ~2 min in with cargo/make :enoent buried in the log (burned two launches
# 2026-07-15). Fail loudly at t=0 instead.
for tool in make cargo; do
  command -v "$tool" >/dev/null || {
    echo "[newera] $tool not on PATH — run me from 'devenv shell'" | tee -a "$LOG"
    exit 1
  }
done

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
  "$HOME/Slippi/Game_20260715T194348.slp"
)

probe() {
  local policy=$1 tag=$2
  shift 2
  local extra_flags=("$@")
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
    --deterministic --on-game-end stop \
    "${extra_flags[@]}" >>"$LOG" 2>&1 || {
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
  local trace_out
  trace_out=$(mix run "$TRACE" "$new_replay" 2>/dev/null | tee -a "$LOG")
  echo "$trace_out"
  # Accumulate this round's conversions (gates the next round)
  local conv
  conv=$(grep -oE 'conversions=[0-9]+' <<<"$trace_out" | head -1 | cut -d= -f2)
  ROUND_CONV=$((ROUND_CONV + ${conv:-0}))
  # Tier-1 report card (jump/shield/deadlock gates) into the log
  mix run scripts/report_card.exs "$new_replay" 2>/dev/null | tee -a "$LOG"
  PROBE_REPLAY=$new_replay
}

for R in $(seq 9 $((8 + ROUNDS))); do
  POLICY=checkpoints/mewtwo_combo_newera_r${R}_policy.bin
  ROLLOUTS=$(IFS=,; echo "${POOL[*]}")
  echo "[newera] === round $R: ${#POOL[@]} replays, repaired teacher, dropout 0.4, transition 2.0 -> $POLICY ===" | tee -a "$LOG"
  mix run scripts/dagger_drill.exs \
    --expert mewtwo_combo \
    --max-epochs 100 \
    --prev-action-dropout 0.4 \
    --transition-weight 2.0 \
    --rollouts "$ROLLOUTS" \
    --out "$POLICY" >>"$LOG" 2>&1
  if [ ! -f "$POLICY" ]; then
    echo "[newera] round $R produced no checkpoint — stopping" | tee -a "$LOG"
    exit 1
  fi
  # A/B probes: plain vs jump-debounce (does the fixed teacher obsolete
  # the decode band-aid?)
  ROUND_CONV=0
  for p in 1 2 3; do
    PROBE_REPLAY=""
    if probe "$POLICY" "r${R}_p${p}_plain" && [ -n "$PROBE_REPLAY" ]; then
      POOL+=("$PROBE_REPLAY")
    fi
  done
  for p in 4 5 6; do
    PROBE_REPLAY=""
    if probe "$POLICY" "r${R}_p${p}_debounce" --jump-debounce 10 && [ -n "$PROBE_REPLAY" ]; then
      POOL+=("$PROBE_REPLAY")
    fi
  done
  # Gate: only continue to another round if this one converts at all —
  # a 0-conversion policy feeds the next round pathological on-policy data
  if [ "$ROUND_CONV" -eq 0 ]; then
    echo "[newera] round $R: 0 conversions across probes — not continuing" | tee -a "$LOG"
    break
  fi
  echo "[newera] round $R total conversions: $ROUND_CONV" | tee -a "$LOG"
done

# Pre-registered measurement: attribution/compression across the eras
# (r9 entry added; missing checkpoints are skipped by the script)
echo "[newera] running interp_p3_case3..." | tee -a "$LOG"
mix run scripts/interp_p3_case3.exs >>"$LOG" 2>&1 || true
# GOTCHAS #58: kill the FINAL probe's Dolphin too — orphans record
# garbage >500k replays that poison size-globbed pools
sleep 3
pgrep -f "/tmp/libmelee_" >/dev/null && pkill -f "/tmp/libmelee_"
echo "[newera] done" | tee -a "$LOG"
