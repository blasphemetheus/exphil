#!/usr/bin/env bash
# NaN-robust 2e-4 test (2026-07-13): r1's exact 42-replay pool, default LR
# 2e-4, with dagger_drill's new NaN-restore (on NaN, restore best params and
# continue with a fresh shuffle; up to 5 restores) + per-epoch shuffle seed.
# Question: does surviving past the NaN point let 2e-4 train deeper and beat
# r1's 27.9% pooled? Benchmarks on this pool: 2e-4 halt-on-NaN = 27.9%
# (export loss 0.225), 1.5e-4 = 21.1% (export 0.106).
set -uo pipefail

if [ -z "${NANROBUST_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export NANROBUST_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="nanrobust" \
    --why="NaN-robust 2e-4 drill test in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

SLIPPI_DIR=$HOME/Slippi
DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
LOG=logs/overnight_nanrobust_20260713.log
TRACE=scripts/trace_tech_chase.exs

if pgrep -x beam.smp >/dev/null; then
  echo "[nanrobust] BEAM ALREADY LIVE — refusing to start (NO-MIX rule)" | tee -a "$LOG"
  exit 1
fi

# r1's exact 42-replay pool (28 originals + 14 clean 2026-07-12 probes)
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
  echo "[nanrobust] probing $tag..." | tee -a "$LOG"
  timeout 900 mix run scripts/play_dolphin_async.exs \
    --policy "$policy" \
    --dolphin "$DOLPHIN" --iso "$ISO" \
    --character mewtwo --stage final_destination \
    --dummy tech_random --dummy-cpu-level 3 \
    --press-threshold 0.45 --release-threshold 0.3 \
    --no-audio \
    --deterministic --on-game-end stop >>"$LOG" 2>&1 || {
    echo "[nanrobust] $tag probe failed or timed out" | tee -a "$LOG"
    return 1
  }
  local new_replay
  new_replay=$(find "$SLIPPI_DIR" -maxdepth 1 -name '*.slp' ${last_replay:+-newer "$last_replay"} -size +500k | sort | tail -1)
  if [ -z "$new_replay" ]; then
    echo "[nanrobust] $tag: no new replay" | tee -a "$LOG"
    return 1
  fi
  echo "[nanrobust] $tag replay: $new_replay" | tee -a "$LOG"
  mix run "$TRACE" "$new_replay" 2>/dev/null | tee -a "$LOG"
}

POLICY=checkpoints/mewtwo_combo_nanrobust_policy.bin
ROLLOUTS=$(IFS=,; echo "${POOL[*]}")
echo "[nanrobust] === training (42-pool, LR 2e-4 default, NaN-restore, max 150 epochs) -> $POLICY ===" | tee -a "$LOG"
mix run scripts/dagger_drill.exs \
  --expert mewtwo_combo \
  --max-epochs 150 \
  --rollouts "$ROLLOUTS" \
  --out "$POLICY" >>"$LOG" 2>&1
if [ ! -f "$POLICY" ]; then
  echo "[nanrobust] training produced no checkpoint — aborting" | tee -a "$LOG"
  exit 1
fi
for p in 1 2 3; do
  probe "$POLICY" "p${p}"
done
echo "[nanrobust] done" | tee -a "$LOG"
