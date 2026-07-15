#!/usr/bin/env bash
# NaN forensics run (2026-07-13): reproduce the endemic 2e-4 NaN on the
# 42-pool with --nan-forensics instrumentation (per-batch loss checks +
# numeric-vitals trail). Training only — halts with a post-mortem dump at
# the first non-finite loss. No probes; checkpoint goes to scratch.
set -uo pipefail
if [ -z "${NANDIAGW16_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export NANDIAGW16_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="nandiag_w16" \
    --why="window-16 BPTT test in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."
LOG=logs/diagnose_nan_w16_20260713.log
if pgrep -x beam.smp >/dev/null; then
  echo "[nandiag_w16] BEAM ALREADY LIVE — refusing to start (NO-MIX rule)" | tee -a "$LOG"
  exit 1
fi
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
ROLLOUTS=$(IFS=,; echo "${POOL[*]}")
echo "[nandiag_w16] === w16 test: 42-pool, LR 2e-4, bf16, window 16 (BPTT theory) ===" | tee -a "$LOG"
mix run scripts/dagger_drill.exs \
  --expert mewtwo_combo \
  --max-epochs 150 \
  --nan-forensics \
  --window 16 \
  --rollouts "$ROLLOUTS" \
  --out checkpoints/scratch_nandiag_w16_policy.bin >>"$LOG" 2>&1
echo "[nandiag_w16] exited with code $? (2 = forensics NaN dump captured)" | tee -a "$LOG"
