#!/usr/bin/env bash
# New gate (2026-09-11): shines/min vs the level-1 CPU on the async runner at
# the local rung (d3, id 2), 2 runs x 90s per snapshot. Usage: gate_cpu.sh <prefix> <outdir> "<epochs>"
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
PREFIX=$1; OUT=$2; EPOCHS=$3
for ep in $EPOCHS; do
  snap="${PREFIX}_ep${ep}.bin"; [ -f "$snap" ] || continue
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh "$snap" "$OUT/ep$ep" --runs 2 --seconds 90 --dummy cpu --runner async --temperature 1.0 -- --frame-delay 3 --delay-id-override 2 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "^\[[0-9:]+\] r[12] " | sed "s/^/ep$ep /" | tee -a "$OUT/cpu_table.txt"
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
