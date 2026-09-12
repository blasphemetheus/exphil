#!/usr/bin/env bash
# Probe-vs-Slippi clock pin: async fd 3 (the chain-aligned rung for ep57),
# 60s so the .slp finalizes; then compare the probe's reading with the
# Slippi frame the marker was recorded on.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
O=eval_runs/0912_sync_rung
for cell in "async 4 --allow-latency-mismatch" "sync 4 --allow-latency-mismatch"; do
  set -- $cell; runner=$1; k=$2; shift 2; extra="$*"
  echo "=== $runner k$k $extra ==="
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g23a_ep57.bin "$O/pin_${runner}_k${k}" --runs 1 --seconds 60 --dummy stand --runner $runner --temperature 1.0 -- --reaction-delay $k $extra --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "measured|^\[[0-9:]+\] r1 " | tail -3
  sleep 3; pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
echo "=== DONE ==="
