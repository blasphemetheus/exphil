#!/usr/bin/env bash
# Re-pin after the probe rides the async decision hop and sync pipeline = 1:
#  async k4 (fd 3): probe must read 5 == expected; chains (rung law)
#  sync  k4 (fd 4): probe 5 == expected; ep57 chains LESS here (sync anomaly)
#  sync  k3 (fd 3): probe 4 == expected; ep57 chains 427-class (one faster than labels)
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
O=eval_runs/0912_sync_rung
for cell in "async 4" "sync 4" "sync 3"; do
  set -- $cell; runner=$1; k=$2
  echo "=== $runner --reaction-delay $k ==="
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g23a_ep57.bin "$O/smoke3_${runner}_k${k}" --runs 1 --seconds 60 --dummy stand --runner $runner --temperature 1.0 -- --reaction-delay $k --allow-latency-mismatch --headless --emulation-speed 0 --blocking-input --slippi-port 51442 > "$O/smoke3_${runner}_k${k}.out" 2>&1
  grep -ahE "measured" "$O/smoke3_${runner}_k${k}.out" "$O/smoke3_${runner}_k${k}/"*.log 2>/dev/null | sed 's/^.*\] //' | sort -u | head -2
  grep -aE "^\[[0-9:]+\] r1 " "$O/smoke3_${runner}_k${k}.out" | tail -1
  sleep 2; pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
echo "=== DONE ==="
