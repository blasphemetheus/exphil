#!/usr/bin/env bash
# Second pass: cells the "sync == async (L = fd+2)" model predicts BREAK
# (one frame faster than trained): fd2/id2 and fd1/id1.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
O=eval_runs/0912_sync_rung
for pair in "2 2" "1 1"; do
  set -- $pair; fd=$1; id=$2
  echo "=== sync fd $fd id $id ==="
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh checkpoints/ms_g23a_ep57.bin "$O/fd${fd}_id$id" --runs 2 --seconds 60 --dummy stand --runner sync --temperature 1.0 -- --frame-delay $fd --delay-id-override $id --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "^\[[0-9:]+\] r[12] " | sed "s/^/fd$fd id$id /" | tee -a "$O/table.txt"
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
echo "=== DONE ==="
