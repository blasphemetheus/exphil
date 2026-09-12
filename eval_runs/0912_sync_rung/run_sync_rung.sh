#!/usr/bin/env bash
# Sync-runner latency pin (2026-09-12, INVARIANTS item 12). Model A (sync L =
# fd+1, from "sync d3 == async d2" 07-28): fd4/id2 + fd3/id1 aligned, fd3/id2 +
# fd2/id1 one frame FAST -> chains break. Model B (sync L = fd+2): the reverse
# pairs align. ep57, stand dummy, T=1.0, 2 x 60s per cell.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
O=eval_runs/0912_sync_rung
for pair in "3 2" "4 2" "3 1" "2 1"; do
  set -- $pair; fd=$1; id=$2
  echo "=== sync fd $fd id $id ==="
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh checkpoints/ms_g23a_ep57.bin "$O/fd${fd}_id$id" --runs 2 --seconds 60 --dummy stand --runner sync --temperature 1.0 -- --frame-delay $fd --delay-id-override $id --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "^\[[0-9:]+\] r[12] " | tee -a "$O/table.txt"
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
echo "=== DONE ==="
