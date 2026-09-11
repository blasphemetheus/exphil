#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
P=checkpoints/ms_g23a_ep57.bin; O=eval_runs/0910_g23_consistent/confirm_ep57
echo "=== ep57 d3 id2 x3 ==="
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh $P $O/d3_id2 --runs 3 --dummy stand --runner sync --temperature 1.0 -- --frame-delay 3 --delay-id-override 2 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "^\[[0-9:]+\] r[123] "
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== per-state ep57 id2 ==="
EXPHIL_GPU_MEMORY_FRACTION=0.25 mix run scripts/probe_ms_state_confidence.exs --policy $P --k 24 --limit 500 --delay-id 2 2>&1 | grep -aE "\| \{" | sed "s/^\[[0-9:]*\] //" | cut -c1-70
