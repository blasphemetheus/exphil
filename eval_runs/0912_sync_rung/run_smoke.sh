#!/usr/bin/env bash
# Wiring smoke (INVARIANTS item 12): no rung flags anywhere — the suite must
# derive --response-delay 2 for ep57's smallest id (0) and chain the control;
# the sync runner at --frame-delay 3 must derive id 2 (log line) and chain.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
O=eval_runs/0912_sync_rung
echo "=== suite, no --delay-id / --response-delay ==="
mix run scripts/scenario_suite.exs --driver policy --policy checkpoints/ms_g23a_ep57.bin --character fox --manifest scenarios/ms_midchain_control.json --runs 1 --temperature 1.0 --out $O/smoke_suite.json --run-dir $O/smoke_suite --quiet 2>&1 | grep -aE "response-delay|delay_id|multishine_reentry runs=|error" | tail -4
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== sync runner fd 3, no override ==="
DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless" EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g23a_ep57.bin $O/smoke_sync --runs 1 --seconds 45 --dummy stand --runner sync --temperature 1.0 -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "delay_id|^\[[0-9:]+\] r1 " | tail -3
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== DONE ==="
