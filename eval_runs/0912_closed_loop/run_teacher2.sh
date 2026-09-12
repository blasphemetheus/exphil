#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
bash scripts/x11_socket_fix.sh
M=scenarios/ms_breaks_manifest.json; O=eval_runs/0912_closed_loop
echo "=== driver: teacher (af live->parsed) ==="
mix run scripts/scenario_suite.exs --driver teacher --character fox --manifest $M --only "0,1,2,3,4,5,6,7,8,9,10,11" --runs 1 --out $O/scores_teacher2.json --run-dir $O/runs_teacher2 --quiet 2>&1 | grep -aE "multishine_reentry runs=|Scoreboard|error" | tail -4
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== DONE ==="
