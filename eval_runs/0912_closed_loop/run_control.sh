#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
bash scripts/x11_socket_fix.sh
M=scenarios/ms_midchain_control.json; O=eval_runs/0912_closed_loop
echo "=== CONTROL teacher ==="
mix run scripts/scenario_suite.exs --driver teacher --character fox --manifest $M --runs 1 --out $O/control_teacher.json --run-dir $O/control_teacher --quiet 2>&1 | grep -aE "multishine_reentry runs=|error" | tail -2
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
for id in 0 1 2; do
  echo "=== CONTROL policy ep57 id $id ==="
  mix run scripts/scenario_suite.exs --driver policy --policy checkpoints/ms_g23a_ep57.bin --delay-id $id --character fox --manifest $M --runs 2 --temperature 1.0 --out $O/control_policy_id$id.json --run-dir $O/control_policy_id$id --quiet 2>&1 | grep -aE "multishine_reentry runs=|error" | tail -2
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
echo "=== DONE ==="
