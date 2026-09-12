#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
bash scripts/x11_socket_fix.sh
M=scenarios/ms_breaks_manifest.json; O=eval_runs/0912_closed_loop
for id in 0 1 2; do
  echo "=== policy ms_g24a_ep55 delay-id $id (T=1.0, 2 runs) ==="
  mix run scripts/scenario_suite.exs --driver policy --policy checkpoints/ms_g24a_ep55.bin --delay-id $id --character fox --manifest $M --only "0,1,2,3,4,5,6,7,8,9,10,11" --runs 2 --temperature 1.0 --out $O/scores_policy_id$id.json --run-dir $O/runs_policy_id$id --quiet 2>&1 | grep -aE "multishine_reentry runs=|error" | tail -2
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
echo "=== DONE ==="
