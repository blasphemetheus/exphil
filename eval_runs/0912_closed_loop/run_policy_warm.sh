#!/usr/bin/env bash
# Real policy column of the closed-loop table (2026-09-12): warm prefix
# (Agent.observe) + aligned harness rung (--response-delay = id + 2, from
# run_control_rd.sh). Same 12 break moments the teacher/neutral rows used.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
M=scenarios/ms_breaks_manifest.json; O=eval_runs/0912_closed_loop
for ck in ms_g23a_ep57 ms_g24a_ep55; do
  for pair in "2 0" "3 1"; do
    set -- $pair; rd=$1; id=$2
    echo "=== policy $ck rd $rd id $id (warm, T=1.0, 2 runs) ==="
    mix run scripts/scenario_suite.exs --driver policy --policy checkpoints/$ck.bin --delay-id $id --response-delay $rd --character fox --manifest $M --only "0,1,2,3,4,5,6,7,8,9,10,11" --runs 2 --temperature 1.0 --out $O/scores_warm_${ck}_rd${rd}_id$id.json --run-dir $O/runs_warm_${ck}_rd${rd}_id$id --quiet 2>&1 | grep -aE "multishine_reentry runs=|error" | tail -2
    pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
  done
done
echo "=== DONE ==="
