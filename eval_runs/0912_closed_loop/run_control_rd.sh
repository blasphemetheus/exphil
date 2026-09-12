#!/usr/bin/env bash
# Harness rung calibration (2026-09-12): the suite's native decision->
# application latency is 1 frame; the drill trains at d + pipeline offset 2.
# Grid --response-delay {1,2,3} x --delay-id {0,1,2} on the mid-chain
# control (ep57's own 438-chain game). Prediction: the diagonal
# (rd 1/id 0, rd 2/id 1, rd 3/id 2) chains; off-diagonal does not.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
M=scenarios/ms_midchain_control.json; O=eval_runs/0912_closed_loop
for rd in 1 2 3; do
  for id in 0 1 2; do
    echo "=== CONTROL(warm) rd $rd id $id ==="
    mix run scripts/scenario_suite.exs --driver policy --policy checkpoints/ms_g23a_ep57.bin --delay-id $id --response-delay $rd --character fox --manifest $M --runs 2 --temperature 1.0 --out $O/control_rd${rd}_id$id.json --run-dir $O/control_rd${rd}_id$id --quiet 2>&1 | grep -aE "multishine_reentry runs=|error" | tail -2
    pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
  done
done
echo "=== DONE ==="
