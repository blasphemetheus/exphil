#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12
echo "=== g22a ep59/ep55 at d3 id2 (== g19's old id3 rung) ==="
GATE_DELAY=3 GATE_ID=2 EPOCHS="55 59" bash scripts/gate_sweep.sh checkpoints/ms_g22a eval_runs/0910_g22_converge/sweep_a_d3_id2
echo "=== g22a ep59 at d2 id1 ==="
GATE_DELAY=2 GATE_ID=1 EPOCHS="59" bash scripts/gate_sweep.sh checkpoints/ms_g22a eval_runs/0910_g22_converge/sweep_a_d2_id1
echo "=== per-state ep59 id2 ==="
EXPHIL_GPU_MEMORY_FRACTION=0.25 mix run scripts/probe_ms_state_confidence.exs --policy checkpoints/ms_g22a_ep59.bin --k 24 --limit 500 --delay-id 2 2>&1 | grep -aE "\| \{" | sed "s/^\[[0-9:]*\] //" | cut -c1-70
