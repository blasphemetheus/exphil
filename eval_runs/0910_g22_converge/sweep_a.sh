#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12
echo "=== g22a d3 id3 (g19_ep58's rung; bar 127/min c12) ==="
GATE_DELAY=3 GATE_ID=3 EPOCHS="30 40 48 52 55 56 57 58 59 60" bash scripts/gate_sweep.sh checkpoints/ms_g22a eval_runs/0910_g22_converge/sweep_a_d3
echo "=== g22a d1 derived id ==="
GATE_DELAY=1 EPOCHS="48 55 57 59" bash scripts/gate_sweep.sh checkpoints/ms_g22a eval_runs/0910_g22_converge/sweep_a_d1
echo "=== g22a d1 id1 ==="
GATE_DELAY=1 GATE_ID=1 EPOCHS="48 55 57 59" bash scripts/gate_sweep.sh checkpoints/ms_g22a eval_runs/0910_g22_converge/sweep_a_d1_id1
