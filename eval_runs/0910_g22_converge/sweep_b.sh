#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12
echo "=== g22b d3 id2 (g19_ep58 rung; bar 127/min c12; g22a 106 c5) ==="
GATE_DELAY=3 GATE_ID=2 EPOCHS="48 53 55 57 59 60" bash scripts/gate_sweep.sh checkpoints/ms_g22b eval_runs/0910_g22_converge/sweep_b_d3_id2
echo "=== g22b d2 id1 ==="
GATE_DELAY=2 GATE_ID=1 EPOCHS="57 60" bash scripts/gate_sweep.sh checkpoints/ms_g22b eval_runs/0910_g22_converge/sweep_b_d2_id1
echo "=== g22b d1 id0 ==="
GATE_DELAY=1 EPOCHS="57 60" bash scripts/gate_sweep.sh checkpoints/ms_g22b eval_runs/0910_g22_converge/sweep_b_d1
echo "=== per-state ep60 id2 ==="
EXPHIL_GPU_MEMORY_FRACTION=0.25 mix run scripts/probe_ms_state_confidence.exs --policy checkpoints/ms_g22b_ep60.bin --k 24 --limit 500 --delay-id 2 2>&1 | grep -aE "\| \{" | sed "s/^\[[0-9:]*\] //" | cut -c1-70
