#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12
echo "=== g23b d3 id2 ==="
GATE_DELAY=3 GATE_ID=2 EPOCHS="48 53 55 56 57 58 59 60" bash scripts/gate_sweep.sh checkpoints/ms_g23b eval_runs/0910_g23_consistent/sweep_b_d3_id2
echo "=== per-state ep57 id2 ==="
EXPHIL_GPU_MEMORY_FRACTION=0.25 mix run scripts/probe_ms_state_confidence.exs --policy checkpoints/ms_g23b_ep57.bin --k 24 --limit 500 --delay-id 2 2>&1 | grep -aE "\| \{" | sed "s/^\[[0-9:]*\] //" | cut -c1-70
