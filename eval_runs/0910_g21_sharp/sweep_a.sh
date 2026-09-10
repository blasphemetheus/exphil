#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12
GATE_DELAY=1 EPOCHS="8 16 24 32 40 44 48 52 56 60" bash scripts/gate_sweep.sh checkpoints/ms_g21a eval_runs/0910_g21_sharp/sweep_a_d1
echo "=== g20b ep13 at TRUE id 0 (d1) ==="
GATE_DELAY=1 EPOCHS="13" bash scripts/gate_sweep.sh checkpoints/ms_g20b eval_runs/0910_g21_sharp/g20b_ep13_id0
