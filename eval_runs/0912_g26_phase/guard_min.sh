#!/usr/bin/env bash
# Minimal behavioral pair: accepted ep33 vs rejected ep38 (loss 7e-5 vs 2e-6).
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
O=eval_runs/0912_g26_phase; N=checkpoints/ms_g26a
echo "=== STAND k=4 ep33 vs ep38 ($(date +%H:%M))"
GATE_REACTION=4 EPOCHS="33 38" bash scripts/gate_sweep.sh $N $O/stand_guard 2>&1 | grep -aE "^ep[0-9]+:|ARGMAX"
echo "=== CPU k=4 ep33 vs ep38 ($(date +%H:%M))"
bash eval_runs/0911_g24_dagger/gate_cpu.sh $N $O/cpu_guard "33 38" 2>&1 | grep -aE "^ep[0-9]+ "
echo "=== DONE $(date +%H:%M)"
