#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
echo "=== g24a CPU gate (async d3 id2, T=1.0, 2x90s) ==="
bash eval_runs/0911_g24_dagger/gate_cpu.sh checkpoints/ms_g24a eval_runs/0911_g24_dagger/cpu_a "48 53 55 57 59 60"
echo "=== reference: g23a ep57 CPU (same gate) ==="
bash eval_runs/0911_g24_dagger/gate_cpu.sh checkpoints/ms_g23a eval_runs/0911_g24_dagger/cpu_ref "57"
