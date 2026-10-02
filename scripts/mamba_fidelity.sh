#!/usr/bin/env bash
# Fidelity scorecard (self-play vs the expert reference) for the two full-size
# Fox Mamba checkpoints (2026-10-02). No training. Waits for queue 6.
set -uo pipefail
until ! systemctl --user is-active -q exphil-coh-queue6; do sleep 30; done
export EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.45 EXPHIL_EXLA_PRECISION=highest
out=eval_runs/1002_fidelity
for spec in "mamba_v1_ep2|checkpoints/fox_mamba_v1_20260925_ep2/model_best_policy.bin" "mamba_v2_prevact|checkpoints/fox_mamba_v2_prevact_20260930/model_best_policy.bin"; do
  name=${spec%%|*}; policy=${spec#*|}
  mix run --no-compile --no-deps-check scripts/fidelity_scorecard.exs --policy "$policy" --label "$name" \
    --envs 32 --frames 3600 --seeds 1001,1002,1003 --stateful-step --out "$out/$name.json" > "$out/$name.log" 2>&1
  grep RESULT "$out/$name.log" | sed 's/^\[[0-9:]*\] //'
  echo "DONE $name"
done
echo "MAMBA FIDELITY DONE"
