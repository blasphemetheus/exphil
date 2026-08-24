#!/usr/bin/env bash
# Phase 2 (after the train_delays metadata fix, 635e08e): train r3,
# then gate-sweep ALL THREE replicates (r1/r2 trained fine in phase 1;
# only their sweeps failed on the guard-vs-bad-metadata collision).
# Run inside devenv shell. NO-MIX while this runs.
set -uo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

BASE=eval_runs/0824_g19_replicates
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

if [ ! -f checkpoints/ms_g19r3.bin ]; then
  OUT="$BASE/r3"
  mkdir -p "$OUT"
  echo "=== R3 TRAIN $(date +%H:%M:%S)"
  EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
    --opp-scramble-frames 12000 \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
    --lr 2.0e-4 \
    --target-loss 1.0e-9 \
    --snapshot-all \
    --out "checkpoints/ms_g19r3.bin" \
    > "$OUT/train.log" 2>&1 || echo "=== R3 TRAIN FAILED" >&2
fi

for i in 1 2 3; do
  OUT="$BASE/r$i"
  [ -f "checkpoints/ms_g19r$i.bin" ] || { echo "=== r$i checkpoint missing, skip" >&2; continue; }
  echo "=== R$i GATE-SWEEP $(date +%H:%M:%S)"
  rm -rf "$OUT/sweep"
  bash scripts/gate_sweep.sh "checkpoints/ms_g19r$i" "$OUT/sweep" --confirm \
    > "$OUT/sweep.log" 2>&1
  grep -aE "ARGMAX|argmax" "$OUT/sweep.log" | tail -2
done

echo "=== PHASE 2 DONE $(date +%H:%M:%S)"
