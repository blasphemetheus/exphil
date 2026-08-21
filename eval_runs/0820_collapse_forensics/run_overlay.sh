#!/usr/bin/env bash
# Cross-stack per-batch loss overlay (2026-08-20): 2 epochs from the
# ep50 warm start, per-batch losses appended to the --out .batchloss
# file. Run once per nx stack; the per-epoch-seeded shuffle should give
# identical batch order, so the two series overlay until the first
# corrupted step (if corruption is cross-step, e.g. buffer aliasing).
# Usage: run_overlay.sh <label>   (label names the .batchloss file)
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL="${1:?usage: run_overlay.sh <label>}"

export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0820_collapse_forensics
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== OVERLAY ${LABEL} START $(date +%H:%M:%S) nx=[$(git -C $HOME/git/nx log --oneline -1 | head -1)]"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 2 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 1 \
  --lr 2.0e-4 \
  --init-from checkpoints/ms_g15r2_latest.bin \
  --collapse-forensics \
  --out "checkpoints/ms_overlay_${LABEL}.bin" \
  2>&1 | tee "$OUT/overlay_${LABEL}.log" \
  | grep -aE "SCENE|Warm-started|epoch|error" | tail -6 || true
echo "=== OVERLAY ${LABEL} DONE $(date +%H:%M:%S) — series: checkpoints/ms_overlay_${LABEL}.bin.batchloss"