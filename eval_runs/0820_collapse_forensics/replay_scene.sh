#!/usr/bin/env bash
# Replay a captured scene under the CURRENT nx checkout.
# Usage: replay_scene.sh <prefix>   e.g. checkpoints/collapse_scene/spike1
# Run once on the wild stack (f843aa1a) and once on the calm one
# (a7497612 or bisect/a74-plus-1814 for the single-commit comparison),
# then compare the REPLAY lines. Same recipe flags as the forensics run
# so the trainer/model shape matches the scene checkpoint.
set -euo pipefail
cd "$(dirname "$0")/../.."
PREFIX="${1:?usage: replay_scene.sh <scene-prefix>}"

export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 40 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 1 \
  --lr 2.0e-4 \
  --replay-scene "$PREFIX" \
  --out checkpoints/ms_replay_scratch.bin \
  2>&1 | grep -aE "REPLAY|error|Error|raise"