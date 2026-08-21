#!/usr/bin/env bash
# g17 = LR retune arms at FULL scale: champion recipe + --awbc (the g16
# lineage) at the two fixture-sweep winners. Fixed-grad stack (GOTCHA
# #98 static unroll; nx `integration`, g16 stack is ancestor,
# lib byte-identical).
#
# PRE-REGISTERED (written 2026-08-19, before the run):
#   Fixture sweep (0817_lr_sweep_fixture): 8e-4 converged @epoch1 to
#   1.6e-4 (fastest+lowest, no divergence); 4e-4 @epoch1 0.00168;
#   baseline 2e-4 @epoch2 8.9e-4. Old lr was tuned in the wrong-grad
#   era; hypothesis: higher lr recovers champion-era levels on the
#   fixed stack.
#   References (fixed stack): g16 (lr 2e-4) fox 253.6/min c203, mewtwo
#   109.8/min c14. g15 (OLD grads) 430.4/c426 — aspiration, not a bar.
#   Reads:
#     - An arm beats g16 G1 by >=20% (>=304/min) with mewtwo held
#       (>=100) -> new stage-3 base; carry into F3/drill programs.
#     - Both arms <= g16 -> lr is not the recovery lever; next knob =
#       epochs (90) then truncate_bptt/window (handoff queue).
#     - Divergence/NaN at 8e-4 full scale -> fixture stability didn't
#       transfer; fall back to 4e-4 read.
#   NO CROWN from stand numbers (g6 rule).
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0819_g17_lr
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

for LR in 8.0e-4 4.0e-4; do
  NAME="g17_lr${LR}"
  echo "=== ${NAME} TRAIN $(date +%H:%M:%S) (champion+awbc, lr=$LR)"
  EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
    --opp-scramble-frames 12000 \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
    --lr "$LR" \
    --out "checkpoints/ms_${NAME}.bin" \
    2>&1 | tee "$OUT/${NAME}_train.log" \
    | grep -aE "AWBC|Converged|diverged|exported|error|\*\*" | tail -6
  [ -f "checkpoints/ms_${NAME}.bin" ] || { echo "=== ${NAME} TRAIN FAILED" >&2; continue; }

  echo "=== ${NAME} GATE 1: stand-fox d3 x3 (g16 ref 253.6 c203)"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "checkpoints/ms_${NAME}.bin" "$OUT/${NAME}_stand_fox" \
    --runs 3 --dummy stand --runner sync \
    -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

  echo "=== ${NAME} GATE 2: stand-mewtwo d3 (g16 ref 109.8 c14)"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "checkpoints/ms_${NAME}.bin" "$OUT/${NAME}_stand_mewtwo" \
    --runs 1 --dummy stand --runner sync \
    -- --frame-delay 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442
done
echo "=== G17 DONE $(date +%H:%M:%S). Score vs prereg reads above."
