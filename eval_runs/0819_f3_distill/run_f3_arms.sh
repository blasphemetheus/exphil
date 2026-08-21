#!/usr/bin/env bash
# F3 Route A — KL-distillation anchor arms (A2/A3 [+A4 conditional]).
# Prereg v2 in docs/planning/DISTILL_ANCHOR_SPEC.md, AMENDED 2026-08-21
# for the gate-sweep era (that amendment is this header; the spec's v2
# decision rules apply with the bars below).
#
# PRE-REGISTERED (2026-08-21, before the run):
#   Base = the g19 recipe (champion+awbc, 2e-4, 60ep) with snapshot-all
#   + post-run gate-sweep selection (the standing recipe since
#   0820_g19_gatesweep). Teacher = ms_g19_ep4.bin (the human-validated
#   production candidate; per spec v2 the teacher is the base
#   checkpoint itself — NOT ms_g15).
#   A1 (dose baseline) = the g19+g20 sweeps themselves, zero GPU:
#     fox ceiling 437.4/435.4, argmax profiles known.
#   A2 = + distill anchor (w=0.5, tau=1.0), snippet dose unchanged.
#   A3 = anchor + snippet dose DOUBLED (file listed twice).
#   A4 = doubled dose NO anchor — attribution control, run only if A3
#        holds its peak (manual launch).
#   Bars (peak statistics, NOT single gates; ceiling n=2 = 435-437):
#     HOLD  = arm's sweep argmax >= 390 (~0.9x ceiling).
#     A2 fails HOLD -> anchor costs the peak at w=0.5; one w=0.25 arm
#       before abandoning Route A at full scale.
#     A3 HOLDs -> anchor unlocks dose; A4 for attribution; F3
#       graduates.
#     A3 fails, A2 holds -> anchor safe but dose still toxic; escalate
#       per spec (4x arm) only if A2's profile shows gains.
#   Secondary (recorded, NOT gated — profiles vary wildly across peaks,
#   see g19-ep4 vs g20-ep13): argmax mewtwo + d4-id3 gates; KL
#   magnitude per epoch; peak width (#epochs >= 300).
#   NO CROWN from stand numbers (g6 rule).
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_CKPT=checkpoints/ms_g19_ep4.bin
BASE_LR=2.0e-4
BASE_EPOCHS=60
echo "BASE refs: ceiling 437.4/435.4 (g19-ep4/g20-ep13); teacher $BASE_CKPT"
[ -f "$BASE_CKPT" ] || { echo "missing teacher $BASE_CKPT" >&2; exit 1; }

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0819_f3_distill
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'
SNIP="eval_runs/0804_snippets_human_ad2/snippets.frames"

run_arm() {  # $1=name  $2=snippet_glob  $3...=extra flags
  local NAME="$1"; local SNIPGLOB="$2"; shift 2
  echo "=== ${NAME} TRAIN $(date +%H:%M:%S)"
  EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --snippet-frames "$SNIPGLOB" \
    --opp-scramble-frames 12000 \
    --max-epochs "$BASE_EPOCHS" --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
    --lr "$BASE_LR" \
    --target-loss 1.0e-9 \
    --snapshot-all "$@" \
    --out "checkpoints/ms_${NAME}.bin" \
    2>&1 | tee "$OUT/${NAME}_train.log" \
    | grep -aE "AWBC|distill|COLLAPSE|Converged|diverged|exported|error" | tail -8
  [ -f "checkpoints/ms_${NAME}.bin" ] || { echo "=== ${NAME} TRAIN FAILED" >&2; return 1; }

  echo "=== ${NAME} GATE-SWEEP $(date +%H:%M:%S)"
  bash scripts/gate_sweep.sh "checkpoints/ms_${NAME}" "$OUT/${NAME}_sweep" --confirm

  echo "=== ${NAME} argmax d4-id3 (profile record)"
  local BEST
  BEST=$(grep -a "ARGMAX" "$OUT/${NAME}_sweep/sweep_table.txt" | grep -aoE "checkpoints/[^ ]+" || true)
  if [ -n "$BEST" ]; then
    EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
      "$BEST" "$OUT/${NAME}_argmax_d4id3" --runs 1 --dummy stand --runner sync \
      -- --frame-delay 4 --delay-id-override 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
      2>&1 | grep -aE "r1 " | tail -1
  fi
}

# A2: anchor, dose unchanged
run_arm "f3_a2_anchor" "$SNIP" \
  --distill-from "$BASE_CKPT" --distill-weight 0.5 --distill-tau 1.0 || true

# A3: anchor + doubled snippet dose (duplicate entry doubles the lists)
run_arm "f3_a3_anchor2x" "$SNIP,$SNIP" \
  --distill-from "$BASE_CKPT" --distill-weight 0.5 --distill-tau 1.0 || true

echo "=== F3 ARMS DONE $(date +%H:%M:%S). Score sweeps vs HOLD>=390; A4 is a manual launch."