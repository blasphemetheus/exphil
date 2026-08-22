#!/usr/bin/env bash
# Post-peak-science chain (2026-08-21): waits for run_queue.sh (P1+P2)
# to exit, then (1) completes A2's truncated sweep (resumable — skips
# ep1-31), (2) runs the A4 attribution control.
#
# A4 PRE-REGISTERED (per DISTILL_ANCHOR_SPEC v2 + run_f3_arms.sh
# amendment): doubled snippet dose, NO anchor. Triggered by A3's HOLD.
#   Reads: A4 argmax < 390 while A3 held -> the ANCHOR owns the dose
#   tolerance (attribution confirmed; F3 fully graduates). A4 also
#   holds -> the doubled dose was never toxic on this recipe; the
#   anchor keeps its trajectory-stabilization (46/60 peak epochs) and
#   profile findings, but the dose claim is not attributable to it.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "=== POSTQUEUE: waiting for peak-science queue..."
while pgrep -f "[r]un_queue.sh" >/dev/null; do sleep 60; done
echo "=== POSTQUEUE: queue done at $(date +%H:%M:%S)"

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0819_f3_distill

echo "=== A2 SWEEP COMPLETION $(date +%H:%M:%S)"
bash scripts/gate_sweep.sh checkpoints/ms_f3_a2_anchor "$OUT/f3_a2_anchor_sweep" --confirm \
  || echo "=== A2 sweep completion FAILED"

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'
SNIP="eval_runs/0804_snippets_human_ad2/snippets.frames"

echo "=== A4 TRAIN $(date +%H:%M:%S) (dose 2x, NO anchor — attribution control)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "$SNIP,$SNIP" \
  --opp-scramble-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
  --lr 2.0e-4 \
  --target-loss 1.0e-9 \
  --snapshot-all \
  --out checkpoints/ms_f3_a4_dose2x.bin \
  2>&1 | tee "$OUT/f3_a4_dose2x_train.log" \
  | grep -aE "AWBC|COLLAPSE|Converged|diverged|exported|error" | tail -6

if [ -f checkpoints/ms_f3_a4_dose2x.bin ]; then
  echo "=== A4 GATE-SWEEP $(date +%H:%M:%S)"
  bash scripts/gate_sweep.sh checkpoints/ms_f3_a4_dose2x "$OUT/f3_a4_dose2x_sweep" --confirm
else
  echo "=== A4 TRAIN FAILED"
fi
echo "=== POSTQUEUE DONE $(date +%H:%M:%S)"