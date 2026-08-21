#!/usr/bin/env bash
# g18 = champion recipe + --awbc at lr 2e-4, --max-epochs 90 (epochs
# knob). Triggered by g17's negative read (eval_runs/0819_g17_lr/
# RESULTS.md): lr is not the recovery lever — quality was monotone in
# LOWER lr, so "best g17 lr" = the 2e-4 baseline; next registered knob
# = epochs. Context: g15r hit max-epochs (60) unconverged; g16
# converged at 0.00127 but 41% below the old-grad champion.
#
# PRE-REGISTERED (written 2026-08-19, before the run):
#   References (fixed stack): g16 (2e-4, 60ep) fox 253.6/min c203,
#   mewtwo 109.8/min c14, loss 0.00127. g15 (OLD grads) 430.4/c426 —
#   aspiration, not a bar.
#   Reads:
#     - fox >=304/min (g16+20%) with mewtwo held (>=100) -> epochs is
#       the lever; 90ep@2e-4 becomes the stage-3 base recipe.
#     - 254-304 (<=+20%) -> unresolved by the 2x-ish discipline; next
#       knob = truncate_bptt/window (handoff queue), epochs read as
#       marginal.
#     - <= g16 -> epochs is NOT the lever either; escalate to the
#       window/bptt knob with both g17+g18 as evidence that the
#       fixed-stack gap is structural, not optimizer-schedule.
#   Secondary watch: does loss keep falling ep60->90 or plateau?
#   (If plateaued by ~ep60, the read is capacity/data, not schedule.)
#   NO CROWN from stand numbers (g6 rule).
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0819_g18_ep90
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== G18 TRAIN $(date +%H:%M:%S) (champion+awbc, lr=2e-4, 90 epochs)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 90 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
  --lr 2.0e-4 \
  --out checkpoints/ms_g18_ep90.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|Converged|diverged|exported|error|\*\*" | tail -6
[ -f checkpoints/ms_g18_ep90.bin ] || { echo "=== G18 TRAIN FAILED" >&2; exit 1; }

echo "=== GATE 1: stand-fox d3 x3 (g16 ref 253.6 c203; bar 304)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g18_ep90.bin "$OUT/stand_fox" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 2: stand-mewtwo d3 (g16 ref 109.8 c14; hold >=100)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g18_ep90.bin "$OUT/stand_mewtwo" \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== G18 DONE $(date +%H:%M:%S). Score vs prereg reads above."