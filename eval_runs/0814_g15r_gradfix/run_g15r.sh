#!/usr/bin/env bash
# g15r = THE CHAMPION RECIPE, NO --awbc, on the FIXED stack (nx 0.13.1
# feat/nx-fuzz-errors tip 0e487064 + unroll: :static, GOTCHA #98).
# Purpose: disambiguate g16's G1 fail (eval_runs/0814_g16_awbc/RESULTS.md).
# g16 = champion+awbc+correct-grads was two changes at once; this arm
# isolates the grad-fix.
#
# PRE-REGISTERED (written 2026-08-14 ~12:10, before the run):
#   References: g15 (old grads) stand-fox d3 430.4/min c426, mewtwo
#   80.9/min c2. g16 (correct grads + awbc) 253.6/min c203, mewtwo
#   109.8/min c14.
#   Read:
#     R1 g15r stand-fox NEAR g16 (~<=300/min) -> drop belongs to the
#        GRAD-FIX; AWBC keeps its arms-validated win and g16's
#        generalization gains stand.
#     R2 g15r stand-fox NEAR g15 (>=390/min) -> drop belongs to AWBC;
#        AWBC's offline story weakens materially.
#     R3 intermediate (300-390) -> both contribute; further arms needed.
#   Secondary: mewtwo readout — if g15r mewtwo ALSO jumps (~110), the
#   generalization gain is the grad-fix's, not AWBC's.
#   NO CROWN implications from this run (stand numbers never crown).
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0814_g15r_gradfix
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== G15R TRAIN $(date +%H:%M:%S) (champion recipe, NO awbc, fixed stack)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g15r_gradfix.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|Snippets:|Converged|diverged|exported|error|\*\*" | tail -8
[ -f checkpoints/ms_g15r_gradfix.bin ] || { echo "=== G15R TRAIN FAILED" >&2; exit 1; }

echo "=== GATE 1: stand-fox d3 x3 (discriminator: g15 430.4 / g16 253.6)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g15r_gradfix.bin "$OUT/stand_fox" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 2: stand-MEWTWO d3 (secondary: g15 80.9 / g16 109.8)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g15r_gradfix.bin "$OUT/stand_mewtwo" \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== DONE $(date +%H:%M:%S). Score vs prereg reads R1/R2/R3 above."
