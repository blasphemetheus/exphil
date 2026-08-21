#!/usr/bin/env bash
# g20 = EXACT g19 REPLICATE (B1 of the 08-20 direction menu): same
# recipe, snapshot-all, full 60 epochs, post-run gate-sweep. n=2 for
# the peak-structure question.
#
# PRE-REGISTERED (2026-08-20 evening, before the run):
#   g19 reference (eval_runs/0820_g19_gatesweep): wide peak ep3-8 (all
#   >=380, argmax ep4 437.4 c438), 1-epoch revisit spikes to ep42,
#   terminal decay to ~110-150 after ep43. Training nondeterministic,
#   so this is a true replicate.
#   Reads:
#     R-STABLE  peak again lands ep~3-8 with height >=380 -> peak
#               location/height is a RECIPE PROPERTY; gate-sweep+early
#               window is reliable; knob arms can be read against it.
#     R-WANDER  peak exists but elsewhere/lower -> peaks are real but
#               stochastic; every arm needs its own full sweep (no
#               fixed early-stop window); compare arm PEAKS not epochs.
#     R-NONE    no epoch >=304 -> even peak existence is a lottery;
#               n>=3 replicates before any recipe conclusions.
#   Secondary: argmax off-dist mini-battery (d4-id3) for the deploy
#   comparison vs g19-ep4's 388.4.
#   NO CROWN from stand numbers (g6 rule).
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0820_g20_replicate
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== G20 TRAIN $(date +%H:%M:%S) (g19 replicate: champion+awbc, 2e-4, 60ep, snapshot-all)"
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
  --out checkpoints/ms_g20.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|COLLAPSE|Converged|diverged|exported|error" | tail -6
[ -f checkpoints/ms_g20.bin ] || { echo "=== G20 TRAIN FAILED" >&2; exit 1; }

echo "=== G20 GATE-SWEEP $(date +%H:%M:%S)"
bash scripts/gate_sweep.sh checkpoints/ms_g20 "$OUT/sweep" --confirm

echo "=== G20 ARGMAX d4-id3 check (g19-ep4 ref 388.4 c389)"
BEST=$(grep -a "ARGMAX" "$OUT/sweep/sweep_table.txt" | grep -aoE "checkpoints/[^ ]+")
if [ -n "$BEST" ]; then
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "$BEST" "$OUT/argmax_d4id3" --runs 1 --dummy stand --runner sync \
    -- --frame-delay 4 --delay-id-override 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    2>&1 | grep -aE "r1 " | tail -1
fi

echo "=== G20 DONE $(date +%H:%M:%S). Score vs prereg reads R-STABLE/R-WANDER/R-NONE."