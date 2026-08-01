#!/usr/bin/env bash
# Grind session 2 (2026-07-31 night, machine free — no pinning/nice).
# One arm: SS-on-queue, the principled exposure-bias fix for the queue
# channel (built tonight; EXPOSURE_BIAS.md item 6 / HANDOFF_2026-07-31).
#   arm mdq_ss — grind-1's mdq recipe (multi-delay {2,3,4}, pipeline-offset
#                2, queue-depth 4, delay-id) + --scheduled-sampling 0.5
#                ramped over 10 epochs. All 4 queue slots self-sampled
#                (~2x step time vs mdq's ~30 min train).
#   Prereg: beats mdq at every rung; the queue channel's d4 c9 survives.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

run_arm () { # name, train flags...
  local name=$1; shift
  echo "=== GRIND2 $name TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    "$@" \
    --out checkpoints/ms_g2_$name.bin \
    2>&1 | grep -aE "jitter draws|Converged|diverged|exported|error|\*\*" | tail -4
  [ -f checkpoints/ms_g2_$name.bin ] || { echo "=== GRIND2 $name FAILED" >&2; return 1; }
  for d in 2 3 4; do
    echo "=== GRIND2 $name EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g2_$name.bin \
      eval_runs/0731_g2_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --slippi-port 51442
  done
}

run_arm mdq_ss --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10
echo "=== GRIND2 done $(date +%H:%M:%S)"
