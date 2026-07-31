#!/usr/bin/env bash
# Grind session 1 (2026-07-31 evening, Bradley playing — VRAM capped 0.25,
# bot-scoped kills, port 51442). Three arms at the d3 rung, leverage order:
#   armA jitter5 — single pool, shift 5±1 per source (R3's accidental-smear
#                  hypothesis, isolated). Prereg: chains >= 10 at d3.
#   armB mdq     — multi-delay {2,3,4} with pipeline-corrected shifts (d+2),
#                  queue-depth 4 + delay-id (everything-aligned combo).
#                  Prereg: beats ctrl1 at every rung.
#   armC jq      — armA's jitter + armB's queue (if both help, they stack).
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
  echo "=== GRIND1 $name TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    "$@" \
    --out checkpoints/ms_g1_$name.bin \
    2>&1 | grep -aE "jitter draws|Converged|diverged|exported|error|\*\*" | tail -4
  [ -f checkpoints/ms_g1_$name.bin ] || { echo "=== GRIND1 $name FAILED" >&2; return 1; }
  for d in 2 3 4; do
    echo "=== GRIND1 $name EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g1_$name.bin \
      eval_runs/0731_g1_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0 --slippi-port 51442
  done
}

run_arm jitter5 --action-delay 3 --pipeline-offset 2 --shift-jitter 1 --with-delay-id
run_arm mdq --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id
run_arm jq --multi-delay "2,3,4" --pipeline-offset 2 --shift-jitter 1 --queue-depth 4 --with-delay-id
echo "=== GRIND1 done $(date +%H:%M:%S)"
