#!/usr/bin/env bash
# Grind 6 (task #18): the rung-spacing causal A/B.
# Context: four spacing-1 recipes ({2,3,4} = shifts {4,5,6}) pinned d2 at
# ~140 c<=6; grind-5's spacing-2 pool ({2,4,6,8}) broke it (205.8 c73).
# Hypothesis: ADJACENT-shift rungs interfere (1-frame-apart label sets
# collide on shared states); spacing 2 dodges it.
# Arms (champion recipe otherwise; NO jitter per grind-3 rule):
#   armA multi-delay "2,4"  (spacing 2, minimal pool)
#   armB multi-delay "2,3"  (spacing 1, minimal pool)
# Prereg:
#   P1: armA d2 >= 200 AND armB d2 ~ 140 => interference CONFIRMED
#       causally; champion recipe moves to spacing-2 rungs.
#   P2: both ~140 => spacing exonerated; grind-5's unpin came from pool
#       SIZE/composition (4 rungs); next suspect = rung count.
#   P3: both >= 200 => pool MINIMALISM unpins d2 (3-rung pools were the
#       problem); spacing irrelevant.
# Both arms also eval d4 (armA's second rung / armB off-rung control).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.7
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

run_arm () { # name, delays, eval rungs...
  local name=$1 delays=$2; shift 2
  echo "=== GRIND6 $name TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "$delays" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 \
    --out checkpoints/ms_g6_$name.bin \
    2>&1 | grep -aE "Converged|diverged|exported|error|\*\*" | tail -4
  [ -f checkpoints/ms_g6_$name.bin ] || { echo "=== GRIND6 $name FAILED" >&2; return 1; }
  for d in "$@"; do
    echo "=== GRIND6 $name EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
      checkpoints/ms_g6_$name.bin \
      eval_runs/0803_g6_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0 --blocking-input --slippi-port 51442
  done
}

run_arm sp2 "2,4" 2 4
run_arm sp1 "2,3" 2 3 4
echo "=== GRIND6 done $(date +%H:%M:%S)"
