#!/usr/bin/env bash
# Grind 7 (task #20): the {3,4} arm completes the 2-rung matrix
# ({2,3} d2-record / {2,4} d2-pin+d4-zero / {2,3,4} d4-record).
# Prereg (small-pools-win-their-low-rungs): strong d3, d4 better than
# {2,3}'s 71-at-honest-id; d2 evaluated at id override 3 is the
# bonus probe. All rungs eval at their honest ids; interpret d2 with
# the untrained-id rule in mind (id 2 IS trained here? NO — {3,4} pool
# trains ids 3,4 only; d2 honest id 2 is UNTRAINED -> expect collapse;
# the informative cells are d3/d4).
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
  echo "=== GRIND7 $name TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "$delays" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 \
    --out checkpoints/ms_g9_$name.bin \
    2>&1 | grep -aE "Converged|diverged|exported|error|\*\*" | tail -4
  [ -f checkpoints/ms_g9_$name.bin ] || { echo "=== GRIND7 $name FAILED" >&2; return 1; }
  for d in "$@"; do
    echo "=== GRIND7 $name EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
      checkpoints/ms_g9_$name.bin \
      eval_runs/0803_g9_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0 --blocking-input --slippi-port 51442
  done
}

run_arm sp34 "3,4" 2 3 4

echo "=== GRIND7 done $(date +%H:%M:%S)"
