#!/usr/bin/env bash
# Grind 5 (task #12): extend the delay ladder to d8. mdq_ss champion
# recipe (SS-on-queue, NO jitter per grind-3 rule), multi-delay
# {2,4,6,8} — delay-id one-hot (size 9) fits d<=8 without layout change.
# Prereg:
#   P1: d6/d8 shine >= 200/min with chains >= 20 => self-paced skill
#       ladders to the realistic netplay range; per-rung fight-state
#       ceiling becomes the next question.
#   P2: d6/d8 collapse => the SS-on-queue scheme has a depth limit
#       between 4 and 6; investigate before any remote-friend plan.
#   (d2 expected ~140 regardless — the #18 structural pin.)
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.7
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== GRIND5 ladder8 TRAIN $(date +%H:%M:%S)"
mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,4,6,8" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g5_ladder8.bin \
  2>&1 | grep -aE "Converged|diverged|exported|error|\*\*" | tail -4
[ -f checkpoints/ms_g5_ladder8.bin ] || { echo "=== GRIND5 FAILED" >&2; exit 1; }
for d in 2 4 6 8; do
  echo "=== GRIND5 ladder8 EVAL d$d $(date +%H:%M:%S)"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    checkpoints/ms_g5_ladder8.bin \
    eval_runs/0802_g5_ladder8_d$d --runs 1 --dummy stand --runner sync \
    -- --frame-delay $d --headless --emulation-speed 0 --blocking-input --slippi-port 51442
done
echo "=== GRIND5 done $(date +%H:%M:%S)"
