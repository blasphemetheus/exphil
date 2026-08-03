#!/usr/bin/env bash
# Grind 4 (task #16): mdq_ss recipe + the d2-native pool mixed into the
# sources. Prereg:
#   P1: d2 >= 280/min chains >= 50 (2x+) with d3/d4 within 20% of the
#       records => source-distribution hypothesis CONFIRMED; per-rung
#       collection becomes standard.
#   P2: d2 unchanged => both cheap hypotheses dead; the inversion goes
#       to interp (boundary maps per rung, task-#5-adjacent).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.7
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== GRIND4 d2mix TRAIN $(date +%H:%M:%S)"
mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g4_d2mix.bin \
  2>&1 | grep -aE "Converged|diverged|exported|error|\*\*" | tail -4
[ -f checkpoints/ms_g4_d2mix.bin ] || { echo "=== GRIND4 FAILED" >&2; exit 1; }
for d in 2 3 4; do
  echo "=== GRIND4 d2mix EVAL d$d $(date +%H:%M:%S)"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    checkpoints/ms_g4_d2mix.bin \
    eval_runs/0802_g4_d2mix_d$d --runs 1 --dummy stand --runner sync \
    -- --frame-delay $d --headless --emulation-speed 0 --blocking-input --slippi-port 51442
done
echo "=== GRIND4 done $(date +%H:%M:%S)"
