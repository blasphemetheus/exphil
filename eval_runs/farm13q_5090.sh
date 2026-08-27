#!/usr/bin/env bash
# FARM 13Q (2026-07-31): queue-as-input prereg. One policy trained on the
# tri-delay pool (d2/d3/d4, fixture + diverse rollouts + openings) with
# --queue-depth 4 --with-delay-id, evaled at sync d2/d3/d4.
# PREREG: locks at ALL THREE rungs (>=150/min, chains >=50 at each) —
# the property Direct's measured delay smear demands. Control arm: same
# pool WITHOUT queue/delay-id (multi-delay alone), prereg: d2-locked or
# muddled everywhere (delay unobservable -> average policy).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

train_one () { # name, extra flags...
  local name=$1; shift
  echo "=== FARM13Q $name TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --multi-delay "2,3,4" --max-epochs 120 \
    --prev-action-dropout 0.6 --transition-weight 2.0 \
    "$@" \
    --out checkpoints/ms_q_$name.bin \
    2>&1 | grep -aE "opening frames|Converged|diverged|exported|error|\*\*" | tail -4
  [ -f checkpoints/ms_q_$name.bin ] || { echo "=== FARM13Q $name FAILED" >&2; return 1; }
  for d in 2 3 4; do
    echo "=== FARM13Q $name EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_q_$name.bin \
      eval_runs/0731_q_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0
  done
}

train_one queue1 --queue-depth 4 --with-delay-id
# ctrl1 done 2026-07-31 13:26 (d2 239 c24, d3 99 c7, d4 95 c12 — generalist, nowhere locked)
echo "=== FARM13Q done $(date +%H:%M:%S)"
