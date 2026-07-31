#!/usr/bin/env bash
# d3 rung, DIVERSE-collect round (2026-07-31 ~2:20am). Hypothesis: sync
# determinism starves DAgger — 6 identical metronome trajectories carry one
# path's worth of states (the laptop's smeared async harness accidentally
# provided diversity). Collect stochastically (temperature) from THREE
# policies (farm11_b1 perfect-d2, d3_full2 metronome-d3, dagger3_r3
# proven-d3-chainer), aggregate ALL d3 rollouts + openings, retrain.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12

echo "=== DIVERSE collect $(date +%H:%M:%S)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_farm11_b1.bin \
  eval_runs/d3_div_b1 --runs 3 --dummy stand --runner sync --temperature 0.5 \
  -- --frame-delay 3 --headless --emulation-speed 0
EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_d3_full2.bin \
  eval_runs/d3_div_full2 --runs 3 --dummy stand --runner sync --temperature 0.5 \
  -- --frame-delay 3 --headless --emulation-speed 0
EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_d1_dagger3_r3.bin \
  eval_runs/d3_div_r3 --runs 3 --dummy stand --runner sync --temperature 0.3 \
  -- --frame-delay 3 --headless --emulation-speed 0

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== DIVERSE TRAIN $(date +%H:%M:%S)"
mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --action-delay 3 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --out checkpoints/ms_d3_diverse1.bin \
  2>&1 | grep -aE "opening frames|corrected|Aggregate|Converged|exported|error|\*\*" | tail -20

[ -f checkpoints/ms_d3_diverse1.bin ] || { echo "=== DIVERSE TRAIN FAILED" >&2; exit 1; }
for d in 2 3 4; do
  echo "=== DIVERSE EVAL d$d $(date +%H:%M:%S)"
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_d3_diverse1.bin \
    eval_runs/0731_diverse1_d$d --runs 1 --dummy stand --runner sync \
    -- --frame-delay $d --headless --emulation-speed 0
done
echo "=== DIVERSE done $(date +%H:%M:%S)"
