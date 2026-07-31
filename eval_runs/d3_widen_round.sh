#!/usr/bin/env bash
# d3 rung, WIDEN round (2026-07-31 ~3am): diverse pool + x-hold-extend 3.
# Boundary maps: R3 (chains at d3) has a soft positive X tail at all afs;
# our deterministic-data arms learn razor windows. Widen the labels on
# purpose (the laptop harness's smear did it by accident).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== WIDEN TRAIN $(date +%H:%M:%S)"
mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" --x-hold-extend 3 \
  --action-delay 3 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --out checkpoints/ms_d3_widen1.bin \
  2>&1 | grep -aE "opening frames|widened|Converged|exported|error|\*\*" | tail -6

[ -f checkpoints/ms_d3_widen1.bin ] || { echo "=== WIDEN TRAIN FAILED" >&2; exit 1; }
for d in 2 3 4; do
  echo "=== WIDEN EVAL d$d $(date +%H:%M:%S)"
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_d3_widen1.bin \
    eval_runs/0731_widen1_d$d --runs 1 --dummy stand --runner sync \
    -- --frame-delay $d --headless --emulation-speed 0
done
echo "=== WIDEN done $(date +%H:%M:%S)"
