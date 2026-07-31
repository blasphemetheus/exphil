#!/usr/bin/env bash
# d3 rung, action-delay sweep (2026-07-31 ~4:20am). Ten arms all lock at d2;
# the un-swept knob is --action-delay itself on the diverse pool (ad3 was
# inferred; collect cal peaks at -4). Arms: ad2 (R3's exact value) and ad4.
# Epoch-capped at 60 (all successful arms converged <40).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

for ad in 2 4; do
  echo "=== ADSWEEP ad$ad TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --action-delay $ad --max-epochs 60 \
    --prev-action-dropout 0.6 --transition-weight 2.0 \
    --out checkpoints/ms_d3_ad$ad.bin \
    2>&1 | grep -aE "Converged|diverged|exported|error|\*\*" | tail -3
  [ -f checkpoints/ms_d3_ad$ad.bin ] || { echo "=== ADSWEEP ad$ad FAILED" >&2; continue; }
  for d in 2 3 4; do
    echo "=== ADSWEEP ad$ad EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_d3_ad$ad.bin \
      eval_runs/0731_ad${ad}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0
  done
done
echo "=== ADSWEEP done $(date +%H:%M:%S)"
