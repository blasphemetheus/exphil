#!/usr/bin/env bash
# FARM 9 (2026-07-30, 5090): train-consistency gate. Champion recipe
# (farm-6 ARM B: opening + crouch + recovery synth, x-hold-extend 3),
# 3 fresh seeds, each evaled through the TRUSTED harness
# (sync runner, --frame-delay 2, headless, stand dummy, 3 runs).
# Passing bar: every seed trains without loss pathology AND the seed
# spread through the trusted harness is measured (the laptop's z=27 /
# zz=1 lottery re-examined on a harness that doesn't lie).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="${DOLPHIN_DIR:-$HOME/.config/Slippi Launcher/netplay-beta-nixos}"
export XLA_TARGET_EVAL=cuda12

for s in a b c; do
  echo "=== FARM10 seed $s: TRAIN $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  mix run scripts/train_multishine_policy.exs \
    --synth-recovery --synth-crouch --synth-opening \
    --x-hold-extend 3 --probe-basin --scheduled-sampling 0.5 \
    --out checkpoints/ms_farm10_$s.bin \
    2>&1 | grep -aE "epoch|loss|Synthetic|X-hold|exported|Basin|error|\*\*" | tail -20
  echo "=== FARM10 seed $s: EVAL $(date +%H:%M:%S)"
  bash scripts/eval_live_protocol.sh checkpoints/ms_farm10_$s.bin \
    eval_runs/0730_farm10_${s}_syncd2 --runs 3 --dummy stand --runner sync \
    -- --frame-delay 2 --headless
done
echo "=== FARM10 done $(date +%H:%M:%S)"
