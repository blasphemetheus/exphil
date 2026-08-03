#!/usr/bin/env bash
# Task #17: lock-state A/B, LOCKED arm. Baseline (unlocked, same block):
# 380.5 c367 x3 (0802_wvh_gap windowed). If this arm reproduces the
# 08-01 degradation (105 c19), the ingredient is lock/DPMS; if it holds
# the record, the suspect moves to ambient load.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_QUEUE_TRACE=1
export EXPHIL_SKIP_NIF_COMPILE=1

echo "=== LOCKARM windowed-under-lock $(date +%H:%M:%S)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g2_mdq_ss.bin \
  eval_runs/0802_lockarm_windowed --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --blocking-input --slippi-port 51442
echo "=== LOCKARM done $(date +%H:%M:%S)"
