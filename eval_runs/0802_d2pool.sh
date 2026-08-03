#!/usr/bin/env bash
# Task #16: collect a d2-native rollout pool. Every existing pool was
# collected at d3 — the last standing suspect for the d2 inversion is
# that d2's training states are d3-shaped. 12 x 60s champion rollouts
# through --frame-delay 2, fast headless recipe (0802: speed0+blocking
# = record-equivalent at half wall time). Replays land in the outdir
# (r*.slp) for the grind-4 mixed-pool retrain.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_SKIP_NIF_COMPILE=1

echo "=== D2POOL collect $(date +%H:%M:%S)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g2_mdq_ss.bin \
  eval_runs/0802_d2pool --runs 12 --dummy stand --runner sync --temperature 0.4 \
  -- --frame-delay 2 --headless --emulation-speed 0 --blocking-input --slippi-port 51442
echo "=== D2POOL done $(date +%H:%M:%S)"
