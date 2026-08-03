#!/usr/bin/env bash
# Task #17 arm 2: deliberate CPU load (12 busy loops ~ the 08-01 regime's
# launcher+netplay+session ambient). Lock arm passed clean (380.5 c367 x3),
# so load is the last cheap suspect for the 08-01 windowed 105 c19.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_QUEUE_TRACE=1
export EXPHIL_SKIP_NIF_COMPILE=1

pids=()
for i in $(seq 12); do (while :; do :; done) & pids+=($!); done
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
echo "=== LOADARM 12 busy loops up, loadavg $(cut -d' ' -f1 /proc/loadavg)"

echo "=== LOADARM windowed-under-load $(date +%H:%M:%S)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_g2_mdq_ss.bin \
  eval_runs/0802_loadarm_windowed --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --blocking-input --slippi-port 51442
echo "=== LOADARM done $(date +%H:%M:%S)"
