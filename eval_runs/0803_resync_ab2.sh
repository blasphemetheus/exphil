#!/usr/bin/env bash
# Task #11 A/B v2 — on the CURRENT champion (ms_g2_mdq_ss @ d3), whose
# 380.5 c367 baseline is current-era and 3-run-deterministic. The v1
# attempt used ms_open_z, whose era baseline doesn't reproduce
# post-2bd9577. Chain-367 behavior gives drift a long horizon: if
# carried-state divergence matters, the stateful arm breaks chains.
# Prereg:
#   P1: windowed 380.5 c367 (regression guard — else regime is off).
#   P2: stateful << windowed on chains => drift real on current champ;
#       then resync30 recovering to windowed => hybrid validated,
#       becomes the recommended stateful deploy.
#   P3: all three at 380.5 c367 => no drift cost at d3 — stateful is
#       deploy-safe as-is for this class; document + close #11
#       (train-unroll fix stays future work).
# Also logs whether the stateful path ACTIVATES for the queue-depth-4
# embed (grep "Stateful step path ACTIVE" in r*.log) — the compat check.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_SKIP_NIF_COMPILE=1

CKPT=checkpoints/ms_g2_mdq_ss.bin
run_arm () { # name, extra args...
  local name=$1; shift
  echo "=== RESYNC-AB2 $name $(date +%H:%M:%S)"
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" \
    "eval_runs/0803_ab2_$name" --runs 3 --dummy stand --runner sync \
    -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input \
       --slippi-port 51442 "$@"
}

run_arm windowed
run_arm stateful --stateful-step
run_arm resync30 --stateful-step --stateful-resync 30
echo "=== ACTIVATION CHECK"
command grep -l "Stateful step path ACTIVE" eval_runs/0803_ab2_stateful/r*.log | wc -l
echo "=== RESYNC-AB2 done $(date +%H:%M:%S)"
