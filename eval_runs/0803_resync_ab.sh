#!/usr/bin/env bash
# Task #11 A/B: windowed vs pure-stateful vs stateful+resync-30, all on
# the trusted fast headless sync recipe (clean regime — the July 1.38x
# deficit was measured on a loaded laptop). Policy: ms_open_z (the July
# arms' lineage). Prereg:
#   P1: pure-stateful < windowed (reproduces the deficit cleanly)
#   P2: resync-30 within noise of windowed => drift was the mechanism,
#       hybrid becomes the recommended stateful deploy.
#   P3: all three equal => the July gap was regime artifact (like #17);
#       stateful is fine as-is.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_SKIP_NIF_COMPILE=1

CKPT=checkpoints/ms_open_z.bin
run_arm () { # name, extra args...
  local name=$1; shift
  echo "=== RESYNC-AB $name $(date +%H:%M:%S)"
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" \
    "eval_runs/0803_resync_$name" --runs 3 --dummy stand --runner "${RUNNER:-sync}" \
    -- ${EXTRA_PLAY:---headless --emulation-speed 0 --blocking-input} --slippi-port 51442 "$@"
}

run_arm windowed
run_arm stateful --stateful-step
run_arm resync30 --stateful-step --stateful-resync 30
echo "=== RESYNC-AB done $(date +%H:%M:%S)"
