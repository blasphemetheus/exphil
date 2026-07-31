#!/usr/bin/env bash
# FARM 12 sweep-only rerun (2026-07-31): the training half of farm12_5090.sh
# succeeded; the sweeps ran against the fish-config Ishiiruka DOLPHIN_DIR and
# died on Null video. Re-sweep all four seeds through the pinned wrapper.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12

for name in shift1a shift1b shift2a fix1a; do
  [ -f checkpoints/ms_farm12_$name.bin ] || { echo "=== no bin for $name, skip" >&2; continue; }
  for d in 2 3 4; do
    echo "=== FARM12 $name: EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_farm12_$name.bin \
      eval_runs/0731_farm12_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0
  done
done
echo "=== FARM12 sweeps done $(date +%H:%M:%S)"
