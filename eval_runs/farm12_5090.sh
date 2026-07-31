#!/usr/bin/env bash
# FARM 12 (2026-07-30, 5090): the delay ladder for Slippi Direct.
# Direct adds online_delay >= 2 on top of the pipeline's intrinsic +2
# (farm-11 delay-0 seeds lock at sync-d2), so a Direct bot needs seeds
# locking at sync-d3/d4+. Two mechanisms, both WITH the farm-11 opening
# coverage that closed the lottery:
#   shift arms:  --action-delay 1 / 2 (label shift — the "shift pathology"
#                verdict of farms 7-8 predates opening coverage; if the
#                absorber WAS the pathology, coverage resurrects shift
#                training and the ladder is one flag)
#   fixture arm: train on the d1-shifted teacher fixture (the DAgger-R3
#                route, no label shift)
# Evals: screening tier (EXLA_TARGET=host, ~3x) across sync d2/d3/d4.
# Prereg: shift-1 arms lock at d3 with chains >= 100 (pathology was the
# absorber); fixture arm locks at d3. Confirm any winner record-grade.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
# Canonical harness build pinned EXPLICITLY: fish config exports DOLPHIN_DIR
# to the old Ishiiruka netplay build, and ${VAR:-default} defers to it —
# farm 9/11 evals silently ran through Ishiiruka windowed because of this.
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
# EXLA_TARGET=host is EVAL-ONLY (screening tier): exported script-wide on
# the first run it put TRAINING on CPU too — 30+ min/seed vs minutes.
OPENING='eval_runs/0730_farm9_*/r*.slp'

train_one () { # name, extra train flags...
  local name=$1; shift
  echo "=== FARM12 $name: TRAIN $(date +%H:%M:%S)"
  mix run scripts/train_multishine_policy.exs \
    --synth-recovery --synth-crouch --synth-opening \
    --opening-replays "$OPENING" \
    --x-hold-extend 3 --probe-basin \
    "$@" \
    --out checkpoints/ms_farm12_$name.bin \
    2>&1 | grep -aE "Synthetic|X-hold|exported|error|\*\*" | tail -8
  [ -f checkpoints/ms_farm12_$name.bin ] || { echo "=== FARM12 $name TRAIN FAILED" >&2; return 1; }
}

sweep_one () { # name
  local name=$1
  for d in 2 3 4; do
    echo "=== FARM12 $name: EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh checkpoints/ms_farm12_$name.bin \
      eval_runs/0730_farm12_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0
  done
}

train_one shift1a --action-delay 1 && sweep_one shift1a
train_one shift1b --action-delay 1 && sweep_one shift1b
train_one shift2a --action-delay 2 && sweep_one shift2a
train_one fix1a --replays test/fixtures/replays/fox_multishine_closed_d1.slp && sweep_one fix1a
echo "=== FARM12 done $(date +%H:%M:%S)"
