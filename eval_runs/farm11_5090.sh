#!/usr/bin/env bash
# FARM 11 (2026-07-30, 5090): opening-lottery levers on the trusted harness.
# Farm-9 baseline: champion recipe (174 opening frames — farm 6 had 870 via
# --opening-replays; the drift is itself a finding) gave a=4/6 escape,
# b=stable c5, c=wrong-side. Two arms, 2 seeds each, evals sync-d2 headless:
#   ARM A: + --opening-replays over farm 9's own live replays (this
#          machine's real absorbed/served openings, 15 slps)
#   ARM B: ARM A + --prev-action --scheduled-sampling 0.5
# Prereg: A raises per-run escape rate vs farm 9; B tests SS on top.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
# Canonical harness build pinned EXPLICITLY: fish config exports DOLPHIN_DIR
# to the old Ishiiruka netplay build, and ${VAR:-default} defers to it —
# farm 9/11 evals silently ran through Ishiiruka windowed because of this.
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
OPENING='eval_runs/0730_farm9_*/r*.slp'

train_eval () { # name, extra flags...
  local name=$1; shift
  echo "=== FARM11 $name: TRAIN $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  mix run scripts/train_multishine_policy.exs \
    --synth-recovery --synth-crouch --synth-opening \
    --opening-replays "$OPENING" \
    --x-hold-extend 3 --probe-basin \
    "$@" \
    --out checkpoints/ms_farm11_$name.bin \
    2>&1 | grep -aE "epoch|loss|Synthetic|X-hold|exported|Basin|error|\*\*" | tail -20
  if [ ! -f checkpoints/ms_farm11_$name.bin ]; then
    echo "=== FARM11 $name: TRAIN FAILED — no checkpoint, skipping eval" >&2
    return 1
  fi
  echo "=== FARM11 $name: EVAL $(date +%H:%M:%S)"
  bash scripts/eval_live_protocol.sh checkpoints/ms_farm11_$name.bin \
    eval_runs/0730_farm11_${name}_syncd2 --runs 4 --dummy stand --runner sync \
    -- --frame-delay 2 --headless
}

train_eval a1
train_eval a2
train_eval b1 --prev-action --scheduled-sampling 0.5
train_eval b2 --prev-action --scheduled-sampling 0.5
echo "=== FARM11 done $(date +%H:%M:%S)"
