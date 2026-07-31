#!/usr/bin/env bash
# d3 rung, opening-coverage arms (2026-07-31 ~2:30am):
#   freeze2 — heads-only fine-tune (donor farm11_b1) WITH opening coverage
#   full2   — fresh full dagger retrain WITH opening coverage
# Openings = farm-9 live replays + the d3 metronome collects (the actual
# live opening distribution at the target delay), relabeled by the drill's
# d1-shifted expert. Evals: screening sync d3+d4 (+d2 for lock check).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12

OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/dagger_d3_round1_collect/r*.slp'
ROLL="eval_runs/dagger_d3_round1_collect/r1.slp,eval_runs/dagger_d3_round1_collect/r2.slp,eval_runs/dagger_d3_round1_collect/r3.slp,eval_runs/dagger_d3_round1_collect/r4.slp,eval_runs/dagger_d3_round1_collect/r5.slp,eval_runs/dagger_d3_round1_collect/r6.slp"
COMMON=(--expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp
        --rollouts "$ROLL" --opening-replays "$OPEN"
        --action-delay 3 --prev-action-dropout 0.6 --transition-weight 2.0)

echo "=== OPENROUND freeze2 TRAIN $(date +%H:%M:%S)"
FREEZE_DONOR=checkpoints/ms_farm11_b1.bin mix run scripts/dagger_drill_freeze.exs \
  "${COMMON[@]}" --out checkpoints/ms_d3_freeze2.bin \
  2>&1 | grep -aE "freeze-trunk|opening frames|Converged|exported|error|\*\*" | tail -6

echo "=== OPENROUND full2 TRAIN $(date +%H:%M:%S)"
mix run scripts/dagger_drill.exs \
  "${COMMON[@]}" --out checkpoints/ms_d3_full2.bin \
  2>&1 | grep -aE "opening frames|Converged|exported|error|\*\*" | tail -5

for name in freeze2 full2; do
  bin=checkpoints/ms_d3_$name.bin
  [ -f "$bin" ] || { echo "=== OPENROUND $name: no bin, skip evals" >&2; continue; }
  for d in 2 3 4; do
    echo "=== OPENROUND $name EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$bin" \
      eval_runs/0731_openround_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0
  done
done
echo "=== OPENROUND done $(date +%H:%M:%S)"
