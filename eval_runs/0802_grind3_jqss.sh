#!/usr/bin/env bash
# Grind 3 (2026-08-02, task #16): the d2-inversion dynamics probe.
# Arm jq_ss = grind-2's champion mdq_ss recipe + --shift-jitter 1 (the
# R3-style label smear). Context: the delay-id patch probe (0802) proved
# the d2 weakness is NOT the id channel; construction audit shows all
# rungs built symmetrically (shift d+2). Reframed question: SS boosted
# d3/d4 ~5x but d2 only ~2x — does deliberate shift jitter close d2?
# Prereg:
#   P1: jq_ss d2 > 200/min c>10 => jitter fixes d2 (jitterless-shift
#       hypothesis CONFIRMED causally).
#   P2: jq_ss d3/d4 stay >= 300/c100 => jitter doesn't break the records.
#   P3: d2 unchanged (~140 c6) => jitter irrelevant; suspect moves to
#       source-distribution (all rollouts were d3-collected).
# VRAM 0.7 per HANDOFF_2026-08-02 (queue-SS graph OOMs at 0.25).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.7

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

run_arm () { # name, train flags...
  local name=$1; shift
  echo "=== GRIND3 $name TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    "$@" \
    --out checkpoints/ms_g3_$name.bin \
    2>&1 | grep -aE "jitter draws|Converged|diverged|exported|error|\*\*" | tail -6
  [ -f checkpoints/ms_g3_$name.bin ] || { echo "=== GRIND3 $name FAILED" >&2; return 1; }
  for d in 2 3 4; do
    echo "=== GRIND3 $name EVAL d$d $(date +%H:%M:%S)"
    EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
      checkpoints/ms_g3_$name.bin \
      eval_runs/0802_g3_${name}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --slippi-port 51442
  done
}

run_arm jq_ss --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 --shift-jitter 1
echo "=== GRIND3 done $(date +%H:%M:%S)"
