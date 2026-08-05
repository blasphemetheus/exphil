#!/usr/bin/env bash
# CYCLE 4a2: y-augmentation at the SAFE dose (frame-budgeted).
#
# 4a run 1 (ms_g11_yaug): P3 — 44,450 aug frames (list-fraction sampling
# overdosed 3.5x the validated snippet scale) collapsed stand 421 -> 82.
# BUT plat X_mean moved −5.8 -> −1.35: the signal is right, the dose was
# wrong. This run: budget 12,000 frames (smallest lists first), all else
# identical to cycle 3b.
#
# Prereg:
#   P1 stand >= 300 AND plat X_mean continues toward 0 (>= −3) or fire > 0
#      => dose-response confirmed; iterate budget upward carefully.
#   P2 stand >= 300 but plat X_mean back near g10b's −5.8 => 12k is below
#      the effective threshold; the safe and effective windows may not
#      overlap -> F3 distill anchor becomes the enabler (anchor core,
#      raise dose).
#   P3 stand < 300 again even at 12k => altitude augmentation itself
#      interferes with the core (not composition) -> drop the approach,
#      cycle 4b records real platform play.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== CYCLE4A2 TRAIN $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --y-augment 1.0 --y-augment-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g11b_yaug.bin \
  2>&1 | grep -aE "Snippets:|Y-augment:|Converged|diverged|exported|error|\*\*" | tail -8
[ -f checkpoints/ms_g11b_yaug.bin ] || { echo "=== CYCLE4A2 FAILED" >&2; exit 1; }

echo "=== CYCLE4A2 GATE 1: dashboard platX (primary; unfiltered — 4a's grep ate an error)"
EXLA_TARGET=host mix run scripts/relapse_dashboard.exs \
  --policies "checkpoints/ms_g11b_yaug.bin,checkpoints/ms_g10b_human.bin" \
  --out eval_runs/interp/relapse_dashboard_g11b.json || true

echo "=== CYCLE4A2 GATE 2: stand d3 (1 run)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g11b_yaug.bin eval_runs/0804_cycle4a2_stand \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== CYCLE4A2 GATE 3: YS (3 runs)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g11b_yaug.bin eval_runs/0804_cycle4a2_ys \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 \
     --blocking-input --slippi-port 51442

echo "=== CYCLE4A2 chains"
for phase in stand ys; do
  echo "--- $phase"
  EXLA_TARGET=host mix run scripts/analyze_shine_source.exs \
    eval_runs/0804_cycle4a2_$phase/r*.slp 2>&1 | grep -aE "replay |r[0-9] "
done
echo "=== CYCLE4A2 done $(date +%H:%M:%S)"
