#!/usr/bin/env bash
# Task #8: P5 curation cycle #1 — the fight-state gap, v1 with SYNTHETIC
# pressure (champion vs cpu-9 Fox; the human-replay targets were lost to
# GOTCHA #84 and a fresh human session is Bradley-gated).
#
# Cycle: measure -> collect -> retrain -> re-measure.
#   Phase M0: baseline fight metric — champion vs cpu-9 at d3, 3 runs;
#             the metric is the hold-B pathology (B-run histogram via
#             analyze_qtrace on the qtrace'd run logs) + shine activity.
#   Phase C:  collect 12 x 60s champion-vs-cpu9 rollouts, temperature
#             0.4 (diversity rule), qtrace on. The eval protocol's
#             r*.slp ARE the pool.
#   Phase T:  retrain champion recipe + pressure pool in sources
#             (relabeling: dagger_drill's multishine expert relabels
#             every visited frame — pressure states get return-to-cycle
#             labels).
#   Phase M1: re-measure vs cpu-9 AND stand (regression gate: stand d3
#             must stay >= 300; prereg fight gate: B-run p95 halves and
#             shines/min vs cpu-9 up >= 1.5x).
# RUN AFTER grind-6 (needs the machine; VRAM 0.7 for the T phase).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1
export EXPHIL_QUEUE_TRACE=1

CKPT=checkpoints/ms_g2_mdq_ss.bin
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0803_pressure_pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== PRESSURE M0 baseline vs cpu9 $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh "$CKPT" \
  eval_runs/0803_pressure_m0 --runs 3 --dummy cpu --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input \
     --dummy-cpu-level 9 --slippi-port 51442

echo "=== PRESSURE C collect $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh "$CKPT" \
  eval_runs/0803_pressure_pool --runs 12 --dummy cpu --runner sync --temperature 0.4 \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input \
     --dummy-cpu-level 9 --slippi-port 51442

echo "=== PRESSURE T retrain $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.7 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g7_pressure.bin \
  2>&1 | grep -aE "Converged|diverged|exported|error|\*\*" | tail -4
[ -f checkpoints/ms_g7_pressure.bin ] || { echo "=== PRESSURE FAILED" >&2; exit 1; }

echo "=== PRESSURE M1 re-measure $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g7_pressure.bin \
  eval_runs/0803_pressure_m1_cpu --runs 3 --dummy cpu --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input \
     --dummy-cpu-level 9 --slippi-port 51442
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g7_pressure.bin \
  eval_runs/0803_pressure_m1_stand --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== PRESSURE B-run analysis $(date +%H:%M:%S)"
for d in eval_runs/0803_pressure_m0 eval_runs/0803_pressure_m1_cpu; do
  for log in "$d"/r*.log; do
    echo "--- $log"
    EXLA_TARGET=host mix run scripts/analyze_qtrace.exs "$log" 2>/dev/null | tail -6
  done
done
echo "=== PRESSURE done $(date +%H:%M:%S)"
