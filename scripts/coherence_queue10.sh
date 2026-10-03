#!/usr/bin/env bash
# Queue 10 (2026-10-03 overnight): after the chunk-target sweep (queue 9:
# K=8 w1/w3 clearly better than the channel alone; K=4 inert; K=16 most
# expert-like but passive; nothing moves L-cancels or thin-mixup recoveries).
#   evt2_ck8          event heads (buttons + sticks, trunk sees no channel) + chunk 8
#   evt2_ck8w3        same, chunk weight 3
#   base_ck8          no channel + chunk 8: does the aux target change a model that already reads state?
#   prev_q_ck8_s906   training-seed replicate of the ck8 win
#   prev_q_ck8w3_s906 training-seed replicate of ck8w3
#   prev_q_ck12       fills the K curve between 8 and 16
#   mamba_prev_q_ck8  does the lever port to the target backbone (small Mamba, same slice)
# Then the Q1/Q2 interp probe on every run. Launch: scripts/launch_unit.sh exphil-q10 'bash scripts/coherence_queue10.sh'
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
# wait for the queue 9 follow-up (calibration + probes) if it is still running
while systemctl --user is-active --quiet exphil-q9b; do sleep 30; done
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2_ck8 "${pq[@]}" "${ev[@]}" --chunk-horizon 8
run evt2_ck8w3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --chunk-weight 3.0
run base_ck8 --chunk-horizon 8
SEED=906 run prev_q_ck8_s906 "${pq[@]}" --chunk-horizon 8
SEED=906 run prev_q_ck8w3_s906 "${pq[@]}" --chunk-horizon 8 --chunk-weight 3.0
run prev_q_ck12 "${pq[@]}" --chunk-horizon 12
BACKBONE=mamba run mamba_prev_q_ck8 "${pq[@]}" --chunk-horizon 8
mkdir -p eval_runs/1002_interp
for name in evt2_ck8 evt2_ck8w3 base_ck8 prev_q_ck8_s906 prev_q_ck8w3_s906 prev_q_ck12 mamba_prev_q_ck8; do
  p=checkpoints/coh_$name/model_best_policy.bin
  [ -f "$p" ] || { echo "SKIP probe $name (no policy)"; continue; }
  echo "== probe $name ($(date +%H:%M))"
  EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.45 mix run --no-compile --no-deps-check scripts/interp_coherence_probe.exs \
    --policy "$p" --games 16 --stride 4 --out eval_runs/1002_interp/probe_$name.json > eval_runs/1002_interp/probe_$name.log 2>&1
  grep -E 'Q[12] ' eval_runs/1002_interp/probe_$name.log | sed 's/^\[[0-9:]*\] //'
done
echo "QUEUE 10 DONE ($(date +%H:%M))"
