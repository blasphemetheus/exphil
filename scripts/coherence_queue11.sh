#!/usr/bin/env bash
# Queue 11 (2026-10-04): after the live look at evt2_ck8 / evt2_ck8w3 (sim
# matched live within 0.02 on every metric; SDs ~1.5/min = 3.5x expert are
# the remaining gap).
#   evt2_ck8_s906    training-seed replicate
#   evt2_ck8w3_s906  training-seed replicate
#   evt2_ck8w2       chunk weight 2 (between best-recovery w1 and best-fidelity w3)
#   evt2_ck8_e3      3 epochs: are the SDs under-training or recipe?
# Then the Q1/Q2 probe on each.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
SEED=906 run evt2_ck8_s906 "${pq[@]}" --chunk-horizon 8
SEED=906 run evt2_ck8w3_s906 "${pq[@]}" --chunk-horizon 8 --chunk-weight 3.0
run evt2_ck8w2 "${pq[@]}" --chunk-horizon 8 --chunk-weight 2.0
EPOCHS=3 run evt2_ck8_e3 "${pq[@]}" --chunk-horizon 8
mkdir -p eval_runs/1002_interp
for name in evt2_ck8_s906 evt2_ck8w3_s906 evt2_ck8w2 evt2_ck8_e3; do
  p=checkpoints/coh_$name/model_best_policy.bin
  [ -f "$p" ] || { echo "SKIP probe $name (no policy)"; continue; }
  echo "== probe $name ($(date +%H:%M))"
  EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.45 mix run --no-compile --no-deps-check scripts/interp_coherence_probe.exs \
    --policy "$p" --games 16 --stride 4 --out eval_runs/1002_interp/probe_$name.json > eval_runs/1002_interp/probe_$name.log 2>&1
  grep -E 'Q1 ablation|Q2 HEAD' eval_runs/1002_interp/probe_$name.log | sed 's/^\[[0-9:]*\] //'
done
echo "QUEUE 11 DONE ($(date +%H:%M))"
