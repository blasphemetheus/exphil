#!/usr/bin/env bash
# Queue 14 (2026-10-05, overnight): the event-context lever replicated,
# trained longer, and on the Mamba testbed. Waits for queue 13 to finish,
# then compiles the timing-aware recovery-means scorecard (first-means
# height + latency), runs its tests, re-scores the expert table and the two
# reference models, then trains:
#   evt2ctx_ck8_s906   seed replicate (fidelity 0.185 at s905 is a single run)
#   evt2ctx_ck8_e3     3 epochs (does the stick-zone conditioning keep sharpening?)
#   mamba_evt2ctx_ck8  the full recipe on Mamba 256x2 — the port de-risk
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue13; do sleep 60; done
echo "== queue 13 finished ($(date +%H:%M)); compiling"
pgrep -af '[b]eam.smp' && { echo "a beam is still alive; refusing to compile"; exit 1; }
mix compile 2>&1 | tail -2
mix test test/exphil/eval/recovery_means_test.exs 2>&1 | tail -3
echo "== timing-aware recovery means ($(date +%H:%M))"
mix run --no-compile scripts/expert_recovery_means.exs 2>&1 | grep RESULT
for name in evt2_ck8 evt2ctx_ck8; do
  mix run --no-compile scripts/recovery_means.exs --policy checkpoints/coh_$name/model_best_policy.bin --label "$name" \
    --out eval_runs/1001_queue/$name/recovery_means.json > eval_runs/1001_queue/$name/recovery_means.log 2>&1
  grep "RESULT.*\(recovery means\|timing\)" eval_runs/1001_queue/$name/recovery_means.log | sed 's/^\[[0-9:]*\] //'
done
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
SEED=906 run evt2ctx_ck8_s906 "${pq[@]}" "${ev[@]}" --chunk-horizon 8
EPOCHS=3 run evt2ctx_ck8_e3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8
PROBE_BATCH=64 BACKBONE=mamba run mamba_evt2ctx_ck8 "${pq[@]}" "${ev[@]}" --chunk-horizon 8
echo "QUEUE 14 DONE ($(date +%H:%M))"
