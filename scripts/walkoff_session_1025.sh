#!/usr/bin/env bash
# After queue 2: (a) the closed-loop readouts that the fused-handler crypto crash ate, with the
# fused handlers OFF; (b) walk-off origins for the candidate, the master arm, the null arm and
# the testbed base / neutral-weight arms (Bradley 10-25: "do 1 and 2 instead of the scoped weight").
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=0 EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45
R=eval_runs/1025_walkoff/REPORT.md; mkdir -p eval_runs/1025_walkoff eval_runs/walkoff
log() { echo "$*" | tee -a "$R"; }
log ""; log "# Walk-off session — waiting for queue 2 ($(date '+%F %H:%M'))"
until grep -q "QUEUE 2 DONE" eval_runs/1025_queue2/REPORT.md 2>/dev/null; do sleep 120; done
while pgrep -f '[b]eam.smp' >/dev/null; do sleep 60; done
run="mix run --no-compile --no-deps-check"
log "## (a) readouts with fused handlers off ($(date +%H:%M))"
for name in fox_mamba_master_ep6 fox_mamba_v3s906_ep3_null9o; do
  log "### $name"
  EVALS="closed_loop recovery recovery_means" scripts/coherence_experiment.sh "$name" 2>&1 | grep -E "RESULT|FAILED" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
  node scripts/recovery_curve_row.js "$name" 2>/dev/null | tee -a "$R" eval_runs/1011_full/curve.txt
done
log "## (b) walk-off origins ($(date +%H:%M))"
declare -A P=( [candidate_s906_ep2]=checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin [master_ep6]=checkpoints/coh_fox_mamba_master_ep6/model_best_policy.bin [null9o]=checkpoints/coh_fox_mamba_v3s906_ep3_null9o/model_best_policy.bin [pq_base]=checkpoints/coh_pq_base/model_best_policy.bin [pq_nw05]=checkpoints/coh_pq_nw05/model_best_policy.bin [pq_nw025]=checkpoints/coh_pq_nw025/model_best_policy.bin )
for name in candidate_s906_ep2 master_ep6 null9o pq_base pq_nw05 pq_nw025; do
  log "### $name"
  $run scripts/walkoff_origins.exs --policy "${P[$name]}" --label "$name" --envs 32 --frames 1800 --out "eval_runs/walkoff/$name.json" 2>&1 | grep -E "RESULT|\*\*|rror" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
done
log ""; log "WALKOFF SESSION DONE ($(date +%H:%M))"
