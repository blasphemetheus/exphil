#!/usr/bin/env bash
# EVALS rule (09-21 17:05): drills at 1,000 starts. Six arms on ONE fresh pool (seed 21, built by epoch-3 self-play), vs self.
cd /home/blewf/git/exphil
log() { echo "[$(date +%H:%M:%S)] $*"; }
O=/home/blewf/git/exphil/eval_runs/0921_evals/drills1000
G="env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest"
declare -A P=( [ep3]=checkpoints/fox_v3_1_20260919_022008_ep3/model_best_policy.bin [ctrl]=checkpoints/fox_v3_1_step8_ctrl/model_policy.bin [mix2]=checkpoints/fox_v3_1_step8_mix2/model_policy.bin [mix3]=checkpoints/fox_v3_1_step8_mix3/model_policy.bin [mix3b]=checkpoints/fox_v3_1_step8_mix3b/model_policy.bin [mix4]=checkpoints/fox_v3_1_step8_mix4/model_policy.bin )
mkdir -p $O/ep3
devenv shell -- $G mix run scripts/sim_drill.exs --policy $PWD/${P[ep3]} --starts 1000 --horizon 240 --defender self --pool play --envs 32 --seed 21 --backend nif --out $O/ep3 > $O/ep3/run.log 2>&1; log "ep3 rc=$? $(grep BASELINE $O/ep3/run.log | cut -c1-160)"
for arm in ctrl mix2 mix3 mix3b mix4; do
  mkdir -p $O/$arm
  devenv shell -- $G mix run scripts/sim_drill.exs --policy $PWD/${P[$arm]} --starts 1000 --horizon 240 --defender self --pool file --pool-file $O/ep3/pool.term --envs 32 --seed 21 --backend nif --out $O/$arm > $O/$arm/run.log 2>&1; log "$arm rc=$? $(grep BASELINE $O/$arm/run.log | cut -c1-160)"
done
devenv shell -- env EXPHIL_GPU=0 mix run scripts/eval_ci.exs --drill "ep3=$O/ep3,ctrl=$O/ctrl,mix2=$O/mix2,mix3=$O/mix3,mix3b=$O/mix3b,mix4=$O/mix4" > $O/ci.txt 2>&1
log "CI: $(grep -E 'mix3b − mix2|mix4 − mix2|mix3 − mix2|mix2 − ep3' $O/ci.txt | tr '\n' ' ' | cut -c1-300)"
log "DRILLS1000_DONE"
