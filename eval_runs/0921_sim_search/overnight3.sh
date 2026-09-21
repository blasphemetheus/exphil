#!/usr/bin/env bash
# Overnight chain v3: search stages on the binary-rows worktree (branch sim-binary-rows)
cd /home/blewf/git/exphil
P=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep3/model_best_policy.bin
W=/home/blewf/git/exphil-boundary
MAIN=/home/blewf/git/exphil
log() { echo "[$(date +%H:%M:%S)] $*"; }
rm -rf $MAIN/eval_runs/0921_sim_search/idle_n1000; mkdir -p $MAIN/eval_runs/0921_sim_search/idle_n1000 $MAIN/eval_runs/0921_sim_search/self_n100
devenv shell -- sh -c "cd $W && EXPHIL_GPU=0 mix run scripts/sim_search.exs --pool $MAIN/eval_runs/0921_sim_drill/self_n1000/pool.jsonl --defender idle --n 64 --horizon 90 --seed 1 --out $MAIN/eval_runs/0921_sim_search/idle_n1000" > $MAIN/eval_runs/0921_sim_search/idle_n1000/run.log 2>&1; log "search idle1000 (binary) rc=$? $(grep ORACLE $MAIN/eval_runs/0921_sim_search/idle_n1000/run.log | cut -c1-220)"
devenv shell -- sh -c "cd $W && EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_search.exs --policy $P --defender self --starts 100 --n 64 --horizon 90 --seed 3 --out $MAIN/eval_runs/0921_sim_search/self_n100" > $MAIN/eval_runs/0921_sim_search/self_n100/run.log 2>&1; log "search self100 (binary) rc=$? $(grep ORACLE $MAIN/eval_runs/0921_sim_search/self_n100/run.log | cut -c1-220)"
log "OVERNIGHT_DONE"
