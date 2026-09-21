#!/usr/bin/env bash
# Overnight chain v2 (batched loop, shared pool) 2026-09-21 03:20
cd /home/blewf/git/exphil
P=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep3/model_best_policy.bin
G="env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest"
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p eval_runs/0921_sim_drill/self_n1000 eval_runs/0921_sim_drill/idle_n1000 eval_runs/0921_sim_search/idle_n1000 eval_runs/0921_sim_search/self_n100
devenv shell -- $G mix run scripts/sim_drill.exs --policy $P --starts 1000 --horizon 240 --defender self --pool play --envs 32 --seed 2 --out eval_runs/0921_sim_drill/self_n1000 > eval_runs/0921_sim_drill/self_n1000/run.log 2>&1; log "baseline self1000 rc=$? $(grep BASELINE eval_runs/0921_sim_drill/self_n1000/run.log | cut -c1-200)"
devenv shell -- $G mix run scripts/sim_drill.exs --policy $P --starts 1000 --horizon 240 --defender idle --pool file --pool-file eval_runs/0921_sim_drill/self_n1000/pool.term --envs 32 --seed 2 --out eval_runs/0921_sim_drill/idle_n1000 > eval_runs/0921_sim_drill/idle_n1000/run.log 2>&1; log "baseline idle1000 rc=$? $(grep BASELINE eval_runs/0921_sim_drill/idle_n1000/run.log | cut -c1-200)"
devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_search.exs --pool eval_runs/0921_sim_drill/self_n1000/pool.jsonl --defender idle --n 64 --horizon 90 --seed 1 --out eval_runs/0921_sim_search/idle_n1000 > eval_runs/0921_sim_search/idle_n1000/run.log 2>&1; log "search idle1000 rc=$? $(grep ORACLE eval_runs/0921_sim_search/idle_n1000/run.log | cut -c1-220)"
devenv shell -- $G mix run scripts/sim_search.exs --policy $P --defender self --starts 100 --n 64 --horizon 90 --seed 3 --out eval_runs/0921_sim_search/self_n100 > eval_runs/0921_sim_search/self_n100/run.log 2>&1; log "search self100 rc=$? $(grep ORACLE eval_runs/0921_sim_search/self_n100/run.log | cut -c1-220)"
log "OVERNIGHT_DONE"
