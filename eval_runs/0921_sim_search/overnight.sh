#!/usr/bin/env bash
# Overnight chain 2026-09-21: waits for the drill baselines, then tests, 1,000-start baseline, search oracles.
cd /home/blewf/git/exphil
P=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep3/model_best_policy.bin
log() { echo "[$(date +%H:%M:%S)] $*"; }
while systemctl --user is-active exphil-sim-drill2 >/dev/null 2>&1; do sleep 30; done
log "drill baselines done"
devenv shell -- mix test test/exphil/sim/drill_test.exs test/exphil/sim/search_test.exs test/exphil_bridge/sim_state_test.exs > eval_runs/0921_sim_search/tests.log 2>&1; log "tests rc=$? $(grep -E 'tests,' eval_runs/0921_sim_search/tests.log | tail -1)"
mkdir -p eval_runs/0921_sim_drill/self_n1000
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_drill.exs --policy $P --starts 1000 --horizon 240 --defender self --pool play --seed 2 --out eval_runs/0921_sim_drill/self_n1000 > eval_runs/0921_sim_drill/self_n1000/run.log 2>&1; log "baseline1000 rc=$? $(grep BASELINE eval_runs/0921_sim_drill/self_n1000/run.log | cut -c1-200)"
mkdir -p eval_runs/0921_sim_search/idle_n200
devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_search.exs --pool eval_runs/0921_sim_drill/self_n200/pool.jsonl --defender idle --n 64 --horizon 90 --seed 1 --out eval_runs/0921_sim_search/idle_n200 > eval_runs/0921_sim_search/idle_n200/run.log 2>&1; log "search idle200 rc=$? $(grep ORACLE eval_runs/0921_sim_search/idle_n200/run.log | cut -c1-220)"
mkdir -p eval_runs/0921_sim_search/idle_n1000
devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_search.exs --pool eval_runs/0921_sim_drill/self_n1000/pool.jsonl --defender idle --n 64 --horizon 90 --seed 1 --out eval_runs/0921_sim_search/idle_n1000 > eval_runs/0921_sim_search/idle_n1000/run.log 2>&1; log "search idle1000 rc=$? $(grep ORACLE eval_runs/0921_sim_search/idle_n1000/run.log | cut -c1-220)"
mkdir -p eval_runs/0921_sim_search/self_n100
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_search.exs --policy $P --defender self --starts 100 --n 32 --horizon 90 --seed 3 --out eval_runs/0921_sim_search/self_n100 > eval_runs/0921_sim_search/self_n100/run.log 2>&1; log "search self100 rc=$? $(grep ORACLE eval_runs/0921_sim_search/self_n100/run.log | cut -c1-220)"
log "OVERNIGHT_DONE"
