#!/usr/bin/env bash
cd /home/blewf/git/exphil
P=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep3/model_best_policy.bin
for arm in self idle; do
  mkdir -p eval_runs/0921_sim_drill/${arm}_n200
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_drill.exs --policy $P --starts 200 --horizon 240 --defender $arm --pool play --seed 1 --out eval_runs/0921_sim_drill/${arm}_n200 > eval_runs/0921_sim_drill/${arm}_n200/run.log 2>&1
  echo RUN_EXIT=$? >> eval_runs/0921_sim_drill/${arm}_n200/run.log
done
echo ALL_DONE
