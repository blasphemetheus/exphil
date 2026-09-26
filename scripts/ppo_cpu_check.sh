#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
base=(--policy checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin --games 32 --frames 28800 --envs 8 --seed 920240)
devenv shell -- env EXLA_TARGET=host CUDA_VISIBLE_DEVICES= ERL_FLAGS='+S 4:4' \
  elixir -pa '_build/dev/lib/*/ebin' -r scripts/ppo_cpu_boot.exs scripts/ppo_eval.exs \
  "${base[@]}" --out eval_runs/0923_ppo/cpu_prior_control
devenv shell -- env EXLA_TARGET=host CUDA_VISIBLE_DEVICES= ERL_FLAGS='+S 4:4' \
  elixir -pa '_build/dev/lib/*/ebin' -r scripts/ppo_cpu_boot.exs scripts/ppo_eval.exs \
  "${base[@]}" --head eval_runs/0923_ppo/v1_fixed/head_iter40.bin --out eval_runs/0923_ppo/cpu_candidate40
python3 scripts/ppo_style_report.py eval_runs/0923_ppo/cpu_candidate40
