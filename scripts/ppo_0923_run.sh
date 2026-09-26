#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
mode=${1:-smoke}
case "$mode" in
  smoke) args=(--envs 8 --frames 60 --iters 2 --epochs 2 --minibatch 256 --save-every 1 --out eval_runs/0923_ppo/smoke_fixed) ;;
  train) args=(--envs 64 --frames 600 --iters 200 --epochs 2 --minibatch 3840 --save-every 10 --out eval_runs/0923_ppo/v1_fixed) ;;
  *) echo 'usage: ppo_0923_run.sh smoke|train' >&2; exit 2 ;;
esac
# Limit reservation to leave room for the desktop and the active Ollama model.
exec devenv shell -- env EXLA_MEMORY_FRACTION=0.35 mix run scripts/ppo_r3.exs \
  --policy checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin \
  --critic eval_runs/0923_r2/refit_v2/critic_best.bin "${args[@]}"
