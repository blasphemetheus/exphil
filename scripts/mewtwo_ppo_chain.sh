#!/usr/bin/env bash
# Mewtwo RL_ON_PRIOR chain: collect critic data -> refit the critic -> run PPO.
#
#   scripts/launch_unit.sh exphil-mewtwo-ppo 'bash scripts/mewtwo_ppo_chain.sh'
#
# Each stage is a separate `mix run`, so a failure stops the chain with the
# failing stage's log intact rather than silently continuing into PPO with a
# garbage critic. Stage boundaries are the only places the GPU is released.
#
# To stop the PPO stage cleanly at an iteration boundary (it saves first):
#   touch eval_runs/0924_mewtwo_ppo/v1/STOP
set -euo pipefail

REPO=/home/blewf/git/exphil
cd "$REPO"

PRIOR=checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin
RUN=eval_runs/0924_mewtwo_ppo
CRITIC_DATA=$RUN/critic_v1
CRITIC_FIT=$RUN/critic_v1_refit
PPO_OUT=$RUN/v1

export EXLA_TARGET=cuda
export EXPHIL_GPU_MEMORY_FRACTION=0.80
export EXPHIL_EXLA_PRECISION=highest

stamp() { date '+%H:%M:%S'; }

echo "=== $(stamp) STAGE 1/3: collect Mewtwo critic rollouts -> $CRITIC_DATA"
mix run scripts/critic_r2.exs \
  --policy "$PRIOR" \
  --character mewtwo --stage final_destination \
  --envs 64 --frames 1800 --rounds 12 \
  --gamma 0.995 --hidden 256 --epochs 20 --seed 924 \
  --out "$CRITIC_DATA"

echo "=== $(stamp) STAGE 2/3: refit the critic with a real train/val/test split -> $CRITIC_FIT"
mix run scripts/critic_refit.exs \
  --data "$CRITIC_DATA" --out "$CRITIC_FIT" --epochs 10

echo "=== $(stamp) STAGE 3/3: PPO on the frozen Mewtwo trunk -> $PPO_OUT"
# kl-coef 0.01 (vs Fox's 0.05): the Mewtwo prior is weak and SDs a lot, so it is
# given more room to leave the imitation manifold. This is a chosen experimental
# value, not a measured optimum — the kl-stop tripwire still bounds the drift.
mix run scripts/ppo_r3.exs \
  --policy "$PRIOR" \
  --critic "$CRITIC_FIT/critic_best.bin" \
  --character mewtwo --stage final_destination \
  --envs 64 --frames 600 --iters 300 \
  --epochs 2 --minibatch 4096 \
  --lr 3.0e-5 --gamma 0.995 --lambda 0.95 \
  --kl-coef 0.01 --kl-stop 0.5 \
  --save-every 10 --seed 924 \
  --out "$PPO_OUT"

echo "=== $(stamp) CHAIN COMPLETE"
