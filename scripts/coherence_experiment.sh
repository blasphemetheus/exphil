#!/usr/bin/env bash
# One input-coherence experiment on the MinGRU testbed (2026-10-01):
# train -> offline scoreboard -> closed-loop sim rollout -> recovery drill.
#
#   scripts/coherence_experiment.sh NAME [extra train flags...]
#
# Run inside devenv from the repo root, with no other beam alive. Outputs:
#   checkpoints/coh_NAME/            trained policy
#   eval_runs/1001_queue/NAME/       coherence.json, closed_loop.json, recovery.json, *.log
# MAX_FILES (default 3000) sets the corpus slice; same slice + seed for every
# variant, so differences are the flags.
set -uo pipefail
name=$1; shift
out=eval_runs/1001_queue/$name
ckpt=checkpoints/coh_$name
mkdir -p "$out"
run="mix run --no-compile --no-deps-check"
export EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.45 EXPHIL_EXLA_PRECISION=highest

if [ ! -f "$ckpt/model_best_policy.bin" ]; then
  $run scripts/train_fox_mamba.exs --backbone min_gru --stage-internals --hidden-sizes 256,256 \
    --batch-size 128 --precision f32 --window-size 80 --stride 5 --dropout 0.0 --learning-rate 0.0005 \
    --replays replays/erickfm_ranked/v2_filtered --train-character fox --select-character-port \
    --max-files "${MAX_FILES:-3000}" --stream-chunk-size 64 --no-cache-streaming --label-delay 0 --epochs 1 \
    --seed 905 --head autoregressive --save-best --save-every-batches 25000 --label-smoothing 0.0 \
    --no-focal-loss --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 --entropy-weight 0.0 \
    --neutral-weight 1.0 --stick-edge-weight 1.0 --name "coh-$name" --no-register --no-cache \
    --checkpoint "$ckpt/model.axon" "$@" > "$out/train.log" 2>&1 || { echo "TRAIN_FAILED $name"; exit 1; }
fi
policy=$ckpt/model_best_policy.bin
grep -o 'val_loss=[0-9.]*' "$out/train.log" | tail -1

$run scripts/offline_input_coherence.exs --policy "$policy" --label "$name" --games 3 \
  --split "$ckpt/split.json" --out "$out/coherence.json" > "$out/coherence.log" 2>&1
grep RESULT "$out/coherence.log" | sed 's/^\[[0-9:]*\] //'

$run scripts/sim_closed_loop.exs --policy "$policy" --label "$name" --envs 32 --frames 1800 \
  --out "$out/closed_loop.json" > "$out/closed_loop.log" 2>&1
grep RESULT "$out/closed_loop.log" | sed 's/^\[[0-9:]*\] //'

$run scripts/recovery_drill.exs --policy "$policy" --label "$name" --out "$out/recovery.json" > "$out/recovery.log" 2>&1
grep RESULT "$out/recovery.log" | sed 's/^\[[0-9:]*\] //'
echo "DONE $name"
