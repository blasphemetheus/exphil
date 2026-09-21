#!/usr/bin/env bash
# Step 8 pilot (2026-09-21): BC on search-oracle episodes, mix arm vs control arm, then drills on one fresh pool.
cd /home/blewf/git/exphil
log() { echo "[$(date +%H:%M:%S)] $*"; }
EP3=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep3
TAGMAP=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep4/player_tag_map.json
EPS=/home/blewf/git/exphil/eval_runs/0921_sim_search/idle_n1000_ep/episodes.frames
O=/home/blewf/git/exphil/eval_runs/0921_step8
G="env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.70 EXPHIL_EXLA_PRECISION=highest"
COMMON="--backbone gru --temporal --stage-internals --hidden-sizes 512,512,256 --batch-size 128 --dropout 0.1 --precision f32 --bptt --unroll 80 --bptt-overlap 0 --bptt-val-files 16 --learn-player-styles --stream-chunk-size 200 --replays replays/erickfm_ranked/v2_filtered --train-character fox --select-character-port --label-delay 0 --player-tag-map $TAGMAP --resume $EP3/model_epoch1.axon --optimizer adamw --weight-decay 0.05 --learning-rate 2e-5 --lr-schedule constant --warmup-steps 1 --max-grad-norm 0.5 --epochs 1 --seed 905 --head autoregressive --save-best --label-smoothing 0.0 --no-focal-loss --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 --entropy-weight 0.0 --neutral-weight 1.0 --stick-edge-weight 1.0 --no-register --max-files 1400"
for arm in mix ctrl; do
  D=/home/blewf/git/exphil/checkpoints/fox_v3_1_step8_$arm; mkdir -p $D
  EXTRA=""; [ "$arm" = "mix" ] && EXTRA="--mix-frames $EPS --mix-oversample 30"
  devenv shell -- $G mix run scripts/train.exs $COMMON $EXTRA --checkpoint $D/model.axon > $D/train.log 2>&1
  log "train $arm rc=$? $(grep -E 'val|Val' $D/train.log | tail -1 | cut -c1-160)"
done
P3=$EP3/model_best_policy.bin; PM=/home/blewf/git/exphil/checkpoints/fox_v3_1_step8_mix/model_policy.bin; PC=/home/blewf/git/exphil/checkpoints/fox_v3_1_step8_ctrl/model_policy.bin
mkdir -p $O/drill_ep3_self
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_drill.exs --policy $P3 --starts 300 --horizon 240 --defender self --pool play --envs 32 --seed 7 --backend nif --out $O/drill_ep3_self > $O/drill_ep3_self/run.log 2>&1; log "drill ep3 self rc=$? $(grep BASELINE $O/drill_ep3_self/run.log | cut -c1-200)"
for arm in mix ctrl; do for def in self idle; do
  P=$PM; [ "$arm" = "ctrl" ] && P=$PC
  mkdir -p $O/drill_${arm}_$def
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_drill.exs --policy $P --starts 300 --horizon 240 --defender $def --pool file --pool-file $O/drill_ep3_self/pool.term --envs 32 --seed 7 --backend nif --out $O/drill_${arm}_$def > $O/drill_${arm}_$def/run.log 2>&1; log "drill $arm $def rc=$? $(grep BASELINE $O/drill_${arm}_$def/run.log | cut -c1-200)"
done; done
mkdir -p $O/drill_ep3_idle
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_drill.exs --policy $P3 --starts 300 --horizon 240 --defender idle --pool file --pool-file $O/drill_ep3_self/pool.term --envs 32 --seed 7 --backend nif --out $O/drill_ep3_idle > $O/drill_ep3_idle/run.log 2>&1; log "drill ep3 idle rc=$? $(grep BASELINE $O/drill_ep3_idle/run.log | cut -c1-200)"
log "STEP8_DONE"
