#!/usr/bin/env bash
# Expert-iteration step 2 (2026-09-21): mix2 plays its own pool, its own samples are the oracle, mix3 trains on them.
cd /home/blewf/git/exphil
log() { echo "[$(date +%H:%M:%S)] $*"; }
EP3=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep3
TAGMAP=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep4/player_tag_map.json
O=/home/blewf/git/exphil/eval_runs/0921_step8
M2=/home/blewf/git/exphil/checkpoints/fox_v3_1_step8_mix2
G="env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest"
# 1. mix2's own 1,000-start pool (self-play) + its baseline on it
mkdir -p $O/pool_mix2
devenv shell -- $G mix run scripts/sim_drill.exs --policy $M2/model_policy.bin --starts 1000 --horizon 240 --defender self --pool play --envs 32 --seed 11 --backend nif --out $O/pool_mix2 > $O/pool_mix2/run.log 2>&1; log "pool_mix2 rc=$? $(grep BASELINE $O/pool_mix2/run.log | cut -c1-200)"
# 2. oracle = mix2's own samples on that pool
mkdir -p $O/oracle_mix2
devenv shell -- $G mix run scripts/sim_search.exs --policy $M2/model_policy.bin --pool-term $O/pool_mix2/pool.term --defender idle --teacher policy --temperature 1.2 --n 64 --horizon 90 --seed 2 --backend nif --out $O/oracle_mix2 --episodes-out $O/oracle_mix2/episodes.frames > $O/oracle_mix2/run.log 2>&1; log "oracle_mix2 rc=$? $(grep -a ORACLE $O/oracle_mix2/run.log | cut -c1-220)"
# 3. mix3 = mix2 + one subset pass with episodes2
COMMON="--backbone gru --temporal --stage-internals --hidden-sizes 512,512,256 --batch-size 128 --dropout 0.1 --precision f32 --bptt --unroll 80 --bptt-overlap 0 --bptt-val-files 16 --learn-player-styles --stream-chunk-size 200 --replays replays/erickfm_ranked/v2_filtered --train-character fox --select-character-port --label-delay 0 --player-tag-map $TAGMAP --resume $M2/model.axon --optimizer adamw --weight-decay 0.05 --learning-rate 2e-5 --lr-schedule constant --warmup-steps 1 --max-grad-norm 0.5 --epochs 1 --seed 906 --head autoregressive --save-best --label-smoothing 0.0 --no-focal-loss --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 --entropy-weight 0.0 --neutral-weight 1.0 --stick-edge-weight 1.0 --no-register --max-files 1400"
D=/home/blewf/git/exphil/checkpoints/fox_v3_1_step8_mix3; mkdir -p $D
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.70 EXPHIL_EXLA_PRECISION=highest mix run scripts/train.exs $COMMON --mix-frames $O/oracle_mix2/episodes.frames --mix-oversample 30 --checkpoint $D/model.axon > $D/train.log 2>&1
log "train mix3 rc=$? $(grep -aE 'Best val_loss' $D/train.log | tail -1 | cut -c1-120)"
# 4. drills on the SAME seed-7 pool as every arm so far + fingerprint
P=$D/model_policy.bin
for def in self idle; do
  mkdir -p $O/drill_mix3_$def
  devenv shell -- $G mix run scripts/sim_drill.exs --policy $P --starts 300 --horizon 240 --defender $def --pool file --pool-file $O/drill_ep3_self/pool.term --envs 32 --seed 7 --backend nif --out $O/drill_mix3_$def > $O/drill_mix3_$def/run.log 2>&1; log "drill mix3 $def rc=$? $(grep BASELINE $O/drill_mix3_$def/run.log | cut -c1-200)"
done
mkdir -p $O/fp_mix3
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_prior_play.exs --policy $P --games 10 --frames 1800 --opponent self --seed 100 --out $O/fp_mix3 > $O/fp_mix3/run.log 2>&1
devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_r1_compare.exs --dolphin $EP3/style_probe/bot_fingerprints.jsonl --dolphin-arm anon --sim $O/fp_mix3/sim_fingerprints.jsonl 2>&1 | sed -n '/games per arm/,$p' | grep -v "warning\|│\|└" > $O/fp_mix3/compare.txt
log "fingerprint mix3: $(grep -E 'tells|GATE' $O/fp_mix3/compare.txt | tr '\n' ' ' | cut -c1-300)"
log "ITER3_DONE"
