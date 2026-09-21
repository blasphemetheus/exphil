#!/usr/bin/env bash
# Step 8 pilot v1 (policy-guided labels): waits for the oracle episodes, trains the mix2 arm, drills on the seed-7 pool, fingerprints.
cd /home/blewf/git/exphil
log() { echo "[$(date +%H:%M:%S)] $*"; }
EP3=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep3
TAGMAP=/home/blewf/git/exphil/checkpoints/fox_v3_1_20260919_022008_ep4/player_tag_map.json
O=/home/blewf/git/exphil/eval_runs/0921_step8
EPS=$O/pg_debug/episodes.frames
until [ -f $EPS ] && ! pgrep -f "sim_search.exs.*pg_debug" >/dev/null; do sleep 30; done
log "episodes ready: $(grep -a 'wrote' $O/pg_debug/run.log | tail -1 | cut -c1-160)"
G="env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.70 EXPHIL_EXLA_PRECISION=highest"
COMMON="--backbone gru --temporal --stage-internals --hidden-sizes 512,512,256 --batch-size 128 --dropout 0.1 --precision f32 --bptt --unroll 80 --bptt-overlap 0 --bptt-val-files 16 --learn-player-styles --stream-chunk-size 200 --replays replays/erickfm_ranked/v2_filtered --train-character fox --select-character-port --label-delay 0 --player-tag-map $TAGMAP --resume $EP3/model_epoch1.axon --optimizer adamw --weight-decay 0.05 --learning-rate 2e-5 --lr-schedule constant --warmup-steps 1 --max-grad-norm 0.5 --epochs 1 --seed 905 --head autoregressive --save-best --label-smoothing 0.0 --no-focal-loss --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 --entropy-weight 0.0 --neutral-weight 1.0 --stick-edge-weight 1.0 --no-register --max-files 1400"
D=/home/blewf/git/exphil/checkpoints/fox_v3_1_step8_mix2; mkdir -p $D
devenv shell -- $G mix run scripts/train.exs $COMMON --mix-frames $EPS --mix-oversample 30 --checkpoint $D/model.axon > $D/train.log 2>&1
log "train mix2 rc=$? $(grep -aE 'Best val_loss' $D/train.log | tail -1 | cut -c1-120)"
P=$D/model_policy.bin
for def in self idle; do
  mkdir -p $O/drill_mix2_$def
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_drill.exs --policy $P --starts 300 --horizon 240 --defender $def --pool file --pool-file $O/drill_ep3_self/pool.term --envs 32 --seed 7 --backend nif --out $O/drill_mix2_$def > $O/drill_mix2_$def/run.log 2>&1; log "drill mix2 $def rc=$? $(grep BASELINE $O/drill_mix2_$def/run.log | cut -c1-200)"
done
mkdir -p $O/fp_mix2
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest mix run scripts/sim_prior_play.exs --policy $P --games 10 --frames 1800 --opponent self --seed 100 --out $O/fp_mix2 > $O/fp_mix2/run.log 2>&1
devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_r1_compare.exs --dolphin $EP3/style_probe/bot_fingerprints.jsonl --dolphin-arm anon --sim $O/fp_mix2/sim_fingerprints.jsonl 2>&1 | sed -n '/games per arm/,$p' | grep -v "warning\|│\|└" > $O/fp_mix2/compare.txt
log "fingerprint mix2: $(grep -E 'tells|GATE' $O/fp_mix2/compare.txt | tr '\n' ' ' | cut -c1-260)"
log "STEP8V1_DONE"
