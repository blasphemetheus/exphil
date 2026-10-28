#!/usr/bin/env bash
# Multi-arm testbed run (2026-10-28): embed the slice ONCE, train every arm from the same chunks
# (scripts/train_fox_mamba.exs --arms), then run the per-arm readouts. Replaces N x pq_experiment.sh
# where ~55 % of each arm was re-embedding the same replays.
#
#   scripts/pq_multi.sh PREFIX "name=--flag v;name2=--flag w" [shared extra train flags...]
#
# Arms land in checkpoints/coh_<PREFIX>_<name>/ and eval_runs/1001_queue/<PREFIX>_<name>/.
# MAX_FILES (3000), SEED (905), EVALS (coherence closed_loop recovery), AGREE_GAMES (3),
# WALKOFF (1) as env. Arm flags go first (the parser takes the first occurrence).
# One beam for the whole training stage; the readouts run one beam at a time after it.
set -uo pipefail
prefix=$1; spec=$2; shift 2
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=${EDIFICE_FUSED_CUSTOM_CALL:-1} EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=${EXPHIL_GPU_MEMORY_FRACTION:-0.45} EXPHIL_EXLA_PRECISION=highest
run="mix run --no-compile --no-deps-check"
# prefix every arm name
spec_p=$(echo "$spec" | tr ';' '\n' | sed "s/^ *\([^=]*\)=/${prefix}_\1=/" | paste -sd';' -)
names=$(echo "$spec" | tr ';' '\n' | sed 's/=.*//' | sed "s/^ */${prefix}_/")
shared=(--backbone "${BACKBONE:-min_gru}" --stage-internals --hidden-sizes 256,256
  --batch-size 128 --precision f32 --window-size "${WINDOW:-80}" --stride 5 --dropout 0.0 --learning-rate 0.0005
  --replays replays/erickfm_ranked/v2_filtered --train-character fox --select-character-port
  --max-files "${MAX_FILES:-3000}" --stream-chunk-size 64 --no-cache-streaming --label-delay 0 --epochs 1
  --seed "${SEED:-905}" --head autoregressive --save-best --save-every-batches 25000 --label-smoothing 0.0
  --no-focal-loss --button-pos-weight 1,1,1,1,1,1,1,1 --entropy-weight 0.0
  --stick-edge-weight 1.0 --no-register --no-cache
  --checkpoint "checkpoints/coh_${prefix}_shared/model.axon" "$@")
mkdir -p "eval_runs/1001_queue/${prefix}_multi"
echo "== multi-arm training: $(echo "$names" | wc -l) arms ($(date +%H:%M))"
$run scripts/train_fox_mamba.exs "${shared[@]}" --arms "$spec_p" > "eval_runs/1001_queue/${prefix}_multi/train.log" 2>&1 || { echo "TRAIN_FAILED ${prefix} (see eval_runs/1001_queue/${prefix}_multi/train.log)"; tail -5 "eval_runs/1001_queue/${prefix}_multi/train.log"; exit 1; }
grep -E "Chunks ready|ARM .* done|Multi-arm run complete" "eval_runs/1001_queue/${prefix}_multi/train.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-200
evals=${EVALS:-coherence closed_loop recovery}
for name in $names; do
  out=eval_runs/1001_queue/$name; ckpt=checkpoints/coh_$name; mkdir -p "$out"
  [ -f "$ckpt/model_best_policy.bin" ] || { echo "NO_POLICY $name"; continue; }
  cp "eval_runs/1001_queue/${prefix}_multi/train.log" "$out/train.log"
  grep "ARM ${name} done" "$out/train.log" | grep -o "val_loss=[0-9.]*" | tail -1 | sed "s/^/$name /"
  policy=$ckpt/model_best_policy.bin
  # the same readouts as pq_experiment.sh
  EVALS="$evals" scripts/pq_experiment.sh "$name" 2>&1 | grep -E "RESULT|FAILED" | sed 's/^\[[0-9:]*\] //' | cut -c1-300
  $run scripts/agreement_eval.exs --policy "$policy" --split "$ckpt/split.json" --games "${AGREE_GAMES:-3}" --stride 2 --out "$out/agreement.json" > "$out/agreement.log" 2>&1
  grep -E "RESULT (CHANGE|frames with|press calibration (jump|attack)|false press)" "$out/agreement.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330
  [ -f "$out/agreement.json" ] && node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1]));const s=j.by_action_change_frames.find(a=>a.action==="STANDING");const e=j.press_edges;console.log("RESULT STANDING change frames: n "+(s?s.n:0)+" change recall "+(s?s.change_recall:"?")+"%  | press edges same-family: "+e.map(x=>x.family+" "+x.model_same_press+"%").join(" "))' "$out/agreement.json"
  if [ "${WALKOFF:-1}" = 1 ]; then
    $run scripts/walkoff_origins.exs --policy "$policy" --label "$name" --envs 32 --frames 1800 --out "eval_runs/walkoff/$name.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250
  fi
done
echo "== multi-arm done ($(date +%H:%M)) — table: node scripts/pq_digest.js $(echo $names | tr '\n' ' ')"
