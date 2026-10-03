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
# variant, so differences are the flags. BACKBONE (default min_gru) and
# TRAINER (default scripts/train_fox_mamba.exs; scripts/train.exs for the
# carried-state `--bptt` GRU) select the model and driver — the parser takes
# a flag's FIRST occurrence, so these cannot be trailing overrides.
set -uo pipefail
name=$1; shift
out=eval_runs/1001_queue/$name
ckpt=checkpoints/coh_$name
mkdir -p "$out"
run="mix run --no-compile --no-deps-check"
export EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.45 EXPHIL_EXLA_PRECISION=highest
trainer=${TRAINER:-scripts/train_fox_mamba.exs}

train_args=(--backbone "${BACKBONE:-min_gru}" --stage-internals --hidden-sizes 256,256
  --batch-size 128 --precision f32 --window-size 80 --stride 5 --dropout 0.0 --learning-rate 0.0005
  --replays replays/erickfm_ranked/v2_filtered --train-character fox --select-character-port
  --max-files "${MAX_FILES:-3000}" --stream-chunk-size 64 --no-cache-streaming --label-delay 0 --epochs 1
  --seed "${SEED:-905}" --head autoregressive --save-best --save-every-batches 25000 --label-smoothing 0.0
  --no-focal-loss --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 --entropy-weight 0.0
  --neutral-weight 1.0 --stick-edge-weight 1.0 --name "coh-$name" --no-register --no-cache
  --checkpoint "$ckpt/model.axon" "$@")

if [ ! -f "$ckpt/model_best_policy.bin" ]; then
  $run "$trainer" "${train_args[@]}" > "$out/train.log" 2>&1 || { echo "TRAIN_FAILED $name"; exit 1; }
fi

# EVALS: space-separated subset of "coherence closed_loop recovery calibration fidelity"
# (default: all). Existing outputs are recomputed.
evals=${EVALS:-coherence closed_loop recovery calibration fidelity}
want() { case " $evals " in *" $1 "*) return 0;; *) return 1;; esac; }
policy=$ckpt/model_best_policy.bin
grep -o 'val_loss=[0-9.]*' "$out/train.log" | tail -1

if want coherence; then
  $run scripts/offline_input_coherence.exs --policy "$policy" --label "$name" --games 3 \
    --split "$ckpt/split.json" --out "$out/coherence.json" > "$out/coherence.log" 2>&1
  grep RESULT "$out/coherence.log" | sed 's/^\[[0-9:]*\] //'
fi

if want closed_loop; then
  $run scripts/sim_closed_loop.exs --policy "$policy" --label "$name" --envs 32 --frames 1800 \
    --out "$out/closed_loop.json" > "$out/closed_loop.log" 2>&1
  grep RESULT "$out/closed_loop.log" | sed 's/^\[[0-9:]*\] //'
fi

if want recovery; then
  $run scripts/recovery_drill.exs --policy "$policy" --label "$name" --out "$out/recovery.json" > "$out/recovery.log" 2>&1
  grep RESULT "$out/recovery.log" | sed 's/^\[[0-9:]*\] //'
fi

if want calibration && [ "$trainer" != scripts/train_fox_mamba.exs ]; then
  echo "SKIP calibration ($trainer has no CALIBRATE_ONLY mode)"
elif want calibration; then
  # CAL_TAG=x writes calibration_x.{json,log} (e.g. re-calibrating an existing
  # checkpoint with different data flags)
  cal=calibration${CAL_TAG:+_$CAL_TAG}
  CALIBRATE_ONLY=1 CALIBRATE_OUT="$out/$cal.json" $run scripts/train_fox_mamba.exs "${train_args[@]}" \
    --resume "$ckpt/model_best.axon" > "$out/$cal.log" 2>&1
  grep RESULT "$out/$cal.log" | sed 's/^\[[0-9:]*\] //'
fi

if want fidelity; then
  $run scripts/fidelity_scorecard.exs --policy "$policy" --label "$name" --envs 32 --frames 3600 \
    --seeds "${FIDELITY_SEEDS:-1001,1002,1003}" --out "$out/fidelity.json" > "$out/fidelity.log" 2>&1
  grep RESULT "$out/fidelity.log" | sed 's/^\[[0-9:]*\] //'
fi
echo "DONE $name"

# ABLATE_TOO=1: repeat the closed-loop evals with the prev-action channel
# zeroed at inference (is the policy competent WITHOUT the channel?).
if [ "${ABLATE_TOO:-0}" = 1 ]; then
  $run scripts/offline_input_coherence.exs --policy "$policy" --label "${name}_ablate" --games 3 \
    --split "$ckpt/split.json" --ablate-prev-action --out "$out/coherence_ablate.json" > "$out/coherence_ablate.log" 2>&1
  grep RESULT "$out/coherence_ablate.log" | sed 's/^\[[0-9:]*\] //'
  $run scripts/sim_closed_loop.exs --policy "$policy" --label "${name}_ablate" --envs 32 --frames 1800 \
    --ablate-prev-action --out "$out/closed_loop_ablate.json" > "$out/closed_loop_ablate.log" 2>&1
  grep RESULT "$out/closed_loop_ablate.log" | sed 's/^\[[0-9:]*\] //'
  $run scripts/recovery_drill.exs --policy "$policy" --label "${name}_ablate" --ablate-prev-action \
    --out "$out/recovery_ablate.json" > "$out/recovery_ablate.log" 2>&1
  grep RESULT "$out/recovery_ablate.log" | sed 's/^\[[0-9:]*\] //'
  echo "DONE ${name}_ablate"
fi
