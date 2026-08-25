#!/usr/bin/env bash
# 0825 rerun of both overnight arms, properly:
#  A1: SPECIALIST stage-internals shakeout — champion recipe +
#      --stage-internals (now actually wired; guard #9 aborts on
#      unknown flags). Read: NULL expected (features ~zero in this
#      pool) + canary/pipeline holds at the new 343-dim layout.
#  A2: GENERALIST PILOT — 300 master-master games (under the
#      non-streaming scale wall; overnight's embedding cache reused),
#      GRU-60 + --stage-internals.
# Run inside devenv shell. NO-MIX while running.
set -uo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

BASE=eval_runs/0825_rerun
mkdir -p "$BASE"
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== A1 TRAIN (champion + stage-internals, wired) $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
  --stage-internals \
  --lr 2.0e-4 \
  --target-loss 1.0e-9 \
  --snapshot-all \
  --out checkpoints/ms_g20si2.bin \
  > "$BASE/a1_train.log" 2>&1
rc=$?
grep -aE "UNRECOGNIZED|AWBC|COLLAPSE|exported|error" "$BASE/a1_train.log" | tail -4
if [ $rc -eq 0 ] && [ -f checkpoints/ms_g20si2.bin ]; then
  echo "=== A1 GATE-SWEEP $(date +%H:%M:%S)"
  bash scripts/gate_sweep.sh checkpoints/ms_g20si2 "$BASE/a1_sweep" --confirm \
    > "$BASE/a1_sweep.log" 2>&1
  grep -aE "ARGMAX" "$BASE/a1_sweep.log" | tail -1
else
  echo "=== A1 TRAIN FAILED (rc=$rc)" >&2
fi

echo "=== A2 GENERALIST PILOT (300 files) $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/train.exs \
  --backbone gru --window-size 60 \
  --replays replays/erickfm_ranked/FOX/extracted --max-files 300 \
  --epochs 20 --batch-size 256 \
  --stage-internals \
  --save-best \
  --name fox_gen_pilot1 \
  > "$BASE/a2_train.log" 2>&1
echo "=== A2 rc=$? $(date +%H:%M:%S)"
grep -aE "epoch 2?0|val|saved|Saved|error" "$BASE/a2_train.log" | tail -6

echo "=== RERUN DONE $(date +%H:%M:%S)"
