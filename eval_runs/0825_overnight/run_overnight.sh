#!/usr/bin/env bash
# Overnight 2026-08-25 (Bradley: "what can I start off training
# overnight?"). Two GPU jobs in sequence + a parallel CPU extraction:
#
#  P0 (parallel, disk-only): extract the full master-master Fox
#     tarball (7,911 games) for the generalist line.
#  P1: SPECIALIST stage-internals arm — the exact champion recipe +
#     --stage-internals (first training exercise of the W4 feature +
#     the embedding canary), gate-swept with the new joint protocol
#     reads (fox argmax + mewtwo confirm; sweep-abort + starvation
#     guards armed).
#  P2: GENERALIST PILOT — first BC base on the master corpus:
#     GRU-60 on 1,000 master-master games with --stage-internals.
#     Deliverable is the pipeline shakeout + a first checkpoint; no
#     behavior gates overnight (scored in the morning).
#
# Run inside devenv shell. NO-MIX while this runs.
set -uo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

BASE=eval_runs/0825_overnight
mkdir -p "$BASE"

# ---- P0: extraction (background, CPU/disk only) ----
EXTRACT_DIR=replays/erickfm_ranked/FOX/extracted
if [ ! -d "$EXTRACT_DIR" ] || [ "$(ls "$EXTRACT_DIR" 2>/dev/null | wc -l)" -lt 1000 ]; then
  mkdir -p "$EXTRACT_DIR"
  echo "=== P0 EXTRACT start $(date +%H:%M:%S)"
  ( tar -xzf replays/erickfm_ranked/FOX/FOX_master-master_a1.tar.gz -C "$EXTRACT_DIR" \
      > "$BASE/extract.log" 2>&1 && echo "=== P0 EXTRACT done $(date +%H:%M:%S)" ) &
  EXTRACT_PID=$!
else
  EXTRACT_PID=""
  echo "=== P0 already extracted"
fi

# ---- P1: specialist stage-internals arm ----
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== P1 TRAIN (champion + --stage-internals) $(date +%H:%M:%S)"
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
  --out checkpoints/ms_g20si.bin \
  > "$BASE/p1_train.log" 2>&1
rc=$?
grep -aE "AWBC|COLLAPSE|Converged|exported|error" "$BASE/p1_train.log" | tail -4
if [ $rc -eq 0 ] && [ -f checkpoints/ms_g20si.bin ]; then
  echo "=== P1 GATE-SWEEP $(date +%H:%M:%S)"
  bash scripts/gate_sweep.sh checkpoints/ms_g20si "$BASE/p1_sweep" --confirm \
    > "$BASE/p1_sweep.log" 2>&1
  grep -aE "ARGMAX" "$BASE/p1_sweep.log" | tail -1
else
  echo "=== P1 TRAIN FAILED (rc=$rc) — continuing to P2" >&2
fi

# ---- P2: generalist pilot on the master corpus ----
if [ -n "$EXTRACT_PID" ]; then
  echo "=== waiting for P0 extraction"
  wait "$EXTRACT_PID" || true
fi

# Find the dir that actually holds .slp files (tarball layout unknown)
SLP_DIR=$(find "$EXTRACT_DIR" -name "*.slp" -printf "%h\n" 2>/dev/null | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
if [ -z "$SLP_DIR" ]; then
  echo "=== P2: no .slp found under $EXTRACT_DIR — falling back to pilot_sample" >&2
  SLP_DIR=replays/erickfm_ranked/pilot_sample
fi
echo "=== P2 GENERALIST PILOT on $SLP_DIR $(date +%H:%M:%S)"

EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/train.exs \
  --backbone gru --window-size 60 \
  --replays "$SLP_DIR" --max-files 1000 \
  --epochs 20 --batch-size 256 \
  --stage-internals \
  --save-best \
  --name fox_gen_pilot1 \
  > "$BASE/p2_train.log" 2>&1
echo "=== P2 rc=$? $(date +%H:%M:%S)"
grep -aE "epoch|loss|saved|error" "$BASE/p2_train.log" | tail -5

echo "=== OVERNIGHT DONE $(date +%H:%M:%S)"
