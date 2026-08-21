#!/usr/bin/env bash
# PEAK SCIENCE P2 — the scheduled-sampling hypothesis: two full arms of
# the g19 recipe with SS at the extremes, swept. Question: is
# exposure-bias pressure (self-conditioned prev-action during training)
# the driver of the two-attractor hopping?
#
# PRE-REGISTERED (2026-08-21, before the run):
#   Baselines (ss=0.5 ramp 10): g19 (peak ep3-8, decay tail) and g20
#   (later onset, peaks to ep60). Ceiling 435-437 both.
#   Arm ss0    = NO scheduled sampling (flag omitted).
#   Arm ssfull = ss 0.5 from epoch 1 (--ss-ramp 1).
#   Reads (peak statistics from the sweeps: onset epoch, ceiling,
#   #epochs >=300, basin/ceiling hop count):
#     P2-SS-DRIVES  ss0 shows no hopping (monotone-ish trajectory,
#       whether high or low plateau) while ssfull hops from epoch 1 ->
#       SS pressure IS the attractor-hopping mechanism; the recipe
#       lever is an SS schedule (e.g. anneal SS instead of lr).
#     P2-SS-HEIGHT  ss0 plateaus LOW (never >=300) -> SS is required
#       to reach the ceiling at all (closed-loop skill needs
#       self-conditioning); hopping is its price. Composite schedules
#       (ss on early, off late) become the follow-up.
#     P2-NULL       both arms hop like the baselines -> SS is not the
#       lever; remaining suspects: AWBC weighting noise, pool
#       composition per-epoch shuffle, intrinsic GRU closed-loop
#       sensitivity.
#   NO CROWN from stand numbers (g6 rule).
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0821_peak_p2_ss
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

run_arm() {  # $1=name  $2...=ss flags
  local NAME="$1"; shift
  echo "=== P2 ${NAME} TRAIN $(date +%H:%M:%S)"
  EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
    --opp-scramble-frames 12000 \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --awbc \
    --lr 2.0e-4 \
    --target-loss 1.0e-9 \
    --snapshot-all "$@" \
    --out "checkpoints/ms_p2_${NAME}.bin" \
    2>&1 | tee "$OUT/${NAME}_train.log" \
    | grep -aE "AWBC|COLLAPSE|Converged|diverged|exported|error" | tail -6
  [ -f "checkpoints/ms_p2_${NAME}.bin" ] || { echo "=== P2 ${NAME} TRAIN FAILED" >&2; return 1; }

  echo "=== P2 ${NAME} GATE-SWEEP $(date +%H:%M:%S)"
  bash scripts/gate_sweep.sh "checkpoints/ms_p2_${NAME}" "$OUT/${NAME}_sweep" --confirm
}

run_arm "ss0" || true
run_arm "ssfull" --scheduled-sampling 0.5 --ss-ramp 1 || true

echo "=== P2 DONE $(date +%H:%M:%S). Score peak statistics vs prereg reads."