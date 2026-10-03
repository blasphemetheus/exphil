#!/usr/bin/env bash
# Queue 8 (2026-10-02 evening): carried-state BPTT pair + chunk targets.
#   gru_q     windowed GRU control, prev_q recipe, zero initial state
#   bptt_q    same recipe, `train.exs --bptt` (carry across 80-frame chunks,
#             per-timestep loss), scored on gru_q's held-out games
#   prev_q_ck8  MinGRU prev_q recipe + 8-frame chunk targets
# SMOKE=1 runs each on 200 files with a short eval set to check plumbing +
# cost (GRU updates vs MinGRU's 8 ms) before the 3000-file runs.
set -uo pipefail
cd /home/blewf/git/exphil
recipe=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
if [ "${SMOKE:-0}" = 1 ]; then
  export MAX_FILES=200 EVALS="coherence closed_loop recovery"
  sfx=_smoke
else
  sfx=
fi

echo "== gru_q$sfx"
BACKBONE=gru scripts/coherence_experiment.sh "gru_q$sfx" "${recipe[@]}" --recurrent-state zeros
echo "== bptt_q$sfx"
BACKBONE=gru TRAINER=scripts/train.exs scripts/coherence_experiment.sh "bptt_q$sfx" "${recipe[@]}" \
  --bptt --unroll 80 --bptt-holdout-split "checkpoints/coh_gru_q$sfx/split.json"
echo "== prev_q_ck8$sfx"
scripts/coherence_experiment.sh "prev_q_ck8$sfx" "${recipe[@]}" --chunk-horizon 8
echo "QUEUE 8 DONE"
