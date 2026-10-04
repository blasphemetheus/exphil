#!/usr/bin/env bash
# Queue 12 (2026-10-04): carried-state BPTT retested with the recipe that
# works windowed, at a matched update count. bptt_q (queue 8) froze
# completely, but it paired the carry with the TRUNK channel (the recipe that
# half-freezes windowed) and made 6x fewer updates than its control.
#   bptt_evt2_ck8_e5  GRU --bptt, event heads + chunk 8, 5 epochs (~16.7k updates = the windowed count)
#   bptt_q_e5         GRU --bptt, trunk-channel recipe, 5 epochs (under-training control for queue 8)
# Both share the gru_q holdout (checkpoints/coh_gru_q/split.json). ~70 min train each.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events)
split=checkpoints/coh_gru_q/split.json
run() { echo "== $1 ($(date +%H:%M))"; BACKBONE=gru TRAINER=scripts/train.exs scripts/coherence_experiment.sh "$@"; }
# smoke the new cells (events + chunk under --bptt) on 200 files before the long runs
# (default last-N holdout: the gru_q split's games are not all inside a 200-file slice)
MAX_FILES=200 EVALS="coherence closed_loop" run bptt_evt2_ck8_smoke "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --bptt --unroll 80 \
  2>&1 | tee /dev/stderr | grep -q TRAIN_FAILED && { echo "SMOKE FAILED; stopping"; exit 1; }
EPOCHS=5 run bptt_evt2_ck8_e5 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --bptt --unroll 80 --bptt-holdout-split "$split"
EPOCHS=5 run bptt_q_e5 "${pq[@]}" --bptt --unroll 80 --bptt-holdout-split "$split"
echo "QUEUE 12 DONE ($(date +%H:%M))"
