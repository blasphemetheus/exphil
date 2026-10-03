#!/usr/bin/env bash
# Queue 9 (2026-10-02 night): chunk-target sweep after prev_q_ck8 doubled
# self-play damage (29 -> 56/min) and halved SDs at equal fidelity.
#   prev_q_ck4   horizon 4
#   prev_q_ck16  horizon 16
#   prev_q_ck8w3 horizon 8, chunk weight 3 (future heads dominate the trunk)
set -uo pipefail
cd /home/blewf/git/exphil
recipe=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
for spec in "prev_q_ck4 --chunk-horizon 4" "prev_q_ck16 --chunk-horizon 16" "prev_q_ck8w3 --chunk-horizon 8 --chunk-weight 3.0"; do
  set -- $spec
  name=$1; shift
  echo "== $name"
  scripts/coherence_experiment.sh "$name" "${recipe[@]}" "$@"
done
echo "QUEUE 9 DONE"
