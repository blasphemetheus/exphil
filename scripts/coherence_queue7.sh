#!/usr/bin/env bash
# Queue 7 (2026-10-02): (1) training-seed variance of the base recipe (every
# variant comparison so far is one training run each); (2) change-frame loss
# weighting on the live-format channel model, two doses.
set -uo pipefail
until ! systemctl --user is-active -q exphil-mamba-fidelity; do sleep 30; done
SEED=906 scripts/coherence_experiment.sh base_s906
SEED=907 scripts/coherence_experiment.sh base_s907
ch="--prev-action --prev-action-dropout 0.0 --prev-action-quantize"
scripts/coherence_experiment.sh prev_q_tw4 $ch --transition-weight 4
scripts/coherence_experiment.sh prev_q_tw16 $ch --transition-weight 16
echo "QUEUE7 DONE"
