#!/usr/bin/env bash
# Queue 6 (2026-10-02): (1) does a scheduled-sampling model tell its own inputs
# from the teacher's by FORMAT (re-calibrate with the channel quantized)?
# (2) symmetric stick targets; (3) quantized channel alone; (4) scheduled
# sampling on the quantized channel.
set -uo pipefail
ss="--prev-action --prev-action-dropout 0.0 --ss-steps 4 --ss-ramp-start 2000 --ss-ramp-steps 8000"
for spec in "prev_d00|--prev-action --prev-action-dropout 0.0" "ss25_k4|$ss --scheduled-sampling 0.25" "ss50_k4|$ss --scheduled-sampling 0.5" "ss100_k4|$ss --scheduled-sampling 1.0"; do
  name=${spec%%|*}; flags=${spec#*|}
  echo "== recalibrate $name with quantized channel"
  CAL_TAG=quant EVALS=calibration scripts/coherence_experiment.sh "$name" $flags --prev-action-quantize
done
echo "== base_rn (nearest stick rounding)"
EXPHIL_STICK_ROUNDING=nearest scripts/coherence_experiment.sh base_rn
echo "== prev_q (quantized channel, no scheduled sampling)"
scripts/coherence_experiment.sh prev_q --prev-action --prev-action-dropout 0.0 --prev-action-quantize
echo "== ss50_k4_q (scheduled sampling on the quantized channel)"
scripts/coherence_experiment.sh ss50_k4_q $ss --scheduled-sampling 0.5 --prev-action-quantize
echo "QUEUE6 DONE"
