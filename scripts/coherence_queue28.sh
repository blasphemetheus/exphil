#!/usr/bin/env bash
# Queue 28 (2026-10-08 03:45, overnight, after queue 27) — the danger-readout
# head ON THE ACTUAL PORT RECIPE. Queue 26 ("10-08 03:30") took prev-action
# dropout OUT of the recipe (no dose keeps neutral in band; pd15_e3 fails), so
# queue 27's pd15_dng is a same-dose diagnostic against pd15 and this is the
# arm that can change the recipe:
#   evt2ctx_ck8_off3_dur8e_dng_e3   dur8e + --danger-context, 3 ep, no dropout
# Control = evt2ctx_ck8_off3_dur8e_e3 (the bar-passer: fidelity 0.175, repeat
# 0.759, neutral 0.221, offstage a1-3 enter-silence 0.0338, return 0.338).
# Pass = the port bar kept (fidelity <= 0.21, repeat 0.70-0.80, neutral
# 0.22-0.33) AND DecisionMap rows moved: jump with a jump in hand >= 0.15 at
# -20..-40 and >= 0.20 at -40..-60; Firefox once spent >= 0.04 at <= -40.
# Runs only if queue 27 compiled and passed its tests (its log says so);
# a queue-27 failure means the code needs a fix at the gap first.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue27; do sleep 60; done
echo "== queue 27 finished ($(date +%H:%M))"
if grep -qE "COMPILE_FAILED|TESTS_FAILED" logs/exphil-queue27.log; then
  echo "queue 27 failed compile/tests; refusing to start"; exit 1
fi
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_dng_e3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --danger-context
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_dng_e3
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_e3 evt2ctx_ck8_off3_dur8e_dng_e3
echo "QUEUE 28 DONE ($(date +%H:%M))"
