#!/usr/bin/env bash
# Queue 21 (2026-10-06 21:48): does the dur8e coherence/fidelity win replicate?
# dur8e (1 ep, seed 905) was the first arm inside the fidelity target (0.191)
# and the expert's repeat/neutral band (0.777 / 0.296) with the lowest val
# (3.25); recovery did not move (return 0.23). The port recipe earned its
# place with >= 3 epochs and a second seed, so the candidate gets the same.
#   evt2ctx_ck8_off3_dur8e_s906   seed 906, 1 ep   (seed robustness)
#   evt2ctx_ck8_off3_dur8e_e3     seed 905, 3 ep   (does it keep sharpening?)
# Pass (candidate for the port recipe): fidelity <= 0.21 and repeat 0.70-0.80 /
# neutral 0.22-0.33 on both; e3 return >= off3's 0.37 would be a bonus, not
# expected (the recovery lever is now lever 3).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
SEED=906 run evt2ctx_ck8_off3_dur8e_s906 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_e3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
echo "QUEUE 21 DONE ($(date +%H:%M))"
