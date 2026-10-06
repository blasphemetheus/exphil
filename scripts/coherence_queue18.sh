#!/usr/bin/env bash
# Queue 18 (2026-10-05 night, after queue 17): dose-response on the one lever
# that moved the offstage release floor. --offstage-weight 3 halved the
# teacher-forced release hazard in the jumpless bands (Q7: -20..-60 j0 0.024 ->
# 0.015, < -60 0.026 -> 0.012; expert 0.007 / 0.000) and raised return 0.30 ->
# 0.37 at a fidelity cost (0.185 -> 0.209). The explicit release head (rel)
# did NOT move the floor — it is a learned, weakly state-conditioned prior,
# not a parametrisation artefact. Does x8 push the floor to the expert's, and
# what does it cost?
#   evt2ctx_ck8_off8   --offstage-weight 8
# Pass: Q7 within +0.005 of the expert per band; high-band decided return >= 0.6;
# fidelity <= 0.21; coherence repeat >= 0.70 / neutral >= 0.22.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue17; do sleep 60; done
echo "== queue 17 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off8 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 8
echo "QUEUE 18 DONE ($(date +%H:%M))"
