#!/usr/bin/env bash
# Queue 20 (2026-10-06, after queue 19): the semi-Markov main stick with button
# edges as decision frames. dur8 halved the offstage enter-silence hazards but
# died 35/35 side-B trips: buttons are sampled before sticks, so a B press with
# a committed sideways stick is an Illusion. Now any button edge is a decision
# frame (training mask + sampler), so a press can re-aim the stick.
#   evt2ctx_ck8_off3_dur8e   recipe + --stick-duration 8 (edge decisions)
# Pass: dur8's map gains kept (offstage deep/high j0 <= 0.026/0.024) AND return
# >= off3's 0.37, side_b deaths back to off3's level, mismatch <= 0.10,
# fidelity <= 0.21, onstage enter_silence ratio >= 0.8.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue19; do sleep 60; done
echo "== queue 19 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off3_dur8e "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
echo "QUEUE 20 DONE ($(date +%H:%M))"
