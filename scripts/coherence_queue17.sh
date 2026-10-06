#!/usr/bin/env bash
# Queue 17 (2026-10-05 night): the structural fix for the silent fall.
# Recovery probe Q7 (INPUT_COHERENCE "10-05 20:00"): teacher-forced on expert
# frames the policy RELEASES a deflected stick offstage at a ~2.5 %/frame floor
# regardless of state (expert: 0.7 % at -20..-60 jumpless, 0.0 % below -60);
# closed-loop that compounds to 3-5x the expert's entry-into-silence hazard.
# Mechanism: a release is the change softmax's centre mass (the onstage prior);
# hold is an explicit decision and is calibrated. --stick-release makes release
# an explicit decision too (K change + hold + release logits per axis).
#   evt2ctx_ck8_rel        baseline recipe + --stick-release
#   evt2ctx_ck8_rel_off3   + --offstage-weight 3 (the data lever that halved the floor in jumpless bands)
#   evt2ctx_ck8_off3_s906  off3 seed replicate (mismatch 0.088 at the expert floor; single seed so far)
# Pass (rel): Q7 release hazard within +0.005 of the expert per band
# (-20..-60 j0 <= 0.012, < -60 <= 0.005, -20..-60 j1+ <= 0.033); closed-loop
# P(enter silence | active) per 3 f at -20..-60 <= 0.04; high-band decided
# return >= 0.6 (baseline 0.40); coherence repeat >= 0.70 / neutral >= 0.22;
# fidelity <= 0.20; mismatch <= 0.16.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_rel "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --stick-release
run evt2ctx_ck8_rel_off3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --stick-release --offstage-weight 3
SEED=906 run evt2ctx_ck8_off3_s906 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3
echo "QUEUE 17 DONE ($(date +%H:%M))"
