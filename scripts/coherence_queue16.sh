#!/usr/bin/env bash
# Queue 16 (2026-10-05 evening): lever 2 for the silent fall — sim DAgger.
# data/silent_fall/sim_dagger_r1.frames = the 1-epoch baseline (evt2ctx_ck8)
# rolled out in the sim (3 seeds x 32 envs x 3600 f, self-play), every
# offstage/below airborne frame relabeled by ExPhil.Agents.FoxRecoveryExpert,
# the policy's actual press in :prev_controller, 90 input-only context frames
# per trip (scripts/sim_recovery_dagger.exs). Mixed into the baseline recipe
# at two oversamples; the mix is appended after the main corpus each epoch.
#   evt2ctx_ck8_dag4    --mix-oversample 4
#   evt2ctx_ck8_dag16   --mix-oversample 16
# Pass: high-band decided-trip return >= 0.6 (baseline 0.40, expert 0.95);
# closed-loop resume hazard not decaying with silence (>= .15 at 18-24 f);
# mismatch <= 0.16; coherence repeat >= 0.6 / L-cancel >= 0.7; fidelity <= 0.20.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
[ -s data/silent_fall/sim_dagger_r1.frames ] || { echo "missing data/silent_fall/sim_dagger_r1.frames"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_dag4 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --mix-frames data/silent_fall/sim_dagger_r1.frames --mix-oversample 4
run evt2ctx_ck8_dag16 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --mix-frames data/silent_fall/sim_dagger_r1.frames --mix-oversample 16
echo "QUEUE 16 DONE ($(date +%H:%M))"
