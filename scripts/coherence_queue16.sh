#!/usr/bin/env bash
# Queue 16 (2026-10-05 evening): lever 2 for the silent fall — sim DAgger.
# data/silent_fall/sim_dagger_r1*.frames = the 1-epoch baseline (evt2ctx_ck8)
# rolled out in the sim (3 seeds x 32 envs x 3600 f, self-play), offstage/below
# airborne frames relabeled by ExPhil.Agents.FoxRecoveryExpert, prev_controller =
# the previous label inside each trip (event heads read prev as the hold/change
# selector), 90 input-only context frames per trip (scripts/sim_recovery_dagger.exs).
# Mix interleaved through the epoch (pipeline.ex, 2026-10-05).
#
# History (INPUT_COHERENCE_2026-10-01.md "10-05 15:40"): the full set at
# oversample 4 shifted the policy globally (teacher-forced neutral share
# 0.26 -> 0.10, B presses 14 -> 40/min; closed-loop fidelity 0.33, SDs 2.3/min)
# — +50 % of the corpus's offstage data in a robotic style. Arms now:
#   evt2ctx_ck8_dags2   silent-only set (frames where the policy was silent), oversample 2
#   evt2ctx_ck8_dag1    full set, oversample 1 (dose check)
# Pass: high-band decided-trip return >= 0.6 (baseline 0.40, expert 0.95);
# closed-loop resume hazard >= .15 at 18-24 f; coherence neutral >= 0.22 and
# repeat >= 0.70 teacher-forced; fidelity <= 0.21; mismatch <= 0.16.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
[ -s data/silent_fall/sim_dagger_r1_silent.frames ] || { echo "missing data/silent_fall/sim_dagger_r1_silent.frames"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_dags2 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --mix-frames data/silent_fall/sim_dagger_r1_silent.frames --mix-oversample 2
run evt2ctx_ck8_dag1 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --mix-frames data/silent_fall/sim_dagger_r1.frames --mix-oversample 1
echo "QUEUE 16 DONE ($(date +%H:%M))"
