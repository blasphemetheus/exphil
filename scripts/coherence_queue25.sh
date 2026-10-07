#!/usr/bin/env bash
# Queue 25 (2026-10-07 18:45, after queue 24) — the no-prev-action arm, done
# right. Queue 24's `nopq` (dur8e recipe without --prev-action) failed at
# config time: the event heads (--button-events/--stick-events) REQUIRE
# use_prev_action — the channel cannot be omitted inside the event recipe.
# The equivalent arm is to keep the channel and never let the model see it:
#   evt2ctx_ck8_off3_dur8e_pd100  --prev-action --prev-action-dropout 1.0 (the
#                                 channel is zeroed on every training frame)
#                                 + EVAL_ABLATE=1 (every closed-loop eval zeroes
#                                 it live; coherence_experiment.sh hook)
# Reads vs queue 24's header: return >= 0.45 or dies-with-jump-left <= 15 %
# with repeat >= 0.70 / neutral 0.22-0.33 / fidelity <= 0.23 kept — if pd100
# keeps the band, the prev-action channel leaves the port recipe (the
# duration head carries commitment instead). Decision-hazard bar (DecisionMap,
# now compiled): jump hazard at -20..-40 and -40..-60 with a jump in hand
# >= 30 % (expert 39 / 52 closed-loop; per-frame map 0.20 / 0.30).
# Prediction (doc "10-07 15:30"): jump-in-hand deaths move toward ~10 %.
# Caveat: the teacher-forced probes (Q1–Q8, edge) feed the EXPERT's prev
# input, which this model never saw (always zero) — read them as off-
# distribution; the closed-loop numbers are the arm's read.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue24; do sleep 60; done
echo "== queue 24 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
EVAL_ABLATE=1 run evt2ctx_ck8_off3_dur8e_pd100 --prev-action --prev-action-dropout 1.0 --prev-action-quantize "${ev[@]}" \
  --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_pd100
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_pd50 evt2ctx_ck8_off3_dur8e_pd100
echo "QUEUE 25 DONE ($(date +%H:%M))"
