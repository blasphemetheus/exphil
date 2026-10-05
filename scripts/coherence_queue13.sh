#!/usr/bin/env bash
# Queue 13 (2026-10-05): event CONTEXT — the previous input as a head feature
# on top of the working windowed recipe (events + chunk 8).
#
# Mechanism (interp_recovery_probe on evt2_ck8): the stick-up decision
# offstage reads height (y ablation 0.30 -> 0.13) and fires 7-20x above base
# on the expert's frames, but the BUTTON head cannot see the stick at all —
# the event heads only use the previous input as a selector — so P(B press)
# is flat across stick zones (0.011-0.025) where the expert's spans 40x
# (0.001 neutral .. 0.040 up). Live: B with a neutral stick 20-28 % of
# offstage presses (expert 2.5 %) = lasers; no up-B in 94 % of death sequences.
#
# Pass criteria (vs evt2_ck8 at the same seed/updates):
#   probe Q2c  P(B press | prev stick neutral) <= 0.004 and | up >= 0.03
#   recovery_means  high-band return >= 0.6 (0.28), mismatch <= 0.15, airdodge_with_jump <= 0.10
#   coherence pass criteria kept: repeat >= 0.6, L-cancel >= 0.7, drill >= 0.25 (0.43 now)
#   b_press_stick on sim/live replays: neutral-stick share <= 5 %
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
MAX_FILES=200 EVALS="coherence closed_loop recovery_probe" run evt2ctx_smoke "${pq[@]}" "${ev[@]}" --event-context --chunk-horizon 8 \
  2>&1 | tee /dev/stderr | grep -q TRAIN_FAILED && { echo "SMOKE FAILED; stopping"; exit 1; }
run evt2ctx_ck8 "${pq[@]}" "${ev[@]}" --event-context --chunk-horizon 8
# the w3 variant (best L-cancel / damage in queues 10-11) with context
run evt2ctx_ck8w3 "${pq[@]}" "${ev[@]}" --event-context --chunk-horizon 8 --chunk-weight 3.0
# the new evals on the queue-10 controls, for the same-table comparison
EVALS="recovery_probe" run evt2_ck8w3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --chunk-weight 3.0
echo "QUEUE 13 DONE ($(date +%H:%M))"
