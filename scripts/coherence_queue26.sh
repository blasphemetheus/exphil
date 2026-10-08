#!/usr/bin/env bash
# Queue 26 (2026-10-07 23:30, overnight, after queue 25) — no-code arms on the
# dur8e recipe while the danger-readout head is being written (queue 27).
# The prev-action dose series (pd15 / pd50 / pd100, doc "10-07 20:30") said:
# dropout 0.15 cures the silent fall, 0.5 over-cures it (neutral 0.097),
# the channel must stay for coherence. Two questions left for the recipe:
#   evt2ctx_ck8_off3_dur8e_pd30     dropout 0.3, 1 ep: is there a dose that
#                                   keeps neutral in band (0.22-0.33) with
#                                   died-holding-neutral <= 20 %? (pd15: 0.141
#                                   / 10 %; pd50: 0.097 / 13 %)
#   evt2ctx_ck8_off3_dur8e_pd15_e3  dropout 0.15 at 3 ep: the port-recipe
#                                   candidate (dur8e_e3 passed: fidelity 0.175,
#                                   repeat 0.759, neutral 0.221) WITH the
#                                   silent-fall cure. Pass = the port bar
#                                   (fidelity <= 0.21, repeat 0.70-0.80, neutral
#                                   0.22-0.33) AND offstage age 1-3 enter-silence
#                                   <= 0.035 AND died holding neutral <= 20 %.
#                                   If it passes, pd15 joins the port recipe.
# DecisionMap rows are expected NOT to move (conditioning defects, not channel
# defects) — that is queue 27's job; here they are the control.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue25; do sleep 60; done
echo "== queue 25 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off3_dur8e_pd30 --prev-action --prev-action-dropout 0.3 --prev-action-quantize "${ev[@]}" \
  --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_pd30
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_pd15_e3 --prev-action --prev-action-dropout 0.15 --prev-action-quantize "${ev[@]}" \
  --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_pd15_e3
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_pd30 evt2ctx_ck8_off3_dur8e_pd15_e3
echo "QUEUE 26 DONE ($(date +%H:%M))"
