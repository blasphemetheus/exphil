#!/usr/bin/env bash
# Queue 32 (2026-10-08 13:50) — after the Q9 aim probe (INPUT_COHERENCE
# "10-08 13:45"): the Firefox aim (stick UP once the jump is spent) is
# learned exactly on the expert's decision frames (model P(up) 0.087 vs
# expert 0.088) and fires at 1/4 the rate on the bot's own states. The
# label-side levers are exhausted for the aim; what is left inside
# imitation is DECISION FREQUENCY — under --stick-duration C a silent hold
# consults the stick head once per C frames, and the expert's recovery has
# an input edge every ~3.
#   evt2ctx_ck8_off3_dur4e_on30u_e3   recipe with --stick-duration 4, 3 ep
#   evt2ctx_ck8_off3_dur8e_on30u_e3_s906  seed replication of on30u_e3
# Pass (dur4): loop up-onset >= 7 % / 3 f (recovery_aim_vs_press.js;
# on30u_e3 3.4 %, expert 12.3 %) and Firefox-once-spent >= 0.02 deep, with
# on30u_e3's jump rows (>= 0.15 / 0.20) and band (repeat 0.70-0.80, neutral
# 0.22-0.33, fidelity <= 0.21) kept. Pass (s906): bar + jump rows again.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
EPOCHS=3 run evt2ctx_ck8_off3_dur4e_on30u_e3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 4 --onset-weight 30
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur4e_on30u_e3
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur4e_on30u_e3
node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur4e_on30u_e3
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_e3_s906 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 30
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on30u_e3_s906
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_e3_s906
node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_e3_s906
echo "QUEUE 32 DONE ($(date +%H:%M))"
