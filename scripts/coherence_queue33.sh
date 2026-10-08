#!/usr/bin/env bash
# Queue 33 (2026-10-08 14:10, after queue 32) — EXPERT-LABELED DAGGER on the
# bot's own offstage states (Bradley's ok, 10-08). Q9 ("10-08 13:45"): the
# Firefox aim is calibrated exactly on the expert's decision frames (P(up)
# 0.087 vs 0.088) and fires at 1/4 the rate on the bot's own states — a
# coverage problem, so the labels go where the bot is. Labeler = the expert
# distribution itself (scripts/lib/expert_recovery_labeler.exs: k-NN over
# 214k expert offstage frames from 176 training-split FD games, sampled),
# not the 10-05 rules expert whose style/mistakes sank lever 2.
#   1. roll on30u_e3 out in the sim (3 seeds x 32 envs x 3600 f, self-play),
#      relabel every offstage airborne frame (sequential inside a trip, prev =
#      previous label), export with 90 input-only context frames per trip
#      -> data/silent_fall/sim_dagger_expert_r1.frames; print hold share
#      (corpus 0.785) and the label mix (index: B 18.9 %, jump 11.1 %, up 28.4 %)
#   2. evt2ctx_ck8_off3_dur8e_on30u_dagx2_e3  recipe + --mix-frames (oversample 2), 3 ep
#   3. evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3  oversample 4 (dose)
# Pass: loop up-onset once spent >= 7 % / 3 f (recovery_aim_vs_press.js;
# on30u_e3 3.4 %, expert 12.3 %) AND Firefox-once-spent >= 0.04 at <= -40
# (DecisionMap; on30u_e3 0.004) with on30u_e3's jump rows (>= 0.15 / 0.20),
# band (repeat 0.70-0.80, neutral 0.22-0.33, fidelity <= 0.21) and closed-loop
# rates (SDs/min, dashes/min — the 10-05 style damage) kept. The set's hold
# share must be ~0.76-0.80 before any arm trains (10-05 lesson).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
set_=data/silent_fall/sim_dagger_expert_r1.frames
# 0. labeler self-check on held-out expert games (EXLA; the BinaryBackend
#    brute force over 214k rows is too slow): label hazards must be within
#    x0.5-x2 of the expert's own on the same frames, else no arm trains
echo "== labeler self-check ($(date +%H:%M))"
mix run --no-compile scripts/expert_labeler_selfcheck.exs --split checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_e3/split.json \
  --index data/silent_fall/expert_recovery_index.bin --games 24 --out eval_runs/1001_queue/labeler_selfcheck.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-400
ok=$(node -e 'try{const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_selfcheck.json"));const w=(a,b)=>b>0&&a/b>=0.5&&a/b<=2.0;console.log(w(r.label_up,r.actual_up)&&w(r.label_jump,r.actual_jump)&&w(r.label_b,r.actual_b)?"yes":"no")}catch(e){console.log("no")}')
echo "== labeler within x0.5-x2 of the expert: $ok"
[ "$ok" = yes ] || { echo "LABELER_FAILED"; exit 1; }
pol=checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_e3/model_policy.bin
echo "== rollout + relabel ($(date +%H:%M))"
mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index data/silent_fall/expert_recovery_index.bin \
  --out $set_ --report eval_runs/1001_queue/sim_dagger_expert_r1.json 2>&1 | grep -E "RESULT|seed |error|Error|\*\*" | cut -c1-500
[ -s $set_ ] || { echo "DAGGER_FAILED (no set)"; exit 1; }
hold=$(node -e 'const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/sim_dagger_expert_r1.json"));console.log(r.hold_share ?? "nan")')
echo "== set hold share: $hold"
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 30)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dagx2_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $set_ --mix-oversample 2
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on30u_dagx2_e3
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_dagx2_e3
node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_dagx2_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $set_ --mix-oversample 4
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3
node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3
echo "QUEUE 33 DONE ($(date +%H:%M))"
