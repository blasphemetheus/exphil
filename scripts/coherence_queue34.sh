#!/usr/bin/env bash
# Queue 34 (2026-10-08 18:50, after queue 33) — expert-labeled DAgger,
# ROUND 2 at 4x the rollout. Round 1 (queue 33, set r1: 3 seeds, 18,619
# relabelled frames) carried the expert's hazards on the bot's states
# (stick-up onset once spent 3.08 % vs expert 3.1 %) but only 71 up-onsets
# and 152 jump edges in total; dagx2_e3 moved the closed loop (return rate
# 0.814, jump-in-hand deaths 12 %) and the aim only 3.4 -> 3.9 %. Dose, not
# mechanism, is the reading. Round 2: roll the NEWEST policy (arg 1) out on
# 12 seeds (= 4x set r1, ~25 min), relabel, train on r1 + r2 at oversample 4.
#   usage: scripts/coherence_queue34.sh checkpoints/coh_X/model_policy.bin
# Pass (unchanged from queue 33): loop up-onset once spent >= 7 % / 3 f AND
# Firefox-once-spent >= 0.04 at <= -40, jump rows >= 0.15 / 0.20, band
# (repeat 0.70-0.80, neutral 0.22-0.33, fidelity <= 0.21), closed-loop
# rates kept (SD/min <= 1.4, dashes/min >= 22). Set hold share 0.76-0.80.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pol=${1:?policy path}
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M)); rollout policy $pol"
r2=data/silent_fall/sim_dagger_expert_r2.frames
both=data/silent_fall/sim_dagger_expert_r1r2.frames
echo "== rollout + relabel, 12 seeds ($(date +%H:%M))"
mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index data/silent_fall/expert_recovery_index.bin \
  --seeds 2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 --label-seed 8 \
  --out $r2 --report eval_runs/1001_queue/sim_dagger_expert_r2.json 2>&1 | grep -E "RESULT|seed |error|Error|\*\*" | cut -c1-500
[ -s eval_runs/1001_queue/sim_dagger_expert_r2.json ] || { echo "DAGGER_FAILED (no report)"; exit 1; }
hold=$(node -e 'const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/sim_dagger_expert_r2.json"));console.log(r.hold_share ?? "nan")')
echo "== set r2 hold share: $hold"
elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_label_hazards.exs $r2 2>&1 | grep RESULT | cut -c1-300
# r1 + r2 in one file (same export shape; frame_lists concatenated)
elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_concat.exs data/silent_fall/sim_dagger_expert_r1.frames $r2 $both 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $both ] || { echo "DAGGER_FAILED (no concat)"; exit 1; }
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 30)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag2x4_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $both --mix-oversample 4
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on30u_dag2x4_e3
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag2x4_e3
node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_e3 evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag2x4_e3
echo "QUEUE 34 DONE ($(date +%H:%M))"
