#!/usr/bin/env bash
# Queue 35 (2026-10-08 22:25, after queue 34) — two readings from dag2x4_e3
# ("10-08 22:12": aim half PASSES at 5x dose — up-onset 8.3 %, stick-up share
# 42 % = expert — but P(B|up) 4.9 %, fidelity 0.278, damage/dashes collapse,
# airdodge_with_jump 0.316):
#   A. the DOSE MIDDLE on the same r1+r2 set (v1 labels): oversample 1 and 2
#      (dagx4 = 74k effective frames, 4.6 %; dag2x4 = 268k, 8.3 % + damage)
#   B. the COVERAGE GATE: labeler v2 (velocity = position delta; v1's
#      speed_y_self was all-zero on the replay side — Slippi < 3.5) + relabel
#      ONLY states within the expert's own q95 nearest-row distance (d2 <=
#      0.78 on 24 held-out games): 12 seeds from dagx4_e3 in 4 beams -> r3
#      (gated frames stay as input-only context) -> on30u_dag3g_x4_e3.
# Pass bar unchanged: up-onset >= 7 %/3 f AND Firefox-once-spent >= 0.04 at
# <= -40 with jump rows >= 0.15/0.20, band (repeat 0.70-0.80, neutral
# 0.22-0.33, fidelity <= 0.21), closed-loop rates kept (SD/min <= 1.4,
# dashes/min >= 22, damage/min >= 45).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 30)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
both=data/silent_fall/sim_dagger_expert_r1r2.frames
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag2x4_e3 "$1"
  node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag2x4_e3 "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
# A. dose middle
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag2x2_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $both --mix-oversample 2
readout evt2ctx_ck8_off3_dur8e_on30u_dag2x2_e3
# B. gated v2 rollout from dagx4_e3
pol=checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3/model_policy.bin
idx=data/silent_fall/expert_recovery_index_v2.bin
gate=0.78
r3=data/silent_fall/sim_dagger_expert_r3g.frames
echo "== gated rollout + relabel (v2 index, d2 <= $gate), 12 seeds in 4 beams ($(date +%H:%M))"
parts=()
for grp in 2031,2032,2033 2034,2035,2036 2037,2038,2039 2040,2041,2042; do
  part=data/silent_fall/sim_dagger_expert_r3g_${grp%%,*}.frames
  mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index $idx --max-d2 $gate \
    --seeds $grp --label-seed 9 --out $part --report eval_runs/1001_queue/sim_dagger_expert_r3g_${grp%%,*}.json \
    > logs/dagger_r3g_${grp%%,*}.log 2>&1
  grep -E "RESULT|seed |RESOURCE_EXHAUSTED|\*\*" logs/dagger_r3g_${grp%%,*}.log | cut -c1-400
  [ -s eval_runs/1001_queue/sim_dagger_expert_r3g_${grp%%,*}.json ] || { echo "DAGGER_FAILED (seeds $grp, see logs/dagger_r3g_${grp%%,*}.log)"; exit 1; }
  parts+=("$part")
done
elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_concat.exs "${parts[@]}" $r3 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r3 ] || { echo "DAGGER_FAILED (no r3 concat)"; exit 1; }
elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_label_hazards.exs $r3 2>&1 | grep RESULT | cut -c1-300
mix run --no-compile scripts/expert_labeler_distance.exs --split data/silent_fall/heldout_fd_fox_split.json --index $idx --games 24 \
  --set $r3 --out eval_runs/1001_queue/labeler_distance_v2_r3g.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-400
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r3 --mix-oversample 4
readout evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3
# A'. oversample 1 last (the least informative of the three)
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag2x1_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $both --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_on30u_dag2x1_e3
echo "QUEUE 35 DONE ($(date +%H:%M))"
