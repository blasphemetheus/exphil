#!/usr/bin/env bash
# Queue 37 (2026-10-09 04:10, after queue 36) — from the dag3g_x4_e3 read
# ("03:34": the coverage gate is the lever — up-onset 9.7 %, P(B|up) 14.3 %,
# closed loop kept; Firefox-once-spent still 0.003-0.006 vs bar 0.04) and the
# label-side finding behind it: 28 % of r3g's labelled B edges once spent
# start INSIDE AN AERIAL (the labeler had no attacking dim; the expert's own
# 11 %), and the bot is helpless on ~60 % of its spent-low samples.
#   1. labeler v3 (attacking dim; helpless 36/37 not labelable): self-check on
#      the 24 held-out games (same x0.5-x2 gate as queue 33) and its held-out
#      q95 = the coverage gate for v3.
#   2. gated v3 round r4g3 from dag3g_x4_e3's own states (12 seeds, 4 beams),
#      re-cut for the loader (dagger_set_split_gated.exs), hazards + coverage.
#   3. arms: dag4g3_x4_e3 (r4g3 x4), dag3g_x2_e3 (dose middle on the gated
#      r3g), dag34g_x2_e3 (r3g + r4g3 x2).
# Pass bar unchanged: up-onset >= 7 %/3 f AND Firefox-once-spent >= 0.04 at
# <= -40 with jump rows >= 0.15/0.20, band (repeat 0.70-0.80, neutral
# 0.22-0.33, fidelity <= 0.21), SD/min <= 1.4, dashes/min >= 22, damage/min >= 45.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 30)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
mf() { elixir -pa '_build/dev/lib/*/ebin' "$@"; }
idx=data/silent_fall/expert_recovery_index_v3.bin
held=data/silent_fall/heldout_fd_fox_split.json
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3 "$1"
  node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3 "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
# 1. labeler v3 self-check + gate
[ -s $idx ] || { echo "LABELER_FAILED (no v3 index)"; exit 1; }
echo "== labeler v3 self-check ($(date +%H:%M))"
mf scripts/expert_labeler_selfcheck.exs --split $held --index $idx --games 24 --out eval_runs/1001_queue/labeler_selfcheck_v3.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-400
ok=$(node -e 'try{const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_selfcheck_v3.json"));const w=(a,b,n)=>b*n<20||(b>0&&a/b>=0.5&&a/b<=2.0);console.log(r.n>=10000&&w(r.label_up,r.actual_up,r.n_spent)&&w(r.label_jump,r.actual_jump,r.n_in_hand)&&w(r.label_b,r.actual_b,r.n_spent)?"yes":"no")}catch(e){console.log("no")}')
echo "== labeler v3 within x0.5-x2 of the expert: $ok"
[ "$ok" = yes ] || { echo "LABELER_FAILED"; exit 1; }
mf scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --out eval_runs/1001_queue/labeler_distance_v3_heldout.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-400
gate=$(node -e 'const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_distance_v3_heldout.json"));console.log(r.gate_q95?r.gate_q95.toFixed(3):"")')
[ -n "$gate" ] || { echo "LABELER_FAILED (no gate)"; exit 1; }
echo "== v3 coverage gate d2 <= $gate (held-out q95)"
# 2. gated v3 round from dag3g_x4_e3
pol=checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3/model_policy.bin
r4=data/silent_fall/sim_dagger_expert_r4g3.frames
echo "== gated v3 rollout + relabel from dag3g_x4_e3 (d2 <= $gate), 12 seeds in 4 beams ($(date +%H:%M))"
parts=()
for grp in 2051,2052,2053 2054,2055,2056 2057,2058,2059 2060,2061,2062; do
  part=data/silent_fall/sim_dagger_expert_r4g3_${grp%%,*}.frames
  mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index $idx --max-d2 $gate \
    --seeds $grp --label-seed 10 --out $part --report eval_runs/1001_queue/sim_dagger_expert_r4g3_${grp%%,*}.json \
    > logs/dagger_r4g3_${grp%%,*}.log 2>&1
  grep -E "RESULT|seed |RESOURCE_EXHAUSTED|\*\*" logs/dagger_r4g3_${grp%%,*}.log | cut -c1-400
  [ -s eval_runs/1001_queue/sim_dagger_expert_r4g3_${grp%%,*}.json ] || { echo "DAGGER_FAILED (seeds $grp, see logs/dagger_r4g3_${grp%%,*}.log)"; exit 1; }
  parts+=("$part")
done
mf scripts/dagger_set_concat.exs "${parts[@]}" $r4 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r4 ] || { echo "DAGGER_FAILED (no r4g3 concat)"; exit 1; }
mf scripts/dagger_set_label_hazards.exs $r4 2>&1 | grep RESULT | cut -c1-300
mf scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --set $r4 --out eval_runs/1001_queue/labeler_distance_v3_r4g3.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-400
r4s=data/silent_fall/sim_dagger_expert_r4g3_split.frames
mf scripts/dagger_set_split_gated.exs $r4 $r4s 2>&1 | grep RESULT | cut -c1-300
[ -s $r4s ] || { echo "DAGGER_FAILED (no r4g3 split)"; exit 1; }
r3s=data/silent_fall/sim_dagger_expert_r3g_split.frames
r34=data/silent_fall/sim_dagger_expert_r34g_split.frames
mf scripts/dagger_set_concat.exs $r3s $r4s $r34 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
# 3. arms
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag4g3_x4_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r4s --mix-oversample 4
readout evt2ctx_ck8_off3_dur8e_on30u_dag4g3_x4_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag3g_x2_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r3s --mix-oversample 2
readout evt2ctx_ck8_off3_dur8e_on30u_dag3g_x2_e3
[ -s $r34 ] && { EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r34 --mix-oversample 2; readout evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3; }
echo "QUEUE 37 DONE ($(date +%H:%M))"
