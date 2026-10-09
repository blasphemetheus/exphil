#!/usr/bin/env bash
# Queue 38 (2026-10-09 11:10, after queue 37) — from the dag34g_x2_e3 read
# ("11:01": aim at the expert — up-onset 12.1 %, stick-up share 48 %, P(B|up)
# 20 %; Firefox-once-spent 0.020 at -40..-60 = half the bar; band at its edge
# at 150k effective frames: neutral 0.192, fidelity 0.221, dashes 21.1).
#   1. gated v3 round r5g3 from dag34g_x2_e3's own states (12 seeds, 4 beams,
#      v3 gate 0.814), re-cut for the loader, hazards + coverage.
#   2. arms: dag345g_x1_e3 (r3g + r4g3 + r5g3 at x1, ~115k relabelled — the
#      dose that kept the band on r1+r2), dag345g_x2_e3, and the seed
#      replicate dag34g_x2_e3_s906 (the program's single-seed caveat).
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
# GPU-side labeler reads need EXLA as the default backend = the mix config (queue 37 lesson)
mx() { mix run --no-compile "$@"; }
idx=data/silent_fall/expert_recovery_index_v3.bin
held=data/silent_fall/heldout_fd_fox_split.json
gate=0.814
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3 "$1"
  node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3 "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
# 1. gated v3 round from dag34g_x2_e3
pol=checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3/model_policy.bin
r5=data/silent_fall/sim_dagger_expert_r5g3.frames
echo "== gated v3 rollout + relabel from dag34g_x2_e3 (d2 <= $gate), 12 seeds in 4 beams ($(date +%H:%M))"
parts=()
for grp in 2071,2072,2073 2074,2075,2076 2077,2078,2079 2080,2081,2082; do
  part=data/silent_fall/sim_dagger_expert_r5g3_${grp%%,*}.frames
  mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index $idx --max-d2 $gate \
    --seeds $grp --label-seed 11 --out $part --report eval_runs/1001_queue/sim_dagger_expert_r5g3_${grp%%,*}.json \
    > logs/dagger_r5g3_${grp%%,*}.log 2>&1
  grep -E "RESULT|seed |RESOURCE_EXHAUSTED|\*\*" logs/dagger_r5g3_${grp%%,*}.log | cut -c1-400
  [ -s eval_runs/1001_queue/sim_dagger_expert_r5g3_${grp%%,*}.json ] || { echo "DAGGER_FAILED (seeds $grp, see logs/dagger_r5g3_${grp%%,*}.log)"; exit 1; }
  parts+=("$part")
done
mf scripts/dagger_set_concat.exs "${parts[@]}" $r5 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r5 ] || { echo "DAGGER_FAILED (no r5g3 concat)"; exit 1; }
mf scripts/dagger_set_label_hazards.exs $r5 2>&1 | grep RESULT | cut -c1-300
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --set $r5 --out eval_runs/1001_queue/labeler_distance_v3_r5g3.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-400
r5s=data/silent_fall/sim_dagger_expert_r5g3_split.frames
mf scripts/dagger_set_split_gated.exs $r5 $r5s 2>&1 | grep RESULT | cut -c1-300
[ -s $r5s ] || { echo "DAGGER_FAILED (no r5g3 split)"; exit 1; }
r34=data/silent_fall/sim_dagger_expert_r34g_split.frames
r345=data/silent_fall/sim_dagger_expert_r345g_split.frames
mf scripts/dagger_set_concat.exs $r34 $r5s $r345 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r345 ] || { echo "DAGGER_FAILED (no r345g concat)"; exit 1; }
# 2. arms
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag345g_x1_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r345 --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_on30u_dag345g_x1_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag345g_x2_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r345 --mix-oversample 2
readout evt2ctx_ck8_off3_dur8e_on30u_dag345g_x2_e3
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3_s906 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r34 --mix-oversample 2
readout evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3_s906
echo "QUEUE 38 DONE ($(date +%H:%M))"
