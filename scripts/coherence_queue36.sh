#!/usr/bin/env bash
# Queue 36 (2026-10-09 00:50) — rerun queue 35's arm B. dag3g_x4_e3 died at
# the loader: the --max-d2 gate marks frames input-only in the MIDDLE of a
# trip and Data.from_frame_lists/2 requires [input-only prefix, targets]
# ("input-only frames must precede a nonempty target suffix"). The set was
# re-cut with scripts/dagger_set_split_gated.exs (one list per maximal
# target run, the trip's earlier frames as input-only context): 1821 ->
# 2421 lists, 32,533 targets unchanged. Same pass bar as queue 35.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
until grep -q "QUEUE 35 DONE" logs/exphil-queue35.log; do sleep 60; done
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 30)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
r3=data/silent_fall/sim_dagger_expert_r3g_split.frames
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag2x2_e3 "$1"
  node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_dagx4_e3 evt2ctx_ck8_off3_dur8e_on30u_dag2x2_e3 "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
rm -rf eval_runs/1001_queue/evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3 checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --mix-frames $r3 --mix-oversample 4
readout evt2ctx_ck8_off3_dur8e_on30u_dag3g_x4_e3
echo "QUEUE 36 DONE ($(date +%H:%M))"
