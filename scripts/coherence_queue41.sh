#!/usr/bin/env bash
# Queue 41 (2026-10-10 04:35, after queue 40) — replicates + the near-edge
# labels at the ×1 dose.
# Queue 40 read: dag6w (the wide round alone, 142k) = the best closed loop
# of the program (SD/min 0.84, deaths/min 0.93 = the expert's 0.96, return
# 0.83, damage 66, offstage trips 5.6/min vs 6.8–7.6, carried-off share
# 0.19 — the queue-40 target) but the offstage press regressed (aim 5.0 %,
# Firefox 0.003) and carried-off died stayed 0.88; dag3456w (256k) = past
# the dose cliff, worse everywhere. Seed variance (10-09 16:37, 22:53) says
# a single-seed SD/min or band read is noise-sized, so:
#   1. on5u_dag345g_x1_e3 on seed 906 (the queue-39 winner's replicate)
#   2. on5u + r345g + the NEAR-EDGE labels of 6 r6w seeds (~44k; offstage
#      labels of r6w dropped: scripts/dagger_set_keep_near_edge.exs) ≈ 158k —
#      the wide window's addition on top of the three offstage rounds, at ×1
#   3. on5u_dag6w_x1_e3 on seed 906 (the best closed loop's replicate)
# Pass bar = queue 40's.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
until grep -q "QUEUE 40 DONE" logs/exphil-queue40.log; do sleep 120; done
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== queue 40 done, beam free ($(date +%H:%M))"
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec5=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
mf() { elixir -pa '_build/dev/lib/*/ebin' "$@"; }
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on5u_dag345g_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3 "$1"
  node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on5u_dag345g_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3 "$1"
  node scripts/recovery_firefox_angle.js evt2ctx_ck8_off3_dur8e_on5u_dag345g_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3 "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
r345=data/silent_fall/sim_dagger_expert_r345g_split.frames
r6w=data/silent_fall/sim_dagger_expert_r6w_split.frames
# near-edge-only labels of seeds 2091–2096 (two 3-seed parts), re-cut, + r345g
r6n=data/silent_fall/sim_dagger_expert_r6n6_split.frames
r345n=data/silent_fall/sim_dagger_expert_r345g_r6n6_split.frames
parts=()
for p in 2091 2094; do
  mf scripts/dagger_set_keep_near_edge.exs data/silent_fall/sim_dagger_expert_r6w_$p.frames data/silent_fall/sim_dagger_expert_r6n_$p.frames 2>&1 | grep -E "RESULT|rror" | cut -c1-300
  mf scripts/dagger_set_split_gated.exs data/silent_fall/sim_dagger_expert_r6n_$p.frames data/silent_fall/sim_dagger_expert_r6n_${p}_split.frames 2>&1 | grep -E "RESULT|rror" | cut -c1-300
  parts+=(data/silent_fall/sim_dagger_expert_r6n_${p}_split.frames)
done
mf scripts/dagger_set_concat.exs "${parts[@]}" $r6n 2>&1 | grep -E "RESULT|rror" | cut -c1-300
mf scripts/dagger_set_concat.exs $r345 $r6n $r345n 2>&1 | grep -E "RESULT|rror" | cut -c1-300
[ -s $r345n ] || { echo "SET_FAILED (no r345g_r6n6)"; exit 1; }
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on5u_dag345g_x1_e3_s906 "${pq[@]}" "${ev[@]}" "${rec5[@]}" --mix-frames $r345 --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_on5u_dag345g_x1_e3_s906
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on5u_dag345g_r6n6_x1_e3 "${pq[@]}" "${ev[@]}" "${rec5[@]}" --mix-frames $r345n --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_on5u_dag345g_r6n6_x1_e3
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906 "${pq[@]}" "${ev[@]}" "${rec5[@]}" --mix-frames $r6w --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906
echo "QUEUE 41 DONE ($(date +%H:%M))"
