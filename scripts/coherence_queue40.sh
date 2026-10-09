#!/usr/bin/env bash
# Queue 40 (2026-10-09 18:50, after queue 39) — the state BEFORE offstage.
# Every readout since 10-08 has the sim's carried-off share at 0.27–0.35 vs
# the expert's 0.055 (carried-off died 0.87 vs 0.16): ground side-B off the
# edge, airdodge off the edge, no DI — states the v3 labeler's window
# (airborne past the ledge) never labels. Bradley (10-09): "there's probably
# another failure state before that we can make better". Labeler v4 = the
# `:wide` window (scripts/lib/expert_recovery_labeler.exs): grounded or
# airborne within 15 units of the edge, + grounded/shield/dash dims (grounded
# weight 3 so the gate keeps grounded queries on grounded rows). Same
# mechanism, same expert, same gate (held-out q95), one gated round from
# the queue-39 winner (eval_runs/1001_queue/queue40_winner.sh).
#   1. index v4 (--window wide) + self-check + gate
#   2. gated v4 round r6w from the winner's own states (12 seeds, 4 beams)
#   3. arms: r345g + r6w at ×1 (the compounding line), r6w alone at ×1
#      (does one wide round carry what three offstage rounds did?)
# Pass = queue 39's bar (press, angle, band, closed loop) AND the upstream
# number moves: carried-off share <= 0.20 (0.35; expert 0.055), carried-off
# died <= 0.60 (0.87; expert 0.16), airdodge_with_jump <= 0.06 (0.12; expert
# 0.006), side-B deaths by move down from 40.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
until grep -q "QUEUE 39 DONE" logs/exphil-queue39.log; do sleep 120; done
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== queue 39 done, beam free ($(date +%H:%M))"
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
source eval_runs/1001_queue/queue40_winner.sh
[ -s "$pol" ] || { echo "NO_WINNER_POLICY $pol"; exit 1; }
echo "== winner: $pol (tag $tag, knob: ${knob[*]:-none})"
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec0=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
mf() { elixir -pa '_build/dev/lib/*/ebin' "$@"; }
# GPU-side labeler reads need EXLA as the default backend = the mix config (queue 37 lesson)
mx() { mix run --no-compile "$@"; }
idx=data/silent_fall/expert_recovery_index_v4.bin
held=data/silent_fall/heldout_fd_fox_split.json
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_dag345g_x1_e3 evt2ctx_ck8_off3_dur8e_${tag}_dag345g_x1_e3 "$1"
  node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_dag345g_x1_e3 evt2ctx_ck8_off3_dur8e_${tag}_dag345g_x1_e3 "$1"
  node scripts/recovery_firefox_angle.js evt2ctx_ck8_off3_dur8e_e3 evt2ctx_ck8_off3_dur8e_${tag}_dag345g_x1_e3 "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
# 1. index v4 (wide window) + self-check + gate
echo "== index v4, wide window ($(date +%H:%M))"
mx scripts/build_expert_recovery_index.exs --split checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_e3/split.json --games 400 --window wide --out $idx 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-400
[ -s $idx ] || { echo "LABELER_FAILED (no v4 index)"; exit 1; }
mx scripts/expert_labeler_selfcheck.exs --split $held --index $idx --games 24 --out eval_runs/1001_queue/labeler_selfcheck_v4.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-400
ok=$(node -e 'try{const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_selfcheck_v4.json"));const w=(a,b,n)=>b*n<20||(b>0&&a/b>=0.5&&a/b<=2.0);console.log(r.n>=10000&&w(r.label_up,r.actual_up,r.n_spent)&&w(r.label_jump,r.actual_jump,r.n_in_hand)&&w(r.label_b,r.actual_b,r.n_spent)?"yes":"no")}catch(e){console.log("no")}')
echo "== labeler v4 within x0.5-x2 of the expert: $ok"
[ "$ok" = yes ] || { echo "LABELER_FAILED"; exit 1; }
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --out eval_runs/1001_queue/labeler_distance_v4_heldout.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-400
gate=$(node -e 'const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_distance_v4_heldout.json"));console.log(r.gate_q95?r.gate_q95.toFixed(3):"")')
[ -n "$gate" ] || { echo "LABELER_FAILED (no gate)"; exit 1; }
echo "== v4 coverage gate d2 <= $gate (held-out q95)"
# 2. gated v4 round from the winner (wide window: offstage AND near-edge states)
r6=data/silent_fall/sim_dagger_expert_r6w.frames
echo "== gated v4 rollout + relabel from $tag (d2 <= $gate), 12 seeds in 4 beams ($(date +%H:%M))"
parts=()
for grp in 2091,2092,2093 2094,2095,2096 2097,2098,2099 2100,2101,2102; do
  part=data/silent_fall/sim_dagger_expert_r6w_${grp%%,*}.frames
  mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index $idx --max-d2 $gate \
    --seeds $grp --label-seed 12 --out $part --report eval_runs/1001_queue/sim_dagger_expert_r6w_${grp%%,*}.json \
    > logs/dagger_r6w_${grp%%,*}.log 2>&1
  grep -E "RESULT|seed |RESOURCE_EXHAUSTED|\*\*" logs/dagger_r6w_${grp%%,*}.log | cut -c1-400
  [ -s eval_runs/1001_queue/sim_dagger_expert_r6w_${grp%%,*}.json ] || { echo "DAGGER_FAILED (seeds $grp, see logs/dagger_r6w_${grp%%,*}.log)"; exit 1; }
  parts+=("$part")
done
mf scripts/dagger_set_concat.exs "${parts[@]}" $r6 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r6 ] || { echo "DAGGER_FAILED (no r6w concat)"; exit 1; }
mf scripts/dagger_set_label_hazards.exs $r6 2>&1 | grep RESULT | cut -c1-300
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --set $r6 --out eval_runs/1001_queue/labeler_distance_v4_r6w.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-400
r6s=data/silent_fall/sim_dagger_expert_r6w_split.frames
mf scripts/dagger_set_split_gated.exs $r6 $r6s 2>&1 | grep RESULT | cut -c1-300
[ -s $r6s ] || { echo "DAGGER_FAILED (no r6w split)"; exit 1; }
r345=data/silent_fall/sim_dagger_expert_r345g_split.frames
r3456=data/silent_fall/sim_dagger_expert_r3456w_split.frames
mf scripts/dagger_set_concat.exs $r345 $r6s $r3456 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r3456 ] || { echo "DAGGER_FAILED (no r3456w concat)"; exit 1; }
# 3. arms (the winner's recipe: rec0 + its onset knob)
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag3456w_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --mix-frames $r3456 --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag3456w_x1_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag6w_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --mix-frames $r6s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag6w_x1_e3
echo "QUEUE 40 DONE ($(date +%H:%M))"
