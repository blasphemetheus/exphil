#!/usr/bin/env bash
# Queue 45 (2026-10-10 18:40) — labeler v5 `:air` window: label the drift
# before the Illusion. INPUT_COHERENCE "18:35": on every arm the carried-off
# side-B trip starts with the double jump spent 45–67 units INSIDE the edge
# mid full-jump with the stick held outward (61–87 %), 24–30 f before the
# trip; the expert spends its jump at the edge line (dist q50 0.1, stick
# toward 25 %). The double-jump hazard is the expert's (0.62 vs 0.60 %/f);
# the stick at the spend is 2–3x more outward. The drift cell (airborne over
# the stage, no jump, −80..−15) is ~20k bot frames per round and 1–25 %
# labelled — outside the wide window. `:air` = wide ∪ that cell
# (scripts/lib/expert_recovery_labeler.exs). Chain = queue 40/42: index v5 +
# self-check + held-out gate, THEN the label-content gate (the expert's label
# in the bot's sub-cell — stick outward, moving out — must differ from "keep
# outward" on >= 25 % of rows, else the lever is empty: stop), r8a roll from
# the on5u winner (12 seeds, 4 beams, --keep-actual), concat / hazards /
# split, dose count (labelled drift-cell frames), two arms r8a x1.
# Pass (both seeds): carried side-B trips <= 5 per recovery_means run
# (dag7w 22, dag6w 13 / 20, edge15 14 / 14), carried-off share <= 0.15
# (0.20–0.32 today, expert 0.055), band / up-onset / angle held, Firefox
# −40..−60 row not worse; recovery_jump_spend.js: spend dist q50 moves toward
# the edge line.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
source eval_runs/1001_queue/queue40_winner.sh   # pol, tag=on5u, knob
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec0=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
mf() { elixir -pa '_build/dev/lib/*/ebin' "$@"; }
# GPU-side labeler reads need EXLA as the default backend = the mix config (queue 37 lesson)
mx() { mix run --no-compile "$@"; }
ctl=(evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906)
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_sideb_trips.js "${ctl[@]}" "$1"
  node scripts/recovery_jump_spend.js "${ctl[@]}" "$1"
  node scripts/recovery_jump_hazard.js "${ctl[@]}" "$1"
  node scripts/recovery_aim_vs_press.js "${ctl[@]}" "$1"
  node scripts/recovery_firefox_angle.js "${ctl[@]}" "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
idx=data/silent_fall/expert_recovery_index_v5.bin
held=data/silent_fall/heldout_fd_fox_split.json
# 1. index v5 (air window) + self-check + gate
echo "== index v5, air window ($(date +%H:%M))"
mx scripts/build_expert_recovery_index.exs --split checkpoints/coh_evt2ctx_ck8_off3_dur8e_on30u_e3/split.json --games 400 --window air --out $idx 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-400
[ -s $idx ] || { echo "LABELER_FAILED (no v5 index)"; exit 1; }
mx scripts/expert_labeler_selfcheck.exs --split $held --index $idx --games 24 --out eval_runs/1001_queue/labeler_selfcheck_v5.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-400
ok=$(node -e 'try{const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_selfcheck_v5.json"));const w=(a,b,n)=>b*n<20||(b>0&&a/b>=0.5&&a/b<=2.0);console.log(r.n>=10000&&w(r.label_up,r.actual_up,r.n_spent)&&w(r.label_jump,r.actual_jump,r.n_in_hand)&&w(r.label_b,r.actual_b,r.n_spent)?"yes":"no")}catch(e){console.log("no")}')
echo "== labeler v5 within x0.5-x2 of the expert: $ok"
[ "$ok" = yes ] || { echo "LABELER_FAILED"; exit 1; }
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --out eval_runs/1001_queue/labeler_distance_v5_heldout.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-400
gate=$(node -e 'const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_distance_v5_heldout.json"));console.log(r.gate_q95?r.gate_q95.toFixed(3):"")')
[ -n "$gate" ] || { echo "LABELER_FAILED (no gate)"; exit 1; }
echo "== v5 coverage gate d2 <= $gate (held-out q95)"
# 1b. label-content gate: what the expert does in the bot's sub-cell
mf scripts/expert_index_drift_cell.exs $idx 2>&1 | cut -c1-400
notout=$(mf scripts/expert_index_drift_cell.exs $idx 2>/dev/null | grep -o "label not outward [0-9.]*" | grep -o "[0-9.]*$")
echo "== drift-cell label content: not-outward share $notout % (gate >= 25)"
node -e 'process.exit(parseFloat(process.argv[1]) >= 25 ? 0 : 1)' "${notout:-0}" || { echo "LEVER_EMPTY (the expert's label keeps the stick outward in the drift too)"; exit 1; }
# 2. r8a: gated v5 round from the on5u winner, the bot's own input kept
r8=data/silent_fall/sim_dagger_expert_r8a.frames
r8s=data/silent_fall/sim_dagger_expert_r8a_split.frames
echo "== gated v5 rollout + relabel from $tag (d2 <= $gate), 12 seeds in 4 beams, --keep-actual ($(date +%H:%M))"
parts=()
for grp in 2091,2092,2093 2094,2095,2096 2097,2098,2099 2100,2101,2102; do
  part=data/silent_fall/sim_dagger_expert_r8a_${grp%%,*}.frames
  mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index $idx --max-d2 $gate --keep-actual \
    --seeds $grp --label-seed 12 --out $part --report eval_runs/1001_queue/sim_dagger_expert_r8a_${grp%%,*}.json \
    > logs/dagger_r8a_${grp%%,*}.log 2>&1
  grep -E "RESULT|seed |RESOURCE_EXHAUSTED|\*\*" logs/dagger_r8a_${grp%%,*}.log | cut -c1-400
  [ -s eval_runs/1001_queue/sim_dagger_expert_r8a_${grp%%,*}.json ] || { echo "DAGGER_FAILED (seeds $grp, see logs/dagger_r8a_${grp%%,*}.log)"; exit 1; }
  parts+=("$part")
done
mf scripts/dagger_set_concat.exs "${parts[@]}" $r8 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r8 ] || { echo "DAGGER_FAILED (no r8a concat)"; exit 1; }
mf scripts/dagger_set_label_hazards.exs $r8 2>&1 | grep RESULT | cut -c1-300
mf scripts/dagger_set_split_gated.exs $r8 $r8s 2>&1 | grep RESULT | cut -c1-300
[ -s $r8s ] || { echo "DAGGER_FAILED (no r8a split)"; exit 1; }
# 2b. dose: labelled frames in the drift cell (the lever's examples), on the unsplit set
mf scripts/dagger_set_drift_cell.exs $r8 2>&1 | grep -E "RESULT|rror" | cut -c1-300
# 3. arms
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag8a_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --mix-frames $r8s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag8a_x1_e3
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag8a_x1_e3_s906 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --mix-frames $r8s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag8a_x1_e3_s906
echo "QUEUE 45 DONE ($(date +%H:%M))"
