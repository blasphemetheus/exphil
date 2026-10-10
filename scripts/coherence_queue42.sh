#!/usr/bin/env bash
# Queue 42 (2026-10-10 09:15, after queue 41) — the DAgger disagreement
# ("veto") weight on the one death mode that replicated on every seed.
# Queue 41 read: SD/min-sized differences are seed-sized on this testbed;
# what replicates on every on5u arm and both seeds is (1) the knob's
# up-onset, (2) carried-off died 0.78–0.88, (3) side-B offstage trips that
# die 100 % — on-stage Illusions fired within ~25 units of the edge facing
# out with no jump (recovery_sideb_trips.js), 13–71 per recovery_means run,
# the expert 5 in 1788 episodes. The wide rounds' labels at those frames
# already say "no B" (70 % gated-in, ~100 veto frames per 142k) and the
# teacher-forced excess is 0.04 %/frame (DecisionMap Q1 press∧stick→edge
# 0.0004–0.0008 vs 0.0): invisible to the unweighted loss, and the onset
# weight only lifts the expert's PRESSES. `--veto-weight W` lifts the
# relabelled frames where the bot pressed X/Y/B and the label holds
# (SilentFallWeighting.veto?/3), which needs a round rolled with
# `--keep-actual` (the saved sets dropped the bot's input).
#   0. r7w = the r6w roll again (same winner, index v4, gate, seeds, label
#      seed) with --keep-actual; veto count logged
#   1. on5u + r7w, --veto-weight 10, seed 905
#   2. the same on seed 906
#   3. on5u + r7w, no veto (the re-roll control vs dag6w s905)
#   4. on5u + r7w, --veto-weight 30, seed 905 (the dose bracket; last, killable)
# Pass (two seeds): carried side-B trips <= 5 per recovery_means run (dag6w:
# 13 / 20), carried-off died <= 0.6, band / dashes / aim unchanged, Firefox
# -40..-60 row not worse. Controls: dag6w s905 / s906 (queue 40 / 41).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
until grep -q "QUEUE 41 DONE" logs/exphil-queue41.log; do sleep 120; done
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== queue 41 done, beam free ($(date +%H:%M))"
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
source eval_runs/1001_queue/queue40_winner.sh   # pol, tag=on5u, knob
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec0=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
mf() { elixir -pa '_build/dev/lib/*/ebin' "$@"; }
ctl=(evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906)
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_sideb_trips.js "${ctl[@]}" "$1"
  node scripts/recovery_jump_hazard.js "${ctl[@]}" "$1"
  node scripts/recovery_aim_vs_press.js "${ctl[@]}" "$1"
  node scripts/recovery_firefox_angle.js "${ctl[@]}" "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
idx=data/silent_fall/expert_recovery_index_v4.bin
gate=$(node -e 'const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_distance_v4_heldout.json"));console.log(r.gate_q95?r.gate_q95.toFixed(3):"")')
[ -n "$gate" ] || { echo "LABELER_FAILED (no gate)"; exit 1; }
# 0. r7w: the r6w roll with the bot's own input kept
r7=data/silent_fall/sim_dagger_expert_r7w.frames
r7s=data/silent_fall/sim_dagger_expert_r7w_split.frames
echo "== gated v4 rollout + relabel from $tag (d2 <= $gate), 12 seeds in 4 beams, --keep-actual ($(date +%H:%M))"
parts=()
for grp in 2091,2092,2093 2094,2095,2096 2097,2098,2099 2100,2101,2102; do
  part=data/silent_fall/sim_dagger_expert_r7w_${grp%%,*}.frames
  mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index $idx --max-d2 $gate --keep-actual \
    --seeds $grp --label-seed 12 --out $part --report eval_runs/1001_queue/sim_dagger_expert_r7w_${grp%%,*}.json \
    > logs/dagger_r7w_${grp%%,*}.log 2>&1
  grep -E "RESULT|seed |RESOURCE_EXHAUSTED|\*\*" logs/dagger_r7w_${grp%%,*}.log | cut -c1-400
  [ -s eval_runs/1001_queue/sim_dagger_expert_r7w_${grp%%,*}.json ] || { echo "DAGGER_FAILED (seeds $grp, see logs/dagger_r7w_${grp%%,*}.log)"; exit 1; }
  parts+=("$part")
done
mf scripts/dagger_set_concat.exs "${parts[@]}" $r7 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-300
[ -s $r7 ] || { echo "DAGGER_FAILED (no r7w concat)"; exit 1; }
mf scripts/dagger_set_label_hazards.exs $r7 2>&1 | grep RESULT | cut -c1-300
mf scripts/dagger_set_split_gated.exs $r7 $r7s 2>&1 | grep RESULT | cut -c1-300
[ -s $r7s ] || { echo "DAGGER_FAILED (no r7w split)"; exit 1; }
mf scripts/dagger_set_veto_count.exs $r7s 2>&1 | grep -E "RESULT|rror" | cut -c1-300
# 1–4. arms
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag7w_veto10_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --veto-weight 10 --mix-frames $r7s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag7w_veto10_x1_e3
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag7w_veto10_x1_e3_s906 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --veto-weight 10 --mix-frames $r7s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag7w_veto10_x1_e3_s906
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag7w_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --mix-frames $r7s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag7w_x1_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag7w_veto30_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --veto-weight 30 --mix-frames $r7s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag7w_veto30_x1_e3
echo "QUEUE 42 DONE ($(date +%H:%M))"
