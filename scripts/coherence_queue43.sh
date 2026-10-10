#!/usr/bin/env bash
# Queue 43 (2026-10-10 13:20) — the r7w no-veto control, alone.
# Queue 42 was stopped after its first arm: --veto-weight 10 left the carried
# side-B trips where they were (15 vs dag6w 13 / 20) and trained the Firefox
# out (stick-up share 12 % vs 32–35 %, Firefox in 10 / 167 episodes vs
# 26 / 131). Mechanism flaw: the veto selects a frame by a BUTTON
# disagreement but `loss_weights` is per frame, so all six heads of the
# 3,246 veto frames (two thirds of them jump presses) were weighted ×10 —
# including the expert's not-up stick on the frames where it declines to
# press. The remaining veto arms replicate a known fail; the control is the
# one worth having: r7w is the r6w roll again (same winner, seeds, gate), so
# dag7w vs dag6w s905 / s906 is the roll's own replicate — the baseline any
# next arm is read against.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
source eval_runs/1001_queue/queue40_winner.sh   # pol, tag=on5u, knob
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec0=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
ctl=(evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906)
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_sideb_trips.js "${ctl[@]}" "$1"
  node scripts/recovery_jump_hazard.js "${ctl[@]}" "$1"
  node scripts/recovery_aim_vs_press.js "${ctl[@]}" "$1"
  node scripts/recovery_firefox_angle.js "${ctl[@]}" "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
r7s=data/silent_fall/sim_dagger_expert_r7w_split.frames
[ -s $r7s ] || { echo "DAGGER_FAILED (no r7w split)"; exit 1; }
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag7w_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --mix-frames $r7s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag7w_x1_e3
echo "QUEUE 43 DONE ($(date +%H:%M))"
