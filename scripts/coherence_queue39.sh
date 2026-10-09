#!/usr/bin/env bash
# Queue 39 (2026-10-09 15:20, after queue 38) — the onset-weight knob vs the
# Firefox fire angle. Bradley's live look (12:56) lost 8 of 18 stocks to a
# Firefox fired at the wrong angle; recovery_firefox_angle.js traced it to
# the recipe knob: angle err q50 13° with no onset weight (dur8e_e3) vs
# 30–51° with --onset-weight 30 (the stick-enters-up term is blind to x).
# dag345g_x2 (13:31 → 15:05) repeated the dose law (dashes 11.7, Firefox
# 0.003), so every arm here is the dag345g set at ×1 (the best closed loop).
#   1. apply scripts/patch_onset_buttons_only.js (lib edit, beam-free) +
#      mix compile (coherence_experiment.sh runs --no-compile)
#   2. arms: on0 (knob removed — can the 114k gated labels carry the press
#      alone?), on30b (buttons only), on5u (low dose, current terms), and the
#      on0 seed replicate (s906).
# Pass bar = queue 38's PLUS the angle: err q50 <= 25°, away <= 3 %, died
# after Firefox <= 30 %; press bar unchanged (up-onset >= 7 %/3 f, Firefox-
# once-spent >= 0.04 at <= -40, jump rows >= 0.15/0.20), band (repeat
# 0.70-0.80, neutral 0.22-0.33, fidelity <= 0.21), SD/min <= 1.4, dashes/min
# >= 22, damage/min >= 45.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
until grep -q "QUEUE 38 DONE" logs/exphil-queue38.log; do sleep 120; done
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== queue 38 done, beam free ($(date +%H:%M))"
# never compile while Bradley is editing ../nx: any nx source newer than the built NIF = stop
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
node scripts/patch_onset_buttons_only.js || { echo "PATCH_FAILED"; exit 1; }
mix compile 2>&1 | tail -3
grep -q "onset_buttons_only" _build/dev/lib/exphil/ebin/Elixir.ExPhil.Training.SilentFallWeighting.beam || { echo "COMPILE_FAILED (flag not in beam)"; exit 1; }
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec0=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
readout() {
  node scripts/recovery_death_shape.js "$1"
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30u_dag345g_x1_e3 evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3 "$1"
  node scripts/recovery_aim_vs_press.js evt2ctx_ck8_off3_dur8e_on30u_dag345g_x1_e3 evt2ctx_ck8_off3_dur8e_on30u_dag34g_x2_e3 "$1"
  node scripts/recovery_firefox_angle.js evt2ctx_ck8_off3_dur8e_e3 evt2ctx_ck8_off3_dur8e_on30u_dag345g_x1_e3 "$1"
  node -e 'const s=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/"+process.argv[1]+"/decision_map.json")).summary;console.log(process.argv[1]+" special_up j0: "+["0..-20:j0","-20..-40:j0","-40..-60:j0","<-60:j0"].map(r=>s[r]?r+" "+s[r].special_up+" ["+s[r].frames+"]":r+" -").join("  "))' "$1"
}
r345=data/silent_fall/sim_dagger_expert_r345g_split.frames
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_dag345g_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" --mix-frames $r345 --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_dag345g_x1_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30b_dag345g_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" --onset-weight 30 --onset-buttons-only --mix-frames $r345 --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_on30b_dag345g_x1_e3
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on5u_dag345g_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" --onset-weight 5 --mix-frames $r345 --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_on5u_dag345g_x1_e3
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_dag345g_x1_e3_s906 "${pq[@]}" "${ev[@]}" "${rec0[@]}" --mix-frames $r345 --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_dag345g_x1_e3_s906
echo "QUEUE 39 DONE ($(date +%H:%M))"
