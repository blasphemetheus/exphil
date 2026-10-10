#!/usr/bin/env bash
# Queue 44 (2026-10-10, after queue 43) — the onset weight where the carried
# side-B deaths start. INPUT_COHERENCE "10-10 13:30": on the bot's own
# states (airborne, -30..+10 of the edge, facing out, stick held outward)
# the expert fires the fatal joint too (0.3 %/frame) and the labels carry it
# at the bot's rate, so no relabel weighting moves it; the bot's excess is
# (1) its B lands on the stale outward stick — the expert turns the stick
# with the press 95 % of the time, the bot 20 % — and (2) passivity in the
# state (B 1.0 vs 4.8 %, jump 1.2 vs 4.2 % per frame), so it lingers.
# `--onset-edge-window 15` extends `--onset-weight 5` to airborne frames
# within 15 units of the edge, where the onset is a jump edge or a B edge
# WITH a same-frame stick-zone change (SilentFallWeighting.edge_onset?/2);
# the B-on-held-stick press is not lifted. Two seeds, read against dag7w
# s905 (queue 43) and dag6w s905 / s906.
# Pass (both seeds): carried side-B trips <= 5 per recovery_means run
# (dag6w 13 / 20), carried-off died <= 0.6; band / dashes / up-onset / angle
# unchanged; Firefox -40..-60 row not worse.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
until grep -q "QUEUE 43 DONE" logs/exphil-queue43.log; do sleep 120; done
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== queue 43 done, beam free ($(date +%H:%M))"
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
# the flag must be compiled in (lib edits of 10-10 13:35); refuse to run on a stale build
elixir -pa '_build/dev/lib/*/ebin' -e 'Code.ensure_loaded!(ExPhil.Training.SilentFallWeighting); System.halt(if function_exported?(ExPhil.Training.SilentFallWeighting, :edge_onset?, 2), do: 0, else: 1)' \
  || { echo "BUILD_STALE (edge_onset?/2 missing — run devenv shell -- mix compile)"; exit 1; }
source eval_runs/1001_queue/queue40_winner.sh   # pol, tag=on5u, knob
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec0=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
ctl=(evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906)
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
EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag7w_edge15_x1_e3 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --onset-edge-window 15 --mix-frames $r7s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag7w_edge15_x1_e3
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_${tag}_dag7w_edge15_x1_e3_s906 "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" --onset-edge-window 15 --mix-frames $r7s --mix-oversample 1
readout evt2ctx_ck8_off3_dur8e_${tag}_dag7w_edge15_x1_e3_s906
echo "QUEUE 44 DONE ($(date +%H:%M))"
