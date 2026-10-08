#!/usr/bin/env bash
# Queue 27 (2026-10-08 02:00, overnight, after queue 26) — the danger-readout
# head, INPUT_COHERENCE "10-05 23:30" lever 1 built after "10-07 20:30":
# the two recovery decisions (jump by height, Firefox once the jump is spent)
# are conditioning defects — every arm's hazard is flat in height at every
# prev-action dose, while the expert's jump rises 0.08 -> 0.30 and the up-B
# 0.02 -> 0.056 (DecisionMap). `--danger-context` hands the AR heads the
# current frame's own y / jumps left / on_ground / speed_y / ledge distance
# through a 32-wide ReLU readout (zero-initialised output): the trunk had the
# features all along; this tests whether the DECISION can use them when
# handed them directly.
#   evt2ctx_ck8_off3_dur8e_pd15_dng   dur8e + pd15 + --danger-context, 1 ep
#   evt2ctx_ck8_off3_dur8e_pd15_dng_e3  same, 3 ep (only if the 1-ep arm moves
#                                        a DecisionMap row by >= 1.5x)
# Pass (per-frame DecisionMap rows, model vs expert): jump hazard with a jump
# in hand >= 0.15 at -20..-40 and >= 0.20 at -40..-60 (pd50: 0.035 / 0.113);
# Firefox once the jump is spent >= 0.04 at <= -40 (pd50: 0.002); offstage
# age 1-3 enter-silence <= 0.035; WITH repeat 0.70-0.80 / neutral 0.22-0.33 /
# fidelity <= 0.23 kept. A move on the rows without the band = the lever is
# real but needs the coherence fix alongside; no move = the heads were not the
# bottleneck either and the lever is the training signal (DAgger relabel).
# This queue compiles first (lib changed: heads/policy/sampling/agent/config)
# and runs the three targeted tests; a failure stops the queue before any arm.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue26; do sleep 60; done
echo "== queue 26 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
echo "== compile ($(date +%H:%M))"
mix compile 2>&1 | grep -E "error|Error|\*\*|Compiling [0-9]+ files" | head -20
test "${PIPESTATUS[0]}" = 0 || { echo "COMPILE_FAILED"; exit 1; }
echo "== tests ($(date +%H:%M))"
mix test test/exphil/networks/policy/danger_context_test.exs test/exphil/networks/policy/event_context_test.exs \
  test/exphil/networks/policy/stick_duration_test.exs test/exphil/eval/decision_map_test.exs 2>&1 | grep -E "tests,|failure|\*\*|Error" | head -30
test "${PIPESTATUS[0]}" = 0 || { echo "TESTS_FAILED"; exit 1; }
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.15 --prev-action-quantize)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off3_dur8e_pd15_dng "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --danger-context
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_pd15_dng
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_pd15 evt2ctx_ck8_off3_dur8e_pd15_dng
# 3-ep follow-up when the 1-ep arm moved the jump row at -40..-60 to >= 1.5x pd50's 0.113
moved=$(node -e '
try { const d = JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/evt2ctx_ck8_off3_dur8e_pd15_dng/decision_map.json"));
  const j = d.summary["-40..-60:j1+"] && d.summary["-40..-60:j1+"].jump; const u = d.summary["<-60:j0"] && d.summary["<-60:j0"].special_up;
  console.log((j != null && j >= 0.17) || (u != null && u >= 0.01) ? "yes" : "no"); } catch (e) { console.log("no"); }')
echo "== 1-ep arm moved a decision row: $moved"
if [ "$moved" = yes ]; then
  EPOCHS=3 run evt2ctx_ck8_off3_dur8e_pd15_dng_e3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --danger-context
  node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_pd15_dng_e3
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_pd15_dng evt2ctx_ck8_off3_dur8e_pd15_dng_e3
fi
echo "QUEUE 27 DONE ($(date +%H:%M))"
