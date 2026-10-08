#!/usr/bin/env bash
# Queue 30 (2026-10-08 09:40, after queue 29) — the Firefox half of the onset
# lever, and the seed replication of the new recipe candidate.
# Queue 29 ("10-08 08:00", "09:30"): `on30_e3` = dur8e + --onset-weight 30 at
# 3 ep PASSES the port bar (fidelity 0.162 — best ever, repeat 0.78, neutral
# 0.304) AND the jump rows (DecisionMap 0.176/0.282/0.258 vs expert
# 0.202/0.297/0.299, slope 4.0), jump-in-hand deaths 19 % (dur8e_e3 27 %),
# return 0.389. Firefox once the jump is spent stayed at the floor (0.005)
# and side-B is the top death (27): the B-edge weight never weighted the
# stick-UP change that aims a Firefox. `onset?` now also counts the stick
# entering UP once the jump is spent, and a B edge only with the stick up
# (8.1 onsets per game, from 9.5).
#   evt2ctx_ck8_off3_dur8e_on30u      recipe + onset-weight 30 (extended), 1 ep
#   evt2ctx_ck8_off3_dur8e_on30u_e3   same, 3 ep (only if the 1-ep arm keeps
#                                     the jump rows: -40..-60 jump >= 0.17)
#   evt2ctx_ck8_off3_dur8e_on30_e3_s906  the queue-29 candidate at seed 906
#                                     (replication of the bar + jump rows)
# Pass for on30u(_e3): Firefox once spent >= 0.04 at <= -40 (on30_e3 0.005),
# jump rows kept (>= 0.15 / 0.20), side-B deaths < 15 (on30_e3 27), band kept.
# Pass for s906: port bar + jump rows (>= 0.15 / 0.20) again.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off3_dur8e_on30u "${pq[@]}" "${ev[@]}" "${rec[@]}" --onset-weight 30
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on30u
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30 evt2ctx_ck8_off3_dur8e_on30u
keep=$(node -e '
try { const d = JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/evt2ctx_ck8_off3_dur8e_on30u/decision_map.json"));
  const j = d.summary["-40..-60:j1+"] && d.summary["-40..-60:j1+"].jump; console.log(j != null && j >= 0.17 ? "yes" : "no"); } catch (e) { console.log("no"); }')
echo "== on30u kept the jump row: $keep"
if [ "$keep" = yes ]; then
  EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30u_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --onset-weight 30
  node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on30u_e3
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30_e3 evt2ctx_ck8_off3_dur8e_on30u_e3
fi
SEED=906 EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on30_e3_s906 "${pq[@]}" "${ev[@]}" "${rec[@]}" --onset-weight 30
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on30_e3_s906
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_on30_e3 evt2ctx_ck8_off3_dur8e_on30_e3_s906
echo "QUEUE 30 DONE ($(date +%H:%M))"
