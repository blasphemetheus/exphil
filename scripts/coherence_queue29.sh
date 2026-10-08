#!/usr/bin/env bash
# Queue 29 (2026-10-08 06:40, after queue 28 closed the danger head) — lever
# (c) of INPUT_COHERENCE "10-08 06:10": weight the LABEL, not the readout.
# `--onset-weight W` (SilentFallWeighting) multiplies the loss only on
# offstage falling frames whose label STARTS a jump (X/Y edge) or a B press;
# the ~97 % holds around them stay at the recipe's offstage x3. Every earlier
# lever fitted the offstage slice harder and learned the hold harder (the
# danger head: 91 % died holding neutral); this is the first lever that moves
# gradient toward the decision frames alone.
#   evt2ctx_ck8_off3_dur8e_on10   recipe + --onset-weight 10, 1 ep
#   evt2ctx_ck8_off3_dur8e_on30   recipe + --onset-weight 30, 1 ep
#   evt2ctx_ck8_off3_dur8e_onW_e3 the better of the two at 3 ep (W picked by
#                                 the -40..-60 jump row), vs dur8e_e3
# Controls: dur8e (1 ep) and dur8e_e3 (3 ep: fidelity 0.175, repeat 0.759,
# neutral 0.221, jump trace 18/21/11 %, return 0.338).
# Pass (DecisionMap, jump in hand): jump >= 0.15 at -20..-40 and >= 0.20 at
# -40..-60; Firefox once spent >= 0.04 at <= -40; WITH the band kept (repeat
# 0.70-0.80, neutral 0.22-0.33, fidelity <= 0.23) and no onstage leak
# (y>0:j1+ jump hazard <= 0.15; arms sit at 0.12-0.14 already).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
ev=(--button-events --stick-events --event-context)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
rec=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off3_dur8e_on10 "${pq[@]}" "${ev[@]}" "${rec[@]}" --onset-weight 10
run evt2ctx_ck8_off3_dur8e_on30 "${pq[@]}" "${ev[@]}" "${rec[@]}" --onset-weight 30
for a in on10 on30; do node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_$a; done
node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e evt2ctx_ck8_off3_dur8e_on10 evt2ctx_ck8_off3_dur8e_on30
# pick W by the -40..-60 jump-in-hand row, require the band roughly kept (fidelity <= 0.25)
pick=$(node -e '
const fs=require("fs"); let best=null, bj=-1;
for (const w of ["10","30"]) { try {
  const d=JSON.parse(fs.readFileSync(`eval_runs/1001_queue/evt2ctx_ck8_off3_dur8e_on${w}/decision_map.json`));
  const j=d.summary["-40..-60:j1+"]?.jump ?? 0;
  const f=JSON.parse(fs.readFileSync(`eval_runs/1001_queue/evt2ctx_ck8_off3_dur8e_on${w}/fidelity.json`));
  const fd=f.fidelity_distance?.mean ?? 0;
  if (fd <= 0.25 && j > bj) { bj=j; best=w; }
} catch (e) {} }
console.log(best ?? "none", bj.toFixed(3));')
echo "== picked W=$pick (jump -40..-60 row)"
W=${pick%% *}
if [ "$W" != none ]; then
  EPOCHS=3 run evt2ctx_ck8_off3_dur8e_on${W}_e3 "${pq[@]}" "${ev[@]}" "${rec[@]}" --onset-weight "$W"
  node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_on${W}_e3
  node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_e3 evt2ctx_ck8_off3_dur8e_on${W}_e3
fi
echo "QUEUE 29 DONE ($(date +%H:%M))"
