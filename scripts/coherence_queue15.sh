#!/usr/bin/env bash
# Queue 15 (2026-10-05 afternoon): lever 1 for the silent fall
# (INPUT_COHERENCE_2026-10-01.md "10-05 12:55") — loss weight on the
# data-starved offstage tail, on the MinGRU testbed, 1 epoch, against the
# existing evt2ctx_ck8 baseline (decided-trip return 0.45, high band 0.40,
# Q5 k25-48 model .163 | expert .247):
#   evt2ctx_ck8_sf20   --silent-fall-weight 20 (k >= 13; ~9 frames/game, ~2 % of loss mass)
#   evt2ctx_ck8_off3   --offstage-weight 3    (every offstage frame, 8.5 % of frames — the blunt arm)
# Pass (sf20): Q5 k25+ model >= expert; closed-loop resume hazard >= .15 at
# 18-24 f of silence; high-band decided return >= 0.6; coherence repeat >= 0.6,
# L-cancel >= 0.7, fidelity not worse than 0.198.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_sf20 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --silent-fall-weight 20
run evt2ctx_ck8_off3 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3
echo "QUEUE 15 DONE ($(date +%H:%M))"
