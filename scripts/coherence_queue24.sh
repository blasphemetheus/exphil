#!/usr/bin/env bash
# Queue 24 (2026-10-07 13:10, after queue 23) — the full-scale read named the
# shortcut: fox_mamba_v1 (no prev-action, all files, 512x2) returns 0.518 and
# dies with a jump left 9 % (expert 7 %), fox_mamba_v2 (prev-action, dropout
# 0.15, same data/scale) returns 0.202 / 24 %. Same data, same width: the
# prev-action channel is what stops the recovery decision from being learned,
# and it is also what bought coherence (v1 repeat 0.38). The duration head now
# carries commitment, so the channel may be removable:
#   evt2ctx_ck8_off3_dur8e_nopq   dur8e recipe WITHOUT --prev-action
#   evt2ctx_ck8_off3_dur8e_pd50   dur8e + --prev-action-dropout 0.5 (middle)
# Pass (the lever): return >= 0.45 or dies-with-jump-left <= 15 % with repeat
# >= 0.70 / neutral 0.22-0.33 / fidelity <= 0.23 kept. If nopq keeps the
# coherence band, the prev-action channel leaves the port recipe.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue23; do sleep 60; done
echo "== queue 23 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off3_dur8e_nopq "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_nopq
run evt2ctx_ck8_off3_dur8e_pd50 --prev-action --prev-action-dropout 0.5 --prev-action-quantize "${ev[@]}" \
  --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_pd50
echo "QUEUE 24 DONE ($(date +%H:%M))"
