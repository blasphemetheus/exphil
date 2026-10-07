#!/usr/bin/env bash
# Queue 23 (2026-10-07 00:30, after queue 22) — lever 3 arms that need no code,
# on the dur8e candidate (recipe + stick-duration 8), seed 905, 1 ep:
#   evt2ctx_ck8_off3_dur8e_w240   --window-size 240 (WINDOW env): can the windowed
#                                 path see the whole offstage episode (a recovery
#                                 is ~100-200 frames; window 80 cuts most of them)?
#   evt2ctx_ck8_off3_dur8e_mf9k   MAX_FILES=9000 (3x data, same slice prefix):
#                                 first point of the data-scaling curve on the
#                                 testbed, to read next to fullscale_mamba_v1/v2
#                                 (queue 22) which have all files AND 4x width.
# Reads (vs dur8e: return 0.23, fidelity 0.191, repeat 0.777, neutral 0.296,
# offstage age 1-3 enter_silence ~0.03): return >= 0.37 on either = the lever is
# context / data and the Mamba port carries it; coherence band must hold
# (repeat 0.70-0.80 / neutral 0.22-0.33, fidelity <= 0.21) or the arm is a
# trade, not a win. Death shape: deep-B stick-up share > 35 %, dies-with-jump
# < 20 % would say the recovery *decision* moved, not just the rate.
# ACT (chunk/latent) arm deliberately NOT here: policy_type :act is wired in the
# loss/train loop only, not in Agent/eval loading — needs code first.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue22; do sleep 60; done
echo "== queue 22 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
WINDOW=240 run evt2ctx_ck8_off3_dur8e_w240 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_w240
MAX_FILES=9000 run evt2ctx_ck8_off3_dur8e_mf9k "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_mf9k
echo "QUEUE 23 DONE ($(date +%H:%M))"
