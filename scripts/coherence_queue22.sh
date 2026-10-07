#!/usr/bin/env bash
# Queue 22 (2026-10-06, after queue 21) — Bradley's 10-06 evening order:
# measure before building, then the one-flag shortcut breaker.
#   fullscale_mamba_v1   score checkpoints/fox_mamba_v1_20260925 (Mamba 512x2,
#                        ALL Fox files, 1 ep, windowed 80, no prev-action) on the
#                        recovery scorecard + silence map + fidelity: does data +
#                        scale alone move return (testbed recipe 0.37, expert 0.91)?
#   fullscale_mamba_v2   same for fox_mamba_v2_prevact_20260930 (prev action,
#                        dropout 0.15). Both are symlinks under checkpoints/coh_*
#                        so the experiment script skips training.
#   evt2ctx_ck8_off3_dur8e_pd15   recipe + stick-duration 8 + --prev-action-dropout
#                        0.15: with commitment carried by the duration head, the
#                        copy path is no longer load-bearing for coherence, so the
#                        hold/change decision is forced to read the state.
# Reads: full-scale return >= 0.6 => the testbed under-reports recovery and the
# lever is data/scale; <= 0.45 => scale alone does not close it (build 5/6).
# pd15 pass: repeat 0.70-0.80 / neutral 0.22-0.33 kept, fidelity <= 0.21, and
# any of: offstage age 1-3 enter_silence <= 0.04, return >= 0.37, deep-B
# stick-up share > 35 % (scripts/recovery_death_shape.js).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue21; do sleep 60; done
echo "== queue 21 finished ($(date +%H:%M))"
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
EVALS="coherence fidelity recovery_means recovery_probe" PROBE_BATCH=64 run fullscale_mamba_v1
EVALS="coherence fidelity recovery_means recovery_probe" PROBE_BATCH=64 run fullscale_mamba_v2
for a in fullscale_mamba_v1 fullscale_mamba_v2; do node scripts/recovery_death_shape.js "$a"; done
run evt2ctx_ck8_off3_dur8e_pd15 --prev-action --prev-action-dropout 0.15 --prev-action-quantize "${ev[@]}" \
  --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
node scripts/recovery_death_shape.js evt2ctx_ck8_off3_dur8e_pd15
echo "QUEUE 22 DONE ($(date +%H:%M))"
