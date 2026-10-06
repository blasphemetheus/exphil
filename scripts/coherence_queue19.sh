#!/usr/bin/env bash
# Queue 19 (2026-10-06): the semi-Markov main stick (`--stick-duration C`,
# INPUT_COHERENCE "10-06"). SilenceMap on the off3 arm: the bot's enter-silence
# hazard is flat 0.03-0.06 in every state (expert 0.007 offstage .. 0.052
# onstage) and largest right after a press (age 1-3 offstage 0.059 vs 0.013).
# The duration head decides the main-stick pair only at decision frames (its
# change + every C-th frame of a hold) and predicts how long it is held; the
# sampler holds in between, so a fresh input cannot be fidgeted away per frame.
#   evt2ctx_ck8_off3_dur8    recipe (prev_q + events + context + chunk 8 + off3) + --stick-duration 8
#   evt2ctx_ck16_off3_dur16  dose point: chunk 16 + --stick-duration 16
# Pass: SilenceMap enter_silence offstage ratio <= 2 in every jumpless band
# (off3: 3-7) and age 1-3 offstage <= 0.03 (0.059); high-band decided return
# >= 0.6 (0.40); fidelity <= 0.21; mismatch <= 0.10; coherence repeat >= 0.70,
# neutral 0.22-0.33; dashes/min and SDs/min within the fidelity reference.
# NOTE: val_loss is a different likelihood (decision-frame main stick +
# duration CE) — not comparable with per-frame arms.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -af '[b]eam.smp' && { echo "a beam is alive; refusing to start"; exit 1; }
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
run() { echo "== $1 ($(date +%H:%M))"; scripts/coherence_experiment.sh "$@"; }
run evt2ctx_ck8_off3_dur8 "${pq[@]}" "${ev[@]}" --chunk-horizon 8 --offstage-weight 3 --stick-duration 8
run evt2ctx_ck16_off3_dur16 "${pq[@]}" "${ev[@]}" --chunk-horizon 16 --offstage-weight 3 --stick-duration 16
echo "QUEUE 19 DONE ($(date +%H:%M))"
