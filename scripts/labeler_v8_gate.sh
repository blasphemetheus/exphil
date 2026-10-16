#!/usr/bin/env bash
# Labeler v8 gate (2026-10-16): rebuild the onstage index with the `offstage`
# dim (weight 3 — airborne past the edge, or not), then the self-check, the
# distance gate, labeler_carried_cells.exs (expert vs tf-label on the expert's
# own states) and label_edge_origins.exs (the chained label on the BOT's r9o
# states, re-labelled with this index). Pass = on the bot's states the
# cross-edge borrows are gone: off_high_inhand / off_ledge_inhand airdodge
# abs-rule hazard ~0 (expert 0/537, 0/876; v7 index 0.50 / 0.25 %/f, 70–78 %
# of the edge neighbours from drift_inhand rows) and drift_spent side_b within
# x2 of the expert's 0.03 %/f (v7 index 0.14, from off_*_spent rows); the
# expert-state cells stay inside x0.5–x2. Stops there: no round, no training.
set -euo pipefail
cd ~/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -f '[b]eam.smp' >/dev/null && { echo "a beam is alive"; exit 1; }
[ -z "$(find ../nx/nx ../nx/exla -name '*.ex' -newer ../nx/exla/cache/libexla.so 2>/dev/null | head -1)" ] || { echo "nx is mid-edit"; exit 1; }
split=checkpoints/coh_fox_mamba_v3s906_ep2/split.json
held=data/silent_fall/heldout_fd_fox_split.json
idx=data/silent_fall/expert_recovery_index_v8.bin
q=eval_runs/1001_queue
sets="--set data/silent_fall/sim_dagger_expert_r9o_2091.frames --set data/silent_fall/slp_dagger_expert_live1014.frames"
mx() { devenv shell -- mix run --no-compile "$@"; }
echo "== index v8, onstage window, offstage dim ($(date +%H:%M))"
mx scripts/build_expert_recovery_index.exs --split $split --games 400 --window onstage --out $idx 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-600
echo "== self-check v8 ($(date +%H:%M))"
mx scripts/expert_labeler_selfcheck.exs --split $held --index $idx --games 24 --out $q/labeler_selfcheck_v8.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-600
echo "== distance gate v8 ($(date +%H:%M))"
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --out $q/labeler_distance_v8_heldout.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-600
echo "== carried/free cells v8 ($(date +%H:%M))"
mx scripts/labeler_carried_cells.exs --split $held --index $idx --games 24 $sets --out $q/labeler_carried_cells_v8.json 2>&1 | grep -E "held-out|DAgger|RESULT|\*\*" | cut -c12-360
echo "== label edge origins on the bot's r9o states, v8 index ($(date +%H:%M))"
mx scripts/label_edge_origins.exs --index $idx $sets --ref $q/labeler_carried_cells_v8.json --out $q/label_edge_origins_r9o_v8.json 2>&1 | grep -E "labelled frames|RESULT|\*\*|rror" | cut -c12-400
echo "GATE DONE ($(date +%H:%M))"
