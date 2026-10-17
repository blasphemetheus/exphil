#!/usr/bin/env bash
# Labeler v9 gate (2026-10-17): rebuild the onstage index with SETTLED-stick
# change rows (Bradley on the labels page: "we want to be able to represent
# dash-dancing" — a travelling flick 0.50 -> 0.72 -> 0.99 stored mid-travel
# labels a walk, and the hold then keeps it). Same gate as v8 (self-check,
# distance, carried cells, edge origins on the r9o sets): features are
# unchanged so d2 / hold share must not move; the stick change is read on the
# review export (label_review_export.exs -> stick_bands.mjs: share of label
# stick changes landing in the partial band 0.15 < |x-0.5| < 0.4) and by eye
# on 14_32_43 @0 f23-25. Stops there: no round, no training.
set -euo pipefail
cd ~/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -f '[b]eam.smp' >/dev/null && { echo "a beam is alive"; exit 1; }
[ -z "$(find ../nx/nx ../nx/exla -name '*.ex' -newer ../nx/exla/cache/libexla.so 2>/dev/null | head -1)" ] || { echo "nx is mid-edit"; exit 1; }
split=checkpoints/coh_fox_mamba_v3s906_ep2/split.json
held=data/silent_fall/heldout_fd_fox_split.json
idx=data/silent_fall/expert_recovery_index_v9.bin
q=eval_runs/1001_queue
sets="--set data/silent_fall/sim_dagger_expert_r9o_2091.frames --set data/silent_fall/slp_dagger_expert_live1014.frames"
mx() { devenv shell -- mix run --no-compile "$@"; }
echo "== index v9, onstage window, settled stick ($(date +%H:%M))"
mx scripts/build_expert_recovery_index.exs --split $split --games 400 --window onstage --stick settled --out $idx 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-600
echo "== self-check v9 ($(date +%H:%M))"
mx scripts/expert_labeler_selfcheck.exs --split $held --index $idx --games 24 --out $q/labeler_selfcheck_v9.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-600
echo "== distance gate v9 ($(date +%H:%M))"
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --out $q/labeler_distance_v9_heldout.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-600
echo "== carried/free cells v9 ($(date +%H:%M))"
mx scripts/labeler_carried_cells.exs --split $held --index $idx --games 24 $sets --out $q/labeler_carried_cells_v9.json 2>&1 | grep -E "held-out|DAgger|RESULT|\*\*" | cut -c12-360
echo "== label edge origins on the bot's r9o states, v9 index ($(date +%H:%M))"
mx scripts/label_edge_origins.exs --index $idx $sets --ref $q/labeler_carried_cells_v9.json --out $q/label_edge_origins_r9o_v9.json 2>&1 | grep -E "labelled frames|RESULT|\*\*|rror" | cut -c12-400
echo "GATE DONE ($(date +%H:%M))"
