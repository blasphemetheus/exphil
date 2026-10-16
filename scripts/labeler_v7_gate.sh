#!/usr/bin/env bash
# Labeler v7 gate (2026-10-16): rebuild the onstage index with the v7 dims
# (helpless / tumble) and the expert's previous input per row (button-EDGE
# rule, `row_controller_edges/4`), then the self-check, the distance gate and
# labeler_carried_cells.exs. Pass = the teacher-forced label within x2 of the
# expert on every offstage cell with >= 20 expert events (v6: deep-spent B 12x,
# ledge-in-hand jump 4x) with the self-check cells still inside x0.5-x2. The
# self-check's "B edge, stick not up" line is a player-style metric (5 events
# held-out), not a bar. Stops there: no round, no training.
set -euo pipefail
cd ~/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -f '[b]eam.smp' >/dev/null && { echo "a beam is alive"; exit 1; }
[ -z "$(find ../nx/nx ../nx/exla -name '*.ex' -newer ../nx/exla/cache/libexla.so 2>/dev/null | head -1)" ] || { echo "nx is mid-edit"; exit 1; }
split=checkpoints/coh_fox_mamba_v3s906_ep2/split.json
held=data/silent_fall/heldout_fd_fox_split.json
idx=data/silent_fall/expert_recovery_index_v7.bin
q=eval_runs/1001_queue
mx() { devenv shell -- mix run --no-compile "$@"; }
echo "== index v7, onstage window, edge rows ($(date +%H:%M))"
mx scripts/build_expert_recovery_index.exs --split $split --games 400 --window onstage --out $idx 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-600
echo "== self-check v7 ($(date +%H:%M))"
mx scripts/expert_labeler_selfcheck.exs --split $held --index $idx --games 24 --out $q/labeler_selfcheck_v7.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-600
echo "== distance gate v7 ($(date +%H:%M))"
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --out $q/labeler_distance_v7_heldout.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-600
echo "== carried/free cells v7 ($(date +%H:%M))"
mx scripts/labeler_carried_cells.exs --split $held --index $idx --games 24 --set data/silent_fall/sim_dagger_expert_r9o_2091.frames --set data/silent_fall/slp_dagger_expert_live1014.frames --out $q/labeler_carried_cells_v7.json 2>&1 | grep -E "held-out|DAgger|^\s+[a-z]|RESULT|\*\*" | cut -c12-360
echo "GATE DONE ($(date +%H:%M))"
