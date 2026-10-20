#!/usr/bin/env bash
# Labeler v9 read-out (2026-10-17): chain-relabel one r9o part with the v9
# index (the honest dash-dance read: teacher-forcing resets prev to the real
# mid-travel stick, a chain keeps the settled flick), then re-export the
# label review runs with v9 + the real characters, then stick_bands on both
# exports. No training.
set -euo pipefail
cd ~/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -f 'exphil/.*[b]eam.smp' >/dev/null && { echo "an exphil beam is alive"; exit 1; }
[ -z "$(find ../nx/nx ../nx/exla -name '*.ex' -newer ../nx/exla/cache/libexla.so 2>/dev/null | head -1)" ] || { echo "nx is mid-edit"; exit 1; }
idx=data/silent_fall/expert_recovery_index_v9.bin
held=data/silent_fall/heldout_fd_fox_split.json
mx() { devenv shell -- mix run --no-compile "$@"; }
echo "== chain relabel r9o_2091 with v9 ($(date +%H:%M))"
mx scripts/dagger_set_relabel.exs --index $idx --in data/silent_fall/sim_dagger_expert_r9o_2091.frames --out data/silent_fall/sim_dagger_expert_r9o_2091_v9chain.frames --prev chain --report eval_runs/1001_queue/relabel_r9o_2091_v9chain.json 2>&1 | grep -E "RESULT|labelled|gated|rror|\*\*" | cut -c1-300
echo "== label review export v9 ($(date +%H:%M))"
mx scripts/label_review_export.exs --index $idx --set data/silent_fall/sim_dagger_expert_r9o_2091_v9chain.frames --set data/silent_fall/sim_dagger_expert_r9o_2091.frames --slp $held --games 6 --max-runs 80 --out eval_runs/1017_label_review_v9 2>&1 | grep -E "RESULT|runs|rror|\*\*" | cut -c1-300
echo "== stick bands ($(date +%H:%M))"
node scripts/stick_bands.mjs eval_runs/1016_label_review eval_runs/1017_label_review_v9
echo "READOUT DONE ($(date +%H:%M))"
