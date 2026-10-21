#!/usr/bin/env bash
# 10-21: index v10 = v9 rows + PROVENANCE (which replay + frame each row came
# from), then the label review export with SOURCE CLIPS: for every frame with
# an edge, the expert moment each of the 8 neighbour rows was copied from, as
# a 91-frame clip the page plays beside the run (Bradley: "I just wanted some
# way to hand verify that whatever you're doing with the labeler ... makes
# some sort of sense"). No training, no round.
set -euo pipefail
cd ~/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -f '[b]eam.smp' >/dev/null && { echo "a beam is alive"; exit 1; }
[ -z "$(find ../nx/nx ../nx/exla -name '*.ex' -newer ../nx/exla/cache/libexla.so 2>/dev/null | head -1)" ] || { echo "nx is mid-edit"; exit 1; }
split=checkpoints/coh_fox_mamba_v3s906_ep2/split.json
held=data/silent_fall/heldout_fd_fox_split.json
idx=data/silent_fall/expert_recovery_index_v10.bin
out=eval_runs/1021_label_review_src
mx() { devenv shell -- mix run --no-compile "$@"; }
echo "== index v10 = v9 + provenance ($(date +%H:%M))"
mx scripts/build_expert_recovery_index.exs --split $split --games 400 --window onstage --stick settled --out $idx 2>&1 | tee logs/index_v10.log | grep -E "RESULT|error|\*\*" | cut -c1-600
echo "== review export with source clips ($(date +%H:%M))"
mx scripts/label_review_export.exs --index $idx --set data/silent_fall/sim_dagger_expert_r9o_2091_v9chain.frames --set data/silent_fall/sim_dagger_expert_r9o_2091.frames --slp $held --games 6 --max-runs 80 --out $out 2>&1 | tee logs/export_src.log | grep -E "RESULT|clips|games|rror|\*\*" | cut -c1-300
echo "== done ($(date +%H:%M)): http://127.0.0.1:8012/labels/?run=/labelruns/1021_label_review_src"
