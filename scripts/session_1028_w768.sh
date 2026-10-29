#!/usr/bin/env bash
# Width arm (10-28): the master-only recipe at Mamba 768,768,384 (vs 512,512,256), epochs 1-3, fused on,
# batch 128, light readout + agreement each epoch. READ: does the master arm's epoch-3 plateau (option
# families ~53.4 % on the master split, joint ~26.4) rise with capacity, at the same data? PASS: ep3
# families > 54.5 (floor ±0.5) and joint > 27 with SD/min not up; then a league vs master_ep7_plain.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1028_w768/REPORT.md
log() { echo "$*" | tee -a "$R"; }
waitbeam() { while scripts/beam_busy.sh; do sleep 60; done; }   # exphil/nx mix VMs only (scripts/beam_busy.sh)
log ""; log "# width arm 768 on master data ($(date '+%F %H:%M'))"
for N in 1 2 3; do
  lr=0.0002; [ "$N" -ge 3 ] && lr=0.0001
  waitbeam
  log ""; log "## w768 epoch $N (lr $lr) ($(date +%H:%M))"
  env RECIPE=eval_runs/1028_w768/recipe.sh READOUT=light AGREE_GAMES=8 LR=$lr scripts/full_epoch_recipe.sh "$N" 2>&1 | grep -E "RESULT|DONE|FAILED|val_loss|INCOMPLETE|NO_PREV|STALE|\*\* \(|Parameters" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
done
log ""; log "W768 DONE ($(date +%H:%M))"
