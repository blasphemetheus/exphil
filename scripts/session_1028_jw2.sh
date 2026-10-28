#!/usr/bin/env bash
# Jump-window dose-down queue (embed-once): within-queue seed floor + low doses.
set -uo pipefail
cd /home/blewf/git/exphil
R=eval_runs/1028_jw/REPORT.md
while pgrep -f '[b-b]eam.smp' >/dev/null; do sleep 60; done   # another session's beam at launch time (13:50)
echo "" | tee -a "$R"; echo "# jump-window queue 2: floor + low doses ($(date '+%F %H:%M'))" | tee -a "$R"
base="--neutral-weight 1.0 --action-oversample 1.0"
MAX_FILES=3000 EVALS="coherence closed_loop recovery" scripts/pq_multi.sh jwb "base=$base;base906=$base --seed 906;jw1=$base --jump-window-weight 1;jw2=$base --jump-window-weight 2;jw5=$base --jump-window-weight 5" --prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events --event-context --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5 2>&1 | tee -a "$R"
echo "JW QUEUE 2 DONE ($(date +%H:%M))" | tee -a "$R"
