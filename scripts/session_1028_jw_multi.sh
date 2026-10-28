#!/usr/bin/env bash
# Jump-window testbed queue, embed-once driver (replaces session_1028_jw.sh). 3,000 games, 5 arms.
set -uo pipefail
cd /home/blewf/git/exphil
R=eval_runs/1028_jw/REPORT.md; mkdir -p eval_runs/1028_jw
echo "" | tee -a "$R"; echo "# jump-window queue via pq_multi ($(date '+%F %H:%M'))" | tee -a "$R"
base="--neutral-weight 1.0 --action-oversample 1.0"
MAX_FILES=3000 EVALS="coherence closed_loop recovery" scripts/pq_multi.sh jw "base3k=$base;jw5=$base --jump-window-weight 5;jw10=$base --jump-window-weight 10;jw20=$base --jump-window-weight 20;pw10_jw10=$base --press-window-weight 10 --jump-window-weight 10" --prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events --event-context --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5 2>&1 | tee -a "$R"
echo "JW QUEUE DONE ($(date +%H:%M))" | tee -a "$R"
