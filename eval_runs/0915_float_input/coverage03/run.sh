#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
OUT=eval_runs/0915_float_input/coverage03
exec > >(tee -a "$OUT/pipeline.log") 2>&1
while systemctl --user is-active --quiet exphil-float-smoke11.service; do sleep 2; done
if pgrep -x beam.smp >/dev/null; then echo 'Another BEAM is live'; exit 1; fi
jq -e '.errored_runs == 0 and .diverged_runs == 0 and ([.runs[] | select(.prefix_audit.valid == true)] | length) == 3' eval_runs/0915_float_input/smoke11/report.json >/dev/null
bash eval_runs/0915_float_input/controls03.sh
for report in eval_runs/0915_float_input/controls03/{pipe,direct}_{cold,committed}.json; do
  jq -e '.errored_runs == 0 and .diverged_runs == 0 and .invalid_timing_runs == 0 and ([.runs[] | select(.pass == true)] | length) == 12' "$report" >/dev/null
done
bash "$OUT/teach.sh"
bash "$OUT/train.sh"
echo "$(date -Is) PIPELINE DONE"
