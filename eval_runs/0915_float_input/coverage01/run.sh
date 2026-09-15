#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
OUT=eval_runs/0915_float_input/coverage01
exec > >(tee -a "$OUT/pipeline.log") 2>&1
while systemctl --user is-active --quiet exphil-float-controls02.service; do sleep 2; done
if pgrep -x beam.smp >/dev/null; then echo 'Another BEAM is live; refusing to rebuild'; exit 1; fi
for report in eval_runs/0915_float_input/controls01/pipe_{cold,committed}.json eval_runs/0915_float_input/controls02/direct_{cold,committed}.json; do
  jq -e '.errored_runs == 0 and .diverged_runs == 0 and .invalid_timing_runs == 0 and ([.runs[] | select(.pass == true)] | length) == 12' "$report" >/dev/null
done
EXLA_TARGET=host mix test test/exphil/eval/replay_prefix_audit_test.exs test/exphil/eval/float_input_build_test.exs test/exphil/data/processed_input_test.exs > "$OUT/tests.log" 2>&1
bash "$OUT/teach.sh"
bash "$OUT/train.sh"
echo "$(date -Is) PIPELINE DONE"
