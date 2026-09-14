#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then
  echo "A BEAM is live; wait before evaluation." >&2
  exit 1
fi
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
OUT=eval_runs/0913_policy_control
for reaction in 2 3 4; do
  name="ep57_committed_k${reaction}"
  test ! -e "$OUT/$name.json"
  mix run --no-compile --no-deps-check -r lib/exphil/eval/scenario_history.ex \
    scripts/scenario_suite.exs \
    --driver policy --policy checkpoints/ms_g23a_ep57.bin \
    --reaction-delay "$reaction" --temperature 1.0 --character fox \
    --prefix-history committed --trace-policy-inputs \
    --manifest scenarios/ms_midchain_control.json --only 0 --runs 2 \
    --no-orphan-sweep --no-verify-input-timing --quiet \
    --out "$OUT/$name.json" --run-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  jq '{reaction:'"$reaction"', errors:.errored_runs, diverged:.diverged_runs,
       chains:[.runs[].details.max_chain]}' "$OUT/$name.json"
done
