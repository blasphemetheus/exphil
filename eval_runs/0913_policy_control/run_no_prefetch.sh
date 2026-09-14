#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then
  echo "A BEAM is live; wait before evaluation." >&2
  exit 1
fi
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
OUT=eval_runs/0913_policy_control
test ! -e "$OUT/ep57_no_prefetch_k2.json"
mix run --no-compile --no-deps-check \
  -r "$OUT/console_no_prefetch.exs" \
  -r lib/exphil/eval/scenario_history.ex \
  -r lib/exphil/eval/scenario_input_timing.ex \
  -r lib/exphil/eval/recovery_label_audit.ex scripts/scenario_suite.exs \
  --driver policy --policy checkpoints/ms_g23a_ep57.bin \
  --reaction-delay 2 --temperature 1.0 --character fox \
  --manifest scenarios/ms_midchain_control.json --runs 2 \
  --no-orphan-sweep --quiet --trace-policy-inputs \
  --out "$OUT/ep57_no_prefetch_k2.json" --run-dir "$OUT/ep57_no_prefetch_k2" \
  > "$OUT/ep57_no_prefetch_k2.log" 2>&1
