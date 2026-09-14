#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then
  echo "A BEAM is live; wait before evaluation." >&2
  exit 1
fi
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
OUT=eval_runs/0913_policy_control
DELAY="${1:-3}"
NAME="ep57_library_queue_k${DELAY}"
test ! -e "$OUT/$NAME.json"
mix run --no-compile --no-deps-check \
  -r ../libmelee_ex/lib/melee/console.ex \
  -r lib/exphil/eval/scenario_history.ex \
  -r lib/exphil/eval/scenario_input_timing.ex \
  -r lib/exphil/eval/recovery_label_audit.ex scripts/scenario_suite.exs \
  --driver policy --policy checkpoints/ms_g23a_ep57.bin \
  --reaction-delay "$DELAY" --temperature 1.0 --character fox \
  --manifest scenarios/ms_midchain_control.json --runs 2 \
  --no-orphan-sweep --quiet --trace-policy-inputs \
  --out "$OUT/$NAME.json" --run-dir "$OUT/$NAME" \
  > "$OUT/$NAME.log" 2>&1
