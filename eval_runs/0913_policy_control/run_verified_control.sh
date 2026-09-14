#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then
  echo "A BEAM is live; wait before evaluation." >&2
  exit 1
fi
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
OUT=eval_runs/0913_policy_control
NAME="${1:-ep57_verified_k2}"
test ! -e "$OUT/$NAME.json"
mix run --no-compile --no-deps-check \
  -r lib/exphil/eval/scenario_history.ex \
  -r lib/exphil/eval/scenario_input_timing.ex \
  -r lib/exphil/eval/recovery_label_audit.ex scripts/scenario_suite.exs \
  --driver policy --policy checkpoints/ms_g23a_ep57.bin \
  --reaction-delay 2 --temperature 1.0 --character fox \
  --manifest scenarios/ms_midchain_control.json --runs 2 \
  --no-orphan-sweep --quiet --trace-policy-inputs \
  --out "$OUT/$NAME.json" --run-dir "$OUT/$NAME" \
  > "$OUT/$NAME.log" 2>&1
jq '{errors:.errored_runs, diverged:.diverged_runs, invalid_timing:.invalid_timing_runs,
     runs:[.runs[]|{frame,run,timing_valid,chain:.details.max_chain}],summary}' "$OUT/$NAME.json"
