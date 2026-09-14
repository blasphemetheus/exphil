#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then
  echo "A BEAM is live; wait before starting this evaluation." >&2
  exit 1
fi
export EXPHIL_SKIP_NIF_COMPILE=1 EXPHIL_GPU_MEMORY_FRACTION=0.15
OUT=eval_runs/0913_recovery_labels
MANIFEST=scenarios/ms_recovery_label_confirmation.json

run_suite() {
  local name="$1"
  shift
  test ! -e "$OUT/$name.json"
  mix run --no-compile --no-deps-check \
    -r lib/exphil/eval/recovery_label_audit.ex scripts/scenario_suite.exs \
    --character fox --no-orphan-sweep --quiet \
    --out "$OUT/$name.json" --run-dir "$OUT/$name" "$@" \
    > "$OUT/$name.log" 2>&1
  jq -e '.errored_runs == 0' "$OUT/$name.json"
}

export EXLA_TARGET=host
run_suite teacher_grounded --driver teacher --audit-teacher-labels \
  --manifest "$MANIFEST" --runs 1

export EXLA_TARGET=cuda
run_suite grounded_control_ep57_k4 --driver policy --policy checkpoints/ms_g23a_ep57.bin \
  --reaction-delay 4 --temperature 1.0 \
  --manifest scenarios/ms_midchain_control.json --only 0 --runs 2
jq -e 'all(.runs[]; (.diverged | not) and .details.max_chain >= 10)' "$OUT/grounded_control_ep57_k4.json"

run_suite g26_ep33_grounded_k4 --driver policy --policy checkpoints/ms_g26a_ep33.bin \
  --reaction-delay 4 --temperature 1.0 --manifest "$MANIFEST" --runs 2
