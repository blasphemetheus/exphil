#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0913_tiny_overfit
NAME="${1:?Checkpoint name without extension}"
test ! -e "$OUT/eval_$NAME.json"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
mix run --no-compile --no-deps-check \
  -r ../libmelee_ex/lib/melee/controller.ex \
  -r ../libmelee_ex/lib/melee/console.ex -r lib/exphil_bridge/melee_port.ex \
  -r lib/exphil/eval/scenario_input_timing.ex -r lib/exphil/eval/scenario_history.ex \
  -r lib/exphil/eval/recovery_label_audit.ex scripts/scenario_suite.exs \
  --driver policy --policy "$OUT/$NAME.bin" --reaction-delay 2 \
  --temperature 1.0 --character fox --prefix-history committed \
  --manifest "$OUT/manifest.json" --runs 2 \
  --no-orphan-sweep --quiet --trace-policy-inputs \
  --out "$OUT/eval_$NAME.json" --run-dir "$OUT/eval_$NAME" \
  > "$OUT/eval_$NAME.log" 2>&1
jq -e '.errored_runs == 0 and .diverged_runs == 0 and .invalid_timing_runs == 0
  and (.runs|length == 12) and all(.runs[];
    .timing_valid == true and .truncated == null and .frames_observed == 120
    and .details.max_chain >= 10)' "$OUT/eval_$NAME.json"
