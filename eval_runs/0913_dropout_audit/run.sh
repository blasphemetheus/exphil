#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:?new output directory required}"
test ! -e "$OUT"
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
mkdir -p "$OUT"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
LOAD=(-r ../edifice/lib/edifice/recurrent/recurrent.ex
  -r lib/exphil/networks/policy/execution_contract.ex -r lib/exphil/networks/policy/backbone.ex
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/eval/teacher_fit.ex)
mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/measure_teacher_fit.exs \
  --policy eval_runs/0913_zero_f32_fit/round21/candidate.bin --include-canonical-rows \
  --out "$OUT/fit.json" > "$OUT/fit.log" 2>&1
mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/audit_previous_action_dropout.exs \
  --report "$OUT/fit.json" --out "$OUT/probe.json" > "$OUT/probe.log" 2>&1
