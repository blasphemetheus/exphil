#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:?new output directory required}"
test ! -e "$OUT"
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
mkdir -p "$OUT"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
LOAD=(-r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/eval/teacher_fit.ex)
mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/diagnose_early_windows.exs \
  --out "$OUT/report.json" > "$OUT/run.log" 2>&1
mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/probe_gru_initial_state.exs \
  --out "$OUT/gru_state.json" > "$OUT/gru_state.log" 2>&1
