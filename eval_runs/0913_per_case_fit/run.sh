#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:?provide a NEW output directory}"
test ! -e "$OUT"
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
mkdir -p "$OUT"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
LOAD=(-r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/eval/teacher_fit.ex)
for NAME in candidate initial; do
  mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/measure_teacher_fit.exs \
    --policy "eval_runs/0913_tiny_overfit_cold/$NAME.bin" --out "$OUT/$NAME.json" \
    > "$OUT/$NAME.log" 2>&1
done
mix run --no-compile --no-deps-check --no-start "${LOAD[@]}" scripts/trace_multishine_failures.exs \
  --scores eval_runs/0913_tiny_overfit_cold/eval_candidate.json --out "$OUT/live_trace.json" \
  > "$OUT/live_trace.log" 2>&1
mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/probe_handoff_action_frames.exs \
  --policy eval_runs/0913_tiny_overfit_cold/candidate.bin \
  --scores eval_runs/0913_tiny_overfit_cold/eval_candidate.json --out "$OUT/action_frame_probe.json" \
  > "$OUT/action_frame_probe.log" 2>&1
sha256sum scripts/{measure_teacher_fit,trace_multishine_failures,probe_handoff_action_frames}.exs \
  lib/exphil/eval/teacher_fit.ex lib/exphil/training/{data,labels,recorded_frames}.ex \
  eval_runs/0913_tiny_overfit_cold/{candidate,initial}.bin \
  eval_runs/0913_teacher_ingestion/validated/*.frames > "$OUT/sources.sha256"
