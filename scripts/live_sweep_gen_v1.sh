#!/usr/bin/env bash
# Per-epoch live ranking sweep for fox_gen_v1 at temperature 0.5
# (deterministic decode = absorbing-state collapse for this model class;
# see eval_runs/0826_gen_v1_sweep/RESULTS.md). Async rung, cpu dummy,
# frame-delay 0 (v1 is delay-0 trained). Machine must be QUIET.
set -uo pipefail
cd "$(dirname "$0")/.."

PREFIX=checkpoints/fox_gen_v1_20260825_210355
OUT=eval_runs/0826_gen_v1_sweep
TABLE="$OUT/live_sweep_table.txt"
: > "$TABLE"

for ep in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== ep${ep}" | tee -a "$TABLE"
  bash scripts/eval_live_protocol.sh "${PREFIX}_ep${ep}.bin" \
    "$OUT/live_ep${ep}" --runs 3 --seconds 120 --dummy cpu \
    --temperature 0.5 -- --frame-delay 0 --headless \
    > "$OUT/live_ep${ep}.log" 2>&1 || {
      echo "ep${ep} PROTOCOL FAILED" | tee -a "$TABLE"; continue; }

  for slp in "$OUT/live_ep${ep}"/r*.slp; do
    mix run scripts/coach_report.exs --char fox --bot-port 1 \
      --out "$OUT/coach_ep${ep}_$(basename "$slp" .slp)" "$slp" 2>&1 \
      | grep -a "SCORE:" | sed "s/^/ep${ep} $(basename "$slp" .slp) /" \
      | tee -a "$TABLE" || true
  done
done
echo "LIVE SWEEP DONE" | tee -a "$TABLE"
