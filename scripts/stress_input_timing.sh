#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:?Usage: bash scripts/stress_input_timing.sh NEW_OUTPUT_DIRECTORY}"
if pgrep -x beam.smp >/dev/null; then
  echo "A BEAM is live; wait before evaluation." >&2
  exit 1
fi
mkdir "$OUT"
sha256sum ../libmelee_ex/lib/melee/console.ex ../libmelee_ex/lib/melee/controller.ex lib/exphil_bridge/melee_port.ex \
  scripts/scenario_suite.exs lib/exphil/eval/scenario_input_timing.ex \
  checkpoints/ms_g23a_ep57.bin > "$OUT/sources.sha256"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
failed=0
for polling in 0.0 0.1 0.001; do
  name="poll_${polling}"
  echo "Stress: $name (four fresh launches, reaction 2)"
  status=0
  mix run --no-compile --no-deps-check \
    -r ../libmelee_ex/lib/melee/controller.ex \
    -r ../libmelee_ex/lib/melee/console.ex \
    -r lib/exphil_bridge/melee_port.ex \
    -r lib/exphil/eval/scenario_history.ex \
    -r lib/exphil/eval/scenario_input_timing.ex \
    -r lib/exphil/eval/recovery_label_audit.ex scripts/scenario_suite.exs \
    --driver policy --policy checkpoints/ms_g23a_ep57.bin \
    --reaction-delay 2 --temperature 1.0 --character fox \
    --console-timeout "$polling" \
    --manifest scenarios/ms_midchain_control.json --only 0 --runs 4 \
    --no-orphan-sweep --quiet --trace-policy-inputs \
    --out "$OUT/$name.json" --run-dir "$OUT/$name" \
    > "$OUT/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$OUT/$name.exit"
  if [[ "$status" -ne 0 ]]; then failed=1; fi
  if ! jq -e '.runs | length == 4' "$OUT/$name.json" >/dev/null ||
     ! jq -e '.errored_runs == 0 and .diverged_runs == 0 and
       .invalid_timing_runs == 0 and
       all(.runs[]; .timing_valid == true and .pass == true)' "$OUT/$name.json" >/dev/null; then
    failed=1
  fi
done
jq -n '[inputs | . + {stress_batch: input_filename}] | {batches: map({stress_batch, console_timeout, errored_runs, diverged_runs,
  invalid_timing_runs, runs: [.runs[] | {frame, run, error, timing_valid,
  input_timing, pass, chain: .details.max_chain}]})}' \
  "$OUT"/poll_*.json > "$OUT/summary.json"
echo "Stress artifacts: $OUT (failure=$failed)"
exit "$failed"
