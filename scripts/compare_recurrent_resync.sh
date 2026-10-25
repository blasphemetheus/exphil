#!/usr/bin/env bash
# Run after compiling the integration, with no competing GPU jobs for timings.
# EXLA_TARGET=host bash scripts/compare_recurrent_resync.sh POLICY OUT [SEED ...]
set -euo pipefail
policy=${1:?policy export required}
out=${2:?output directory required}
shift 2
if [ "$#" -eq 0 ]; then set -- 1001 1002; fi
mkdir -p "$out"
for seed in "$@"; do
  for mode in windowed carried resync30 resync60; do
    flags=()
    if [ -n "${OPPONENT:-}" ]; then flags+=(--opponent "$OPPONENT"); fi
    if [ "$mode" != windowed ]; then flags+=(--stateful-step); fi
    if [ "$mode" = resync30 ]; then flags+=(--stateful-resync 30); fi
    if [ "$mode" = resync60 ]; then flags+=(--stateful-resync 60); fi
    label="${mode}_s${seed}"
    mix run --no-compile --no-deps-check scripts/sim_closed_loop.exs \
      --policy "$policy" --label "$label" --envs "${ENVS:-8}" \
      --character "${CHARACTER:-fox}" --opponent-character "${OPPONENT_CHARACTER:-fox}" \
      --frames "${FRAMES:-1800}" --seed "$seed" "${flags[@]}" \
      --out "$out/$label.json" 2>&1 | tee "$out/$label.log"
  done
done
