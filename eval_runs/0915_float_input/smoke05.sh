#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
OUT=eval_runs/0915_float_input/smoke05
mkdir "$OUT"
exec > >(tee -a "$OUT/progress.log") 2>&1
export EXLA_TARGET=host
export MELEE_DIRECT_TRACE="$OUT/direct.log"
mix run scripts/scenario_suite.exs --driver teacher --audit-teacher-labels --no-orphan-sweep --character fox --manifest eval_runs/0915_float_input/manifest.json --only 0 --runs 1 --float-ports 2 --no-pipe-shim --response-opponent neutral --window 360 --prefix-history committed --reaction-delay 4 --finalize --out "$OUT/report.json" --run-dir "$OUT/runs" --quiet
