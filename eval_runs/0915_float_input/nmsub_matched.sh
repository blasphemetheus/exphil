#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
export EXLA_TARGET=host
mix run scripts/scenario_suite.exs --driver teacher --audit-teacher-labels --no-orphan-sweep --character fox --manifest eval_runs/0915_float_input/nmsub_matched_manifest.json --runs 1 --float-ports 1,2 --dolphin /home/blewf/.local/share/slippi/exi-ai-float-nmsub-option-v2/dolphin-emu-headless --no-pipe-shim --response-opponent neutral --window 1 --prefix-history committed --reaction-delay 4 --out eval_runs/0915_float_input/nmsub_matched/report.json --run-dir eval_runs/0915_float_input/nmsub_matched/runs --quiet
