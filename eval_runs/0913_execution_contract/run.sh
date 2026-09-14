#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
LOAD=(-r ../edifice/lib/edifice/recurrent/recurrent.ex
  -r lib/exphil/networks/policy/execution_contract.ex
  -r lib/exphil/networks/policy/backbone.ex
  -r lib/exphil/training/imitation/checkpoint.ex
  -r lib/exphil/training/imitation.ex
  -r lib/exphil/agents/agent.ex)
mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/validate_execution_contract.exs \
  --out "${1:-eval_runs/0913_execution_contract/parity.json}"
