#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export EXPHIL_SKIP_NIF_COMPILE=1
exec mix run --no-compile --no-deps-check --no-start \
  -r lib/exphil/constants.ex -r lib/exphil/embeddings/player.ex \
  -r ../edifice/lib/edifice/recurrent/recurrent.ex \
  -r lib/exphil/networks/policy/execution_contract.ex \
  -r lib/exphil/networks/policy/backbone.ex \
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex \
  -r lib/exphil/training/recorded_frames.ex \
  -r lib/exphil/training/recorded_context.ex \
  -r lib/exphil/agents/agent.ex \
  scripts/prepare_recorded_context.exs --out-dir "${1:?Supply a new output directory}"
