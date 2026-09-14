#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0913_interruption_recovery/live
 test ! -e "$OUT"
mkdir -p "$OUT"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
LOAD=(-r lib/exphil/eval/scenario_opponent.ex -r ../edifice/lib/edifice/recurrent/recurrent.ex
  -r lib/exphil/networks/policy/execution_contract.ex
  -r lib/exphil/networks/policy/backbone.ex
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/training/recorded_prefix_sampling.ex
  -r lib/exphil/training/epoch_loss.ex -r lib/exphil/training/epoch_health.ex
  -r lib/exphil/training/probe_regularizer.ex
  -r lib/exphil/training/imitation/checkpoint.ex -r lib/exphil/training/imitation.ex)
mix run --no-compile --no-deps-check "${LOAD[@]}" \
  -r ../libmelee_ex/lib/melee/controller.ex -r ../libmelee_ex/lib/melee/console.ex \
  -r lib/exphil_bridge/melee_port.ex -r lib/exphil/data/action_frame_convention.ex \
  -r lib/exphil/embeddings/player.ex -r lib/exphil/agents/agent.ex \
  -r lib/exphil/eval/scenario_input_timing.ex -r lib/exphil/eval/scenario_history.ex \
  -r lib/exphil/eval/recovery_label_audit.ex \
  scripts/scenario_suite.exs --driver policy --policy eval_runs/0913_no_dropout_fit/round21/candidate.bin \
  --reaction-delay 2 --temperature 1.0 --character fox --prefix-history committed \
  --manifest eval_runs/0913_interruption_recovery/manifest.json --runs 2 --window 360 --live-af \
  --no-orphan-sweep --quiet --trace-policy-inputs \
  --out "$OUT/eval_candidate.json" --run-dir "$OUT/eval_candidate" \
  > "$OUT/eval_candidate.log" 2>&1
mix run --no-compile --no-deps-check --no-start -r lib/exphil/eval/scenario_opponent.ex scripts/score_interruption_recovery.exs \
  --scores "$OUT/eval_candidate.json" --out "$OUT/recovery.json" > "$OUT/recovery.log" 2>&1
