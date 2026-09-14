#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0913_no_dropout_fit/round21
test ! -e "$OUT"
mkdir -p "$OUT"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
LOAD=(-r ../edifice/lib/edifice/recurrent/recurrent.ex
  -r lib/exphil/networks/policy/execution_contract.ex
  -r lib/exphil/networks/policy/backbone.ex
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/training/recorded_prefix_sampling.ex
  -r lib/exphil/training/epoch_loss.ex -r lib/exphil/training/epoch_health.ex
  -r lib/exphil/training/probe_regularizer.ex
  -r lib/exphil/training/imitation/checkpoint.ex -r lib/exphil/training/imitation.ex)
sha256sum "$0" scripts/{dagger_drill,measure_teacher_fit,check_early_teacher_fit}.exs \
  ../edifice/lib/edifice/recurrent/recurrent.ex \
  lib/exphil/networks/policy/{execution_contract,backbone}.ex \
  lib/exphil/training/{data,labels,recorded_prefix_sampling,epoch_loss,imitation,imitation/checkpoint}.ex \
  test/fixtures/replays/fox_multishine_closed_d1.slp \
  eval_runs/0913_teacher_ingestion/validated/*.frames > "$OUT/sources.sha256"
mix run --no-compile --no-deps-check "${LOAD[@]}" scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --recurrent-state zeros --precision f32 \
  --recorded-frames 'eval_runs/0913_teacher_ingestion/validated/*.frames' \
  --recorded-prefix-weight 64 --recorded-prefix-frames 18 \
  --init-from eval_runs/0913_zero_f32_fit/round21/initial.bin --initial-out "$OUT/initial.bin" \
  --hidden-size 64 --window 16 --action-delay 2 --multi-delay 2 \
  --with-delay-id --queue-depth 3 --prev-action --prev-action-dropout 0.0 --head autoregressive \
  --clean-loss --max-epochs 21 --target-loss 0.0 \
  --out "$OUT/candidate.bin" > "$OUT/train.log" 2>&1
sha256sum "$OUT/initial.bin" "$OUT/candidate.bin" > "$OUT/checkpoints.sha256"
mix run --no-compile --no-deps-check "${LOAD[@]}" -r lib/exphil/eval/teacher_fit.ex \
  scripts/measure_teacher_fit.exs --policy "$OUT/candidate.bin" --include-canonical-rows --out "$OUT/fit.json" \
  > "$OUT/fit.log" 2>&1
mix run --no-compile --no-deps-check --no-start \
  -r lib/exphil/networks/policy/execution_contract.ex -r lib/exphil/eval/early_teacher_gate.ex \
  scripts/check_early_teacher_fit.exs --report "$OUT/fit.json" --out "$OUT/fit_gate.json" \
  > "$OUT/fit_gate.log" 2>&1
mix run --no-compile --no-deps-check "${LOAD[@]}" \
  -r ../libmelee_ex/lib/melee/controller.ex -r ../libmelee_ex/lib/melee/console.ex \
  -r lib/exphil_bridge/melee_port.ex -r lib/exphil/data/action_frame_convention.ex \
  -r lib/exphil/embeddings/player.ex -r lib/exphil/agents/agent.ex \
  -r lib/exphil/eval/scenario_input_timing.ex -r lib/exphil/eval/scenario_history.ex \
  -r lib/exphil/eval/recovery_label_audit.ex \
  scripts/scenario_suite.exs --driver policy --policy "$OUT/candidate.bin" \
  --reaction-delay 2 --temperature 1.0 --character fox --prefix-history cold \
  --manifest eval_runs/0913_matched_handoffs/manifest.json --runs 2 --live-af \
  --no-orphan-sweep --quiet --trace-policy-inputs \
  --out "$OUT/eval_candidate.json" --run-dir "$OUT/eval_candidate" \
  > "$OUT/eval_candidate.log" 2>&1
jq -e '.agent_runtime.af_convention == "libmelee" and .errored_runs == 0
  and .diverged_runs == 0 and .invalid_timing_runs == 0 and (.runs|length == 12)
  and all(.runs[]; .timing_valid == true and .truncated == null
    and .frames_observed == 120 and .details.max_chain >= 10)' "$OUT/eval_candidate.json"
