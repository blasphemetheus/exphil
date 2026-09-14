#!/usr/bin/env bash
# Follow-up to run.sh: (a) cold controls again (2 dolphin_disconnected errors +
# sustain-900 chains 7/1 in the first pass), (b) the ORIGINAL replay-opponent
# interruptions (the human keeps attacking; baseline 0/6 strict cycles).
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=eval_runs/0913_context_fit/round21
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
COMMON=(--reaction-delay 2 --temperature 1.0 --character fox --live-af --no-orphan-sweep --quiet --trace-policy-inputs)
live() { local name=$1 manifest=$2 history=$3 window=$4 opponent=$5
  echo "[$(date +%T)] live $name" | tee -a "$D/progress.log"
  mix run scripts/scenario_suite.exs --driver policy --policy "$D/candidate.bin" "${COMMON[@]}" \
    --prefix-history "$history" --manifest "$manifest" --runs 2 --window "$window" --response-opponent "$opponent" \
    --out "$D/$name.json" --run-dir "$D/$name" > "$D/$name.log" 2>&1 || true
  jq -c '{name: "'"$name"'", errored: .errored_runs, diverged: .diverged_runs, invalid_timing: .invalid_timing_runs, runs: (.runs|length), chains: [.runs[] | .details.max_chain]}' "$D/$name.json" | tee -a "$D/progress.log"; }
live cold_controls_rerun eval_runs/0913_matched_handoffs/manifest.json cold 120 replay
live interruptions_replay_opp eval_runs/0913_interruption_recovery/manifest.json committed 360 replay
mix run --no-start scripts/score_interruption_recovery.exs --scores "$D/interruptions_replay_opp.json" --out "$D/interruptions_replay_opp_recovery.json" > "$D/interruptions_replay_opp_recovery.log" 2>&1 || true
echo "[$(date +%T)] rerun done" | tee -a "$D/progress.log"
