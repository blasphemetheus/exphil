#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
OUT=eval_runs/0915_float_input/controls01
mkdir "$OUT"
exec > >(tee -a "$OUT/progress.log") 2>&1
export EXLA_TARGET=host
DOLPHIN=/home/blewf/.local/share/slippi/exi-ai-float-v3/dolphin-emu-headless
POLICY=eval_runs/0914_delay4_proof/round21/candidate.bin
sha256sum "$POLICY" "$DOLPHIN" > "$OUT/sources.sha256"
for transport in pipe direct; do
  EXTRA=()
  if [[ $transport == direct ]]; then EXTRA=(--direct-inputs --no-pipe-shim); fi
  for history in cold committed; do
    name="${transport}_${history}"
    echo "$(date -Is) START $name"
    mix run scripts/scenario_suite.exs --driver policy --policy "$POLICY" --dolphin "$DOLPHIN" \
      --reaction-delay 4 --temperature 1.0 --character fox --live-af --no-orphan-sweep --quiet --trace-policy-inputs \
      --prefix-history "$history" --manifest eval_runs/0913_matched_handoffs/manifest.json --runs 2 --window 120 --response-opponent replay \
      --out "$OUT/$name.json" --run-dir "$OUT/$name" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    jq -c '{errored: .errored_runs, diverged: .diverged_runs, invalid_timing: .invalid_timing_runs, runs: (.runs|length), passes: ([.runs[] | select(.pass==true)]|length), chains:[.runs[].details.max_chain]}' "$OUT/$name.json"
  done
done
echo "$(date -Is) DONE"
