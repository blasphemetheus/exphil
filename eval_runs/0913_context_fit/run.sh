#!/usr/bin/env bash
# Bounded context-aware fit — HANDOFF_2026-09-13c "Recommended next work" 2-3.
#
# Pool: the canonical fixture + the 18 clips_v6 cold/warm recorded teacher
# clips (10,641 supervised targets; the warm clips carry an 18-frame
# input-only prefix that feeds history/queue but is never a target).
# Recipe: unchanged windowed_gru_f32_v1 proof contract (GRU64 x2, AR head,
# window 16, queue 3, delay 2, prev-action on, dropout 0.0), the SAME
# zero-state/F32 initialization as round21, fresh optimizer.
# Budget (frozen before launch): first 18 supervised targets of every clip
# x64 -> 31,053 draws / 486 batches per epoch; 9 epochs = 4,374 updates.
# Select the ordinary best aggregate-loss export. No budget extension.
#
# Gates, in order, each recorded whether or not the next runs:
#   1. frozen fit + early gate on ALL 18 clips (cold AND warm, supervised idx)
#   2. cold familiar controls   (matched-teacher manifest, cold history, 120f)
#   3. warm familiar controls   (same manifest, committed history, 120f)
#   4. isolated interruptions   (hit 502/1389/3152, committed history,
#                                neutral opponent, 360f) + strict-cycle score
# The frozen gate halts the chain on failure (protocol: no live evaluation
# of an unfit candidate, no checkpoint fishing).
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=${1:-eval_runs/0913_context_fit/round9}
EPOCHS=${2:-9}   # round9: 9 (4,374 updates, FAILED gate 13/18); round21: 21 (10,206 updates = the prior proof's epoch budget)
test ! -e "$OUT"
mkdir -p "$OUT"
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
CLIPS='eval_runs/0913_context_recovery/clips_v6/*.frames'
INIT=eval_runs/0913_zero_f32_fit/round21/initial.bin
FIXTURE=test/fixtures/replays/fox_multishine_closed_d1.slp
COMMON=(--reaction-delay 2 --temperature 1.0 --character fox --live-af
  --no-orphan-sweep --quiet --trace-policy-inputs)

git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
sha256sum "$0" \
  scripts/{dagger_drill,measure_teacher_fit,check_early_teacher_fit,scenario_suite,score_interruption_recovery}.exs \
  ../edifice/lib/edifice/recurrent/recurrent.ex \
  lib/exphil/networks/policy/{execution_contract,backbone}.ex \
  lib/exphil/training/{data,labels,recorded_frames,recorded_context,recorded_prefix_sampling,epoch_loss,epoch_health,imitation,imitation/checkpoint}.ex \
  lib/exphil/eval/{teacher_fit,early_teacher_gate}.ex \
  lib/exphil/agents/{agent,multishine_expert}.ex \
  "$FIXTURE" $CLIPS "$INIT" \
  eval_runs/0913_matched_handoffs/manifest.json eval_runs/0913_interruption_recovery/manifest.json \
  > "$OUT/sources.sha256"

echo "[$(date +%T)] 0/5 train ($EPOCHS epochs, 486 batches/epoch)" | tee -a "$OUT/progress.log"
mix run scripts/dagger_drill.exs \
  --expert multishine --fixture "$FIXTURE" \
  --recurrent-state zeros --precision f32 \
  --recorded-frames "$CLIPS" \
  --recorded-prefix-weight 64 --recorded-prefix-frames 18 \
  --init-from "$INIT" --initial-out "$OUT/initial.bin" \
  --hidden-size 64 --window 16 --action-delay 2 --multi-delay 2 \
  --with-delay-id --queue-depth 3 --prev-action --prev-action-dropout 0.0 --head autoregressive \
  --clean-loss --max-epochs "$EPOCHS" --target-loss 0.0 \
  --out "$OUT/candidate.bin" > "$OUT/train.log" 2>&1
sha256sum "$OUT/initial.bin" "$OUT/candidate.bin" > "$OUT/checkpoints.sha256"
cmp "$OUT/initial.bin" "$INIT" && echo "initial.bin byte-identical to $INIT" >> "$OUT/checkpoints.sha256" || true

echo "[$(date +%T)] 1/5 frozen fit (cold+warm)" | tee -a "$OUT/progress.log"
mix run scripts/measure_teacher_fit.exs --policy "$OUT/candidate.bin" \
  --recorded-frames "$CLIPS" --expected-targets 10641 --include-canonical-rows \
  --out "$OUT/fit.json" > "$OUT/fit.log" 2>&1
set +e
mix run --no-start scripts/check_early_teacher_fit.exs --report "$OUT/fit.json" --out "$OUT/fit_gate.json" \
  > "$OUT/fit_gate.log" 2>&1
GATE=$?
set -e
echo "[$(date +%T)] frozen gate exit $GATE" | tee -a "$OUT/progress.log"
if [[ $GATE -ne 0 ]]; then
  echo "[$(date +%T)] HALT: frozen early gate failed; no live evaluation (see fit_gate.json)" | tee -a "$OUT/progress.log"
  exit 1
fi

live() { # name manifest history window opponent
  local name=$1 manifest=$2 history=$3 window=$4 opponent=$5
  echo "[$(date +%T)] live $name" | tee -a "$OUT/progress.log"
  mix run scripts/scenario_suite.exs --driver policy --policy "$OUT/candidate.bin" "${COMMON[@]}" \
    --prefix-history "$history" --manifest "$manifest" --runs 2 --window "$window" \
    --response-opponent "$opponent" \
    --out "$OUT/$name.json" --run-dir "$OUT/$name" > "$OUT/$name.log" 2>&1 || true
  jq -c '{name: "'"$name"'", errored: .errored_runs, diverged: .diverged_runs, invalid_timing: .invalid_timing_runs,
          runs: (.runs | length), chains: [.runs[] | .details.max_chain]}' "$OUT/$name.json" | tee -a "$OUT/progress.log"
}

echo "[$(date +%T)] 2/5 cold familiar controls" | tee -a "$OUT/progress.log"
live cold_controls eval_runs/0913_matched_handoffs/manifest.json cold 120 replay
echo "[$(date +%T)] 3/5 warm familiar controls" | tee -a "$OUT/progress.log"
live warm_controls eval_runs/0913_matched_handoffs/manifest.json committed 120 replay
echo "[$(date +%T)] 4/5 isolated interruptions (neutral opponent)" | tee -a "$OUT/progress.log"
live interruptions eval_runs/0913_interruption_recovery/manifest.json committed 360 neutral
mix run --no-start scripts/score_interruption_recovery.exs \
  --scores "$OUT/interruptions.json" --out "$OUT/interruptions_recovery.json" \
  > "$OUT/interruptions_recovery.log" 2>&1 || true

echo "[$(date +%T)] 5/5 verdict" | tee -a "$OUT/progress.log"
for name in cold_controls warm_controls; do
  if jq -e '.agent_runtime.af_convention == "libmelee" and .errored_runs == 0 and .diverged_runs == 0
      and .invalid_timing_runs == 0 and (.runs|length == 12)
      and all(.runs[]; .timing_valid == true and .truncated == null
        and .frames_observed == 120 and .details.max_chain >= 10)' "$OUT/$name.json" > /dev/null; then
    echo "$name: PASS 12/12" | tee -a "$OUT/progress.log"
  else
    echo "$name: FAIL ($(jq -c '[.runs[] | .details.max_chain]' "$OUT/$name.json"))" | tee -a "$OUT/progress.log"
  fi
done
echo "[$(date +%T)] done" | tee -a "$OUT/progress.log"
