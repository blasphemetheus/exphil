#!/usr/bin/env bash
# Reaction-4 policy control via the proof recipe (HANDOFF_2026-09-13d NEXT 5 /
# PIPELINE_PROOF gate 1). Question: does a policy trained AT reaction 4 chain
# at --reaction-delay 4 on this harness? ep57 (g-line, ids 3/4/5) chains 1 at
# reaction 4 and nobody has separated "harness rung wrong" from "id 4 badly
# trained". The delay-2 proof passed every gate; this is the same recipe with
# every 2 replaced by 4 (queue depth k+1 = 5, warm prefix 16-1+5 = 20).
#
# Steps (each recorded whether or not the next runs):
#   0. regression: the DEFAULT export must reproduce clips_v6 byte-for-byte
#   1. validate the four teacher scoreboards at delay 4 (check_recovery_targets)
#   2. export cold/warm clips at delay 4 (no parity policy exists yet)
#   3. train: round21 recipe, --action-delay 4 --multi-delay 4 --queue-depth 5
#   4. parity: re-export with the trained candidate as the Agent; cmp clips
#   5. frozen cold+warm fit gate at delay 4
#   6. live at --reaction-delay 4: cold, warm, interruptions neutral + replay
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=${1:-eval_runs/0914_delay4_proof}
K=4; Q=$((K + 1)); CTX=$((16 - 1 + Q))
R=$OUT/round21
test ! -e "$R"; mkdir -p "$R" "$OUT/targets"
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
INIT=eval_runs/0913_zero_f32_fit/round21/initial.bin
FIXTURE=test/fixtures/replays/fox_multishine_closed_d1.slp
P=$OUT/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$P"; }
git rev-parse HEAD > "$OUT/git_head.txt"

say "0/6 regression: default export == clips_v6"
if [[ ! -e $OUT/clips_check_d2 ]]; then
  mix run scripts/prepare_recorded_context.exs --out-dir "$OUT/clips_check_d2" > "$OUT/clips_check_d2.log" 2>&1 || true
fi
ok=1; for f in eval_runs/0913_context_recovery/clips_v6/*.frames; do cmp -s "$f" "$OUT/clips_check_d2/$(basename $f)" || { say "  DIFFERS: $(basename $f)"; ok=0; }; done
[[ $ok == 1 ]] && say "  clips_v6 reproduced byte-for-byte (18 files)" || say "  REGRESSION in the exporter refactor (see clips_check_d2.log)"

say "1/6 validate teacher targets at delay $K"
declare -A SCORES=(
  [neutral]=eval_runs/0913_teacher_ingestion/neutral.json
  [sustain]=eval_runs/0913_recovery_labels/control_teacher.json
  [recovery]=eval_runs/0913_recovery_labels/teacher_grounded.json
  [interruptions]=eval_runs/0913_context_recovery/teacher/eval_candidate.json
)
REPORTS=""
for n in neutral sustain recovery interruptions; do
  mix run --no-start scripts/check_recovery_targets.exs --scores "${SCORES[$n]}" --delay $K --allow-on-loop \
    --out "$OUT/targets/${n}_d$K.json" > "$OUT/targets/${n}_d$K.log" 2>&1
  say "  $n: valid=$(jq -r .valid $OUT/targets/${n}_d$K.json) runs=$(jq '.runs|length' $OUT/targets/${n}_d$K.json) targets=$(jq -c '[.runs[].targets]' $OUT/targets/${n}_d$K.json)"
  REPORTS="$REPORTS,$OUT/targets/${n}_d$K.json"
done
REPORTS=${REPORTS#,}

say "2/6 export clips at delay $K (queue $Q, warm context $CTX; parity skipped)"
mix run scripts/prepare_recorded_context.exs --out-dir "$OUT/clips" --delay $K --queue-depth $Q --context $CTX \
  --reports "$REPORTS" --policy none > "$OUT/clips.log" 2>&1
CLIPS="$OUT/clips/*.frames"
say "  $(command ls $CLIPS | wc -l) clips; targets: $(jq -c '[.results[].targets]' $OUT/clips/report.json)"
EXPECTED=$(( 7079 - K + $(jq '[.results[].targets] | add' $OUT/clips/report.json) ))
say "  expected pool targets (canonical 7079-$K + clips): $EXPECTED"

say "3/6 train (21 epochs, delay $K, queue $Q)"
sha256sum "$0" scripts/{dagger_drill,measure_teacher_fit,check_early_teacher_fit,prepare_recorded_context,scenario_suite,score_interruption_recovery}.exs \
  lib/exphil/training/{data,labels,recorded_frames,recorded_context,recorded_prefix_sampling}.ex \
  lib/exphil/agents/{agent,multishine_expert}.ex "$FIXTURE" $CLIPS "$INIT" > "$R/sources.sha256"
mix run scripts/dagger_drill.exs \
  --expert multishine --fixture "$FIXTURE" \
  --recurrent-state zeros --precision f32 \
  --recorded-frames "$CLIPS" \
  --recorded-prefix-weight 64 --recorded-prefix-frames 18 \
  --init-from "$INIT" --initial-out "$R/initial.bin" \
  --hidden-size 64 --window 16 --action-delay $K --multi-delay $K \
  --with-delay-id --queue-depth $Q --prev-action --prev-action-dropout 0.0 --head autoregressive \
  --clean-loss --max-epochs 21 --target-loss 0.0 \
  --out "$R/candidate.bin" > "$R/train.log" 2>&1
sha256sum "$R/initial.bin" "$R/candidate.bin" > "$R/checkpoints.sha256"
say "  $(grep -oE 'epoch 21/21: loss=[0-9.e-]+' $R/train.log || echo 'TRAIN FAILED')"

say "4/6 parity: re-export with the trained candidate as the Agent"
mix run scripts/prepare_recorded_context.exs --out-dir "$OUT/clips_parity" --delay $K --queue-depth $Q --context $CTX \
  --reports "$REPORTS" --policy "$R/candidate.bin" > "$OUT/clips_parity.log" 2>&1 \
  && say "  parity audit passed: max |diff| $(jq '[.results[].maximum_absolute_difference] | max' $OUT/clips_parity/report.json)" \
  || say "  PARITY AUDIT FAILED (clips_parity.log)"
ok=1; for f in $CLIPS; do cmp -s "$f" "$OUT/clips_parity/$(basename $f)" || ok=0; done
[[ $ok == 1 ]] && say "  parity clips == training clips byte-for-byte" || say "  parity clips DIFFER from training clips"

say "5/6 frozen fit gate at delay $K"
mix run scripts/measure_teacher_fit.exs --policy "$R/candidate.bin" --delay $K --queue-depth $Q \
  --recorded-frames "$CLIPS" --expected-targets "$EXPECTED" --include-canonical-rows \
  --out "$R/fit.json" > "$R/fit.log" 2>&1 || say "  FIT MEASURE FAILED (fit.log)"
set +e
mix run --no-start scripts/check_early_teacher_fit.exs --report "$R/fit.json" --out "$R/fit_gate.json" > "$R/fit_gate.log" 2>&1
GATE=$?; set -e
say "  frozen gate exit $GATE: $(jq -c '[.cases[] | select(.ready == false) | .case]' $R/fit_gate.json 2>/dev/null) not ready"

COMMON=(--reaction-delay $K --temperature 1.0 --character fox --live-af --no-orphan-sweep --quiet --trace-policy-inputs)
live() { local name=$1 manifest=$2 history=$3 window=$4 opponent=$5
  say "live $name"
  mix run scripts/scenario_suite.exs --driver policy --policy "$R/candidate.bin" "${COMMON[@]}" \
    --prefix-history "$history" --manifest "$manifest" --runs 2 --window "$window" --response-opponent "$opponent" \
    --out "$R/$name.json" --run-dir "$R/$name" > "$R/$name.log" 2>&1 || true
  jq -c '{name: "'"$name"'", errored: .errored_runs, diverged: .diverged_runs, invalid_timing: .invalid_timing_runs, runs: (.runs|length), chains: [.runs[] | .details.max_chain]}' "$R/$name.json" | tee -a "$P"; }
say "6/6 live at reaction $K (the gate is informative here, not a halt: the question is the harness rung)"
live cold_controls eval_runs/0913_matched_handoffs/manifest.json cold 120 replay
live warm_controls eval_runs/0913_matched_handoffs/manifest.json committed 120 replay
live interruptions eval_runs/0913_interruption_recovery/manifest.json committed 360 neutral
live interruptions_replay_opp eval_runs/0913_interruption_recovery/manifest.json committed 360 replay
for n in interruptions interruptions_replay_opp; do
  mix run --no-start scripts/score_interruption_recovery.exs --scores "$R/$n.json" --out "$R/${n}_recovery.json" > "$R/${n}_recovery.log" 2>&1 || true
done
say "done"
