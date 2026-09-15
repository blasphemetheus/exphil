#!/usr/bin/env bash
# Coverage round, stage 3: train the proof recipe at delay 4 on the FULL
# pool = canonical fixture + the nine original handoffs' delay-4 clips
# (0914_delay4_proof/clips, cold+warm) + the coverage TRAIN split
# (clips_train, cold+warm). Then gate at reaction 4, in order:
#   1. frozen fit + early gate on every TRAIN clip (halts the chain on fail)
#      + informative frozen fit on the HELD-OUT clips
#   2. original 9 handoffs: cold controls, warm controls, interruptions
#      (the regression gates of the delay-4 proof)
#   3. coverage TRAIN manifest, warm, neutral opponent   (learned?)
#   4. coverage HELD-OUT manifest, warm, neutral opponent (generalizes?)
#   5. coverage HELD-OUT manifest, warm, replay opponent  (under pressure)
# Budget declared before launch: 21 epochs (the converging budget of both
# prior delay-2/delay-4 rounds); batch count grows with the pool.
set -uo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0914_coverage_round
R=$OUT/round21
test -e "$R"
mkdir -p "$R"
K=4; Q=5
P=$OUT/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$P"; }
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
FIXTURE=test/fixtures/replays/fox_multishine_closed_d1.slp
TRAIN_GLOB="eval_runs/0914_delay4_proof/clips/*.frames,$OUT/clips_train/*.frames"
HELD_GLOB="$OUT/clips_heldout/*.frames"
targets_of() { jq "[.results[] | select(.export | test(\"$1\")) | .targets] | add" $2; }
ORIG=$(jq '[.results[].targets] | add' eval_runs/0914_delay4_proof/clips/report.json)
TRAIN_NEW=$(mix run --no-start -e 'IO.puts(Path.wildcard("'$OUT'/clips_train/*.frames") |> Enum.map(fn p -> [l] = (p |> File.read!() |> :erlang.binary_to_term()).frame_lists; Enum.count(l, &(&1[:input_only] != true)) end) |> Enum.sum())' 2>/dev/null | tail -1)
HELD_NEW=$(mix run --no-start -e 'IO.puts(Path.wildcard("'$OUT'/clips_heldout/*.frames") |> Enum.map(fn p -> [l] = (p |> File.read!() |> :erlang.binary_to_term()).frame_lists; Enum.count(l, &(&1[:input_only] != true)) end) |> Enum.sum())' 2>/dev/null | tail -1)
# clips are stored UNSHIFTED; the fit applies the delay-K shift (drops K per clip)
N_TRAIN=$(command ls $OUT/clips_train/*.frames | wc -l); N_HELD=$(command ls $OUT/clips_heldout/*.frames | wc -l)
EXPECTED=$(( 7079 - K + ORIG + TRAIN_NEW - K * N_TRAIN ))
EXPECTED_HELD=$(( 7079 - K + HELD_NEW - K * N_HELD ))
say "=== resume gates on the trained candidate (target count fixed: -K per clip)"
say "1/5 frozen fit: train clips (gate) + held-out clips (informative)"
mix run scripts/measure_teacher_fit.exs --policy "$R/candidate.bin" --delay $K --queue-depth $Q \
  --recorded-frames "$TRAIN_GLOB" --expected-targets "$EXPECTED" --out "$R/fit_train.json" > "$R/fit_train.log" 2>&1 || say "  FIT (train) FAILED"
set +e
mix run --no-start scripts/check_early_teacher_fit.exs --report "$R/fit_train.json" --out "$R/fit_gate.json" > "$R/fit_gate.log" 2>&1
GATE=$?
say "  train gate exit $GATE; not ready: $(jq -c '[.cases[] | select(.ready == false) | .case]' $R/fit_gate.json 2>/dev/null)"
mix run scripts/measure_teacher_fit.exs --policy "$R/candidate.bin" --delay $K --queue-depth $Q \
  --recorded-frames "$HELD_GLOB" --expected-targets "$EXPECTED_HELD" --out "$R/fit_heldout.json" > "$R/fit_heldout.log" 2>&1 || say "  FIT (held-out) FAILED"
say "  held-out early: $(jq -c '[.cases[] | select(.case != "canonical") | .first18.tf_argmax_correct] | {n: length, mean: (add/length), min: min}' $R/fit_heldout.json 2>/dev/null)"
if [[ $GATE -ne 0 ]]; then say "HALT: train frozen gate failed; no live evaluation"; exit 1; fi

COMMON=(--reaction-delay $K --temperature 1.0 --character fox --live-af --no-orphan-sweep --quiet --trace-policy-inputs)
live() { local name=$1 manifest=$2 history=$3 window=$4 opponent=$5
  say "live $name"
  mix run scripts/scenario_suite.exs --driver policy --policy "$R/candidate.bin" "${COMMON[@]}" \
    --prefix-history "$history" --manifest "$manifest" --runs 2 --window "$window" --response-opponent "$opponent" \
    --out "$R/$name.json" --run-dir "$R/$name" > "$R/$name.log" 2>&1 || true
  jq -c '{name: "'"$name"'", errored: .errored_runs, diverged: .diverged_runs, invalid_timing: .invalid_timing_runs, runs: (.runs|length),
          chain_ge10: ([.runs[] | select(.details.max_chain >= 10)] | length), chains: [.runs[] | .details.max_chain]}' "$R/$name.json" | tee -a "$P"; }
say "2/5 original 9 handoffs (regression)"
live orig_cold eval_runs/0913_matched_handoffs/manifest.json cold 120 replay
live orig_warm eval_runs/0913_matched_handoffs/manifest.json committed 120 replay
live orig_interruptions eval_runs/0913_interruption_recovery/manifest.json committed 360 neutral
say "3/5 coverage TRAIN handoffs (warm, neutral opponent)"
live cov_train $OUT/manifest_train.json committed 360 neutral
say "4/5 coverage HELD-OUT handoffs (warm, neutral opponent)"
live cov_heldout $OUT/manifest_heldout.json committed 360 neutral
say "5/5 coverage HELD-OUT handoffs (warm, replay opponent)"
live cov_heldout_replay $OUT/manifest_heldout.json committed 360 replay
for n in orig_interruptions cov_train cov_heldout cov_heldout_replay; do
  mix run --no-start scripts/score_interruption_recovery.exs --scores "$R/$n.json" --out "$R/${n}_recovery.json" > "$R/${n}_recovery.log" 2>&1 || true
done
say "stage 3 done"
