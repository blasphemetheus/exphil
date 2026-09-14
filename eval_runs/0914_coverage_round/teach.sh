#!/usr/bin/env bash
# Coverage round, stage 2: mine held-out handoffs from the stage-1 rollouts
# (+ Bradley's 09-13 session + two of the old Fox-CPU rollouts), execute
# the TEACHER from every handoff (neutral opponent after handoff, 360 f,
# committed history, reaction 4), validate its issued inputs + delay-4
# targets against its own recording, export cold/warm clips at delay 4
# (queue 5, warm context 20), and draw the train/held-out split.
#
# Split rule (deterministic, declared here): within each class (neutral,
# hit), sorted by (opp_char, replay, frame), every 4th VALIDATED handoff is
# HELD OUT (gate-only); the rest train. The nine original handoffs stay in
# the pool and remain the regression gate.
set -uo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0914_coverage_round
POLICY=eval_runs/0914_delay4_proof/round21/candidate.bin
K=4; Q=5; CTX=20
P=$OUT/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$P"; }
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15

say "=== stage 2a: mine handoffs"
# sources = the gen2 input-driven games (stage 1b), never the raw CPU rollouts
SOURCES=$(command ls $OUT/gen2/run*/*.slp)
rm -f $OUT/mined.json
mix run --no-start scripts/mine_coverage_handoffs.exs --out $OUT/mined.json \
  --per-replay-neutral 3 --per-replay-hit 3 --min-frame 240 --max-frame 4800 --gap 300 $SOURCES > $OUT/mine.log 2>&1
say "  $(tail -1 $OUT/mine.log); by class: $(jq -c '[.entries[].class] | group_by(.) | map({(.[0]): length}) | add' $OUT/mined.json); by opp: $(jq -c '[.entries[].opp_char] | group_by(.) | map({(.[0]|tostring): length}) | add' $OUT/mined.json)"

say "=== stage 2b: teacher executes every handoff (neutral opponent, 360 f)"
rm -rf $OUT/teacher $OUT/teacher.json
mix run scripts/scenario_suite.exs --driver teacher --policy "$POLICY" \
  --reaction-delay $K --temperature 1.0 --character fox --prefix-history committed \
  --manifest $OUT/mined.json --runs 1 --window 360 --response-opponent neutral --live-af \
  --no-orphan-sweep --quiet --trace-policy-inputs \
  --out $OUT/teacher.json --run-dir $OUT/teacher > $OUT/teacher.log 2>&1
say "  teacher exit $?: errored=$(jq .errored_runs $OUT/teacher.json) diverged=$(jq .diverged_runs $OUT/teacher.json) invalid_timing=$(jq .invalid_timing_runs $OUT/teacher.json) runs=$(jq '.runs|length' $OUT/teacher.json) pass=$(jq '[.runs[] | select(.pass == true)] | length' $OUT/teacher.json)"
say "  chains: $(jq -c '[.runs[] | .details.max_chain]' $OUT/teacher.json)"

say "=== stage 2c: keep qualified runs, validate at delay $K, export clips"
jq '.runs |= map(select(.error == null and .diverged == false and .pass == true and .truncated == null and .timing_valid == true))' \
  $OUT/teacher.json > $OUT/teacher_qualified.json
say "  qualified runs: $(jq '.runs|length' $OUT/teacher_qualified.json)"
rm -f $OUT/targets_d$K.json $OUT/targets_d$K.frames
mix run --no-start scripts/check_recovery_targets.exs --scores $OUT/teacher_qualified.json --delay $K --allow-on-loop \
  --out $OUT/targets_d$K.json > $OUT/targets_d$K.log 2>&1
say "  targets: valid=$(jq -r .valid $OUT/targets_d$K.json) runs=$(jq '.runs|length' $OUT/targets_d$K.json) mismatches=$(jq -c '[.runs[].mismatches] | add' $OUT/targets_d$K.json 2>/dev/null)"
rm -rf $OUT/clips
mix run scripts/prepare_recorded_context.exs --out-dir $OUT/clips --delay $K --queue-depth $Q --context $CTX \
  --reports $OUT/targets_d$K.json --policy none --tag-source > $OUT/clips.log 2>&1
say "  clips: $(command ls $OUT/clips/*.frames 2>/dev/null | wc -l) files, targets $(jq '[.results[].targets] | add' $OUT/clips/report.json 2>/dev/null)"

say "=== stage 2d: train / held-out split"
mix run --no-start -e '
out = "'$OUT'"
mined = File.read!(Path.join(out, "mined.json")) |> Jason.decode!() |> Map.fetch!("entries")
report = File.read!(Path.join(out, "clips/report.json")) |> Jason.decode!()
by_key = Map.new(mined, &{{&1["slp"], &1["frame"]}, &1})
valid = report["results"] |> Enum.filter(&(&1["history"] == "cold")) |> Enum.map(fn r -> Map.put(by_key[{r["source_replay"], r["handoff"]}], "sha6", String.slice(r["source_sha256"], 0, 6)) end)
{train, held} =
  valid
  |> Enum.group_by(& &1["class"])
  |> Enum.flat_map(fn {_, es} ->
    es |> Enum.sort_by(&{&1["opp_char"], &1["slp"], &1["frame"]}) |> Enum.with_index() |> Enum.map(fn {e, i} -> {e, rem(i, 4) == 3} end)
  end)
  |> Enum.split_with(fn {_, held?} -> not held? end)
train = Enum.map(train, &elem(&1, 0)); held = Enum.map(held, &elem(&1, 0))
File.write!(Path.join(out, "manifest_train.json"), Jason.encode!(%{entries: train}, pretty: true))
File.write!(Path.join(out, "manifest_heldout.json"), Jason.encode!(%{entries: held}, pretty: true))
File.mkdir_p!(Path.join(out, "clips_train")); File.mkdir_p!(Path.join(out, "clips_heldout"))
for e <- train, mode <- ["cold", "warm"], do: File.cp!(Path.join(out, "clips/#{e["frame"]}_#{e["sha6"]}_#{mode}.frames"), Path.join(out, "clips_train/#{e["frame"]}_#{e["sha6"]}_#{mode}.frames"))
for e <- held, mode <- ["cold", "warm"], do: File.cp!(Path.join(out, "clips/#{e["frame"]}_#{e["sha6"]}_#{mode}.frames"), Path.join(out, "clips_heldout/#{e["frame"]}_#{e["sha6"]}_#{mode}.frames"))
IO.puts("train #{length(train)} handoffs (#{Enum.frequencies_by(train, & &1["class"]) |> inspect}), held-out #{length(held)} (#{Enum.frequencies_by(held, & &1["class"]) |> inspect})")
' > $OUT/split.log 2>&1
say "  $(tail -1 $OUT/split.log)"
say "stage 2 done"
