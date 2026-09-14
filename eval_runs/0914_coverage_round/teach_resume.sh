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
say "=== resume 2d (clips reused)"
say "=== stage 2d: train / held-out split"
mix run --no-start -e '
out = "'$OUT'"
mined = File.read!(Path.join(out, "mined.json")) |> Jason.decode!() |> Map.fetch!("entries")
report = File.read!(Path.join(out, "clips/report.json")) |> Jason.decode!()
by_key = Map.new(mined, &{{&1["slp"], &1["frame"]}, &1})
# a clip's source_replay is the TEACHER's recording; map it back to the mined
# (gen2 source, frame) through the teacher scoreboard's run directories
teacher = File.read!(Path.join(out, "teacher_qualified.json")) |> Jason.decode!() |> Map.fetch!("runs")
by_recording = Map.new(teacher, fn run -> [rec] = Path.wildcard(Path.join(run["replay_dir"], "*.slp")); {rec, {run["slp"], run["frame"]}} end)
valid = report["results"] |> Enum.filter(&(&1["history"] == "cold")) |> Enum.map(fn r -> Map.put(Map.fetch!(by_key, Map.fetch!(by_recording, r["source_replay"])), "sha6", String.slice(r["source_sha256"], 0, 6)) end)
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
