# Coverage round train / held-out split (2026-09-14).
#
#   mix run --no-start scripts/split_coverage_clips.exs --dir eval_runs/0914_coverage_round
#
# Inputs in DIR: mined.json (the mined handoffs, with class/opp_char),
# teacher_qualified.json (the teacher scoreboard runs that qualified),
# clips/report.json + clips/*.frames (exported cold/warm clips at the
# training delay; a clip's `source_replay` is the TEACHER's own recording,
# mapped back to the mined (source game, frame) through the run directory).
#
# Rule: within each class (neutral, hit), sorted by (opp_char, source, frame),
# every 4th validated handoff is HELD OUT (gate-only); the rest train.
# Writes manifest_train.json / manifest_heldout.json (suite manifests that
# point at the SOURCE games) and copies clips into clips_train/ clips_heldout/.
{opts, [], []} = OptionParser.parse(System.argv(), strict: [dir: :string, every: :integer])
out = Keyword.fetch!(opts, :dir)
every = opts[:every] || 4
read = fn name -> File.read!(Path.join(out, name)) |> Jason.decode!() end

mined = read.("mined.json")["entries"]
by_key = Map.new(mined, &{{&1["slp"], &1["frame"]}, &1})

by_recording =
  Map.new(read.("teacher_qualified.json")["runs"], fn run ->
    [rec] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
    {rec, {run["slp"], run["frame"]}}
  end)

valid =
  read.("clips/report.json")["results"]
  |> Enum.filter(&(&1["history"] == "cold"))
  |> Enum.map(fn r ->
    entry = Map.fetch!(by_key, Map.fetch!(by_recording, r["source_replay"]))
    Map.put(entry, "sha6", String.slice(r["source_sha256"], 0, 6))
  end)

{train, held} =
  valid
  |> Enum.group_by(& &1["class"])
  |> Enum.flat_map(fn {_, es} ->
    es
    |> Enum.sort_by(&{&1["opp_char"], &1["slp"], &1["frame"]})
    |> Enum.with_index()
    |> Enum.map(fn {e, i} -> {e, rem(i, every) == every - 1} end)
  end)
  |> Enum.split_with(fn {_, held?} -> not held? end)

train = Enum.map(train, &elem(&1, 0))
held = Enum.map(held, &elem(&1, 0))

for {name, entries} <- [{"manifest_train.json", train}, {"manifest_heldout.json", held}],
    do: File.write!(Path.join(out, name), Jason.encode!(%{entries: entries}, pretty: true))

for {dir, entries} <- [{"clips_train", train}, {"clips_heldout", held}] do
  File.mkdir_p!(Path.join(out, dir))

  for e <- entries, mode <- ["cold", "warm"] do
    file = "#{e["frame"]}_#{e["sha6"]}_#{mode}.frames"
    File.cp!(Path.join(out, "clips/#{file}"), Path.join(out, "#{dir}/#{file}"))
  end
end

freq = fn es -> es |> Enum.frequencies_by(& &1["class"]) |> inspect() end
opp = fn es -> es |> Enum.frequencies_by(& &1["opp_char"]) |> inspect() end
IO.puts("train #{length(train)} handoffs (#{freq.(train)}; opp #{opp.(train)}), held-out #{length(held)} (#{freq.(held)}; opp #{opp.(held)})")
