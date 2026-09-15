[root] = System.argv()
read = fn path -> root |> Path.join(path) |> File.read!() |> JSON.decode!() end
mined = read.("mined.json")["entries"]
qualified = read.("teacher_qualified.json")["runs"]
by_key = Map.new(mined, &{{&1["slp"], &1["frame"]}, &1})
by_recording = Map.new(qualified, fn run ->
  [recording] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
  {recording, {run["slp"], run["frame"]}}
end)
valid = for result <- read.("clips/report.json")["results"], result["history"] == "cold" do
  key = Map.fetch!(by_recording, result["source_replay"])
  Map.put(Map.fetch!(by_key, key), "sha6", String.slice(result["source_sha256"], 0, 6))
end
# Declared before qualification: each fresh roster's r2 is held out.
{held, train} = Enum.split_with(valid, fn entry ->
  String.contains?(entry["slp"], "/0914_coverage_round/rollouts/") and
    String.ends_with?(entry["slp"], "/r2.slp")
end)
if train == [] or held == [], do: raise("empty training or held-out partition")
sources = fn entries -> MapSet.new(entries, & &1["slp"]) end
unless MapSet.disjoint?(sources.(train), sources.(held)), do: raise("source leakage")
for {name, entries} <- [{"train", train}, {"heldout", held}] do
  File.write!(Path.join(root, "manifest_#{name}.json"), JSON.encode!(%{entries: entries}) <> "\n")
  dest = Path.join(root, "clips_#{name}")
  File.mkdir!(dest)
  for entry <- entries, mode <- ["cold", "warm"] do
    file = "#{entry["frame"]}_#{entry["sha6"]}_#{mode}.frames"
    File.cp!(Path.join([root, "clips", file]), Path.join(dest, file))
  end
end
IO.puts("source-disjoint split: train #{length(train)}, held out #{length(held)} handoffs")
