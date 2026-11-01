# AI-generated. Rebuild the pilot's table from its recorded JSON measurements.
[directory] = System.argv()
manifest = File.read!(Path.join(directory, "manifest.json")) |> Jason.decode!()
rows = manifest["runs"]

Enum.each(rows, fn row ->
  true = row["equal_steps"]["steps"] == manifest["settings"]["steps"]
  true = row["equal_time"]["training_seconds"] >= manifest["settings"]["seconds"]
  split = File.read!(Path.join([directory, row["label"], "split.json"])) |> Jason.decode!()
  hashes = row["replay_hashes"]
  train = MapSet.new(split["train"], &Map.fetch!(hashes, &1))
  validation = MapSet.new(split["validation"], &Map.fetch!(hashes, &1))
  true = MapSet.disjoint?(train, validation)
end)

format = fn value -> :erlang.float_to_binary(value * 1.0, decimals: 3) end

table =
  for row <- rows, budget <- ["equal_steps", "equal_time"] do
    result = row[budget]
    warm = result["training_seconds"] - result["first_step_seconds"]
    ms = if result["steps"] > 1, do: warm * 1000 / (result["steps"] - 1), else: 0.0

    "| #{row["label"]} | #{budget} | #{result["steps"]} | #{format.(result["training_seconds"])} | #{format.(result["validation_loss"])} | #{format.(ms)} |"
  end

report = """
# Mamba learning pilot results

**AI-generated.** Read `../README.md` for the method and limits.

Status: #{manifest["status"] || "in progress"}; #{length(rows)} completed runs.
All models start fresh on the same 29 training games and three held-out games.
Two seeds are a pilot, not a reliable estimate of uncertainty.

| Model and seed | Budget | Updates | Training seconds | Held-out loss | Warm ms/update |
| --- | --- | ---: | ---: | ---: | ---: |
#{Enum.join(table, "\n")}

Lower held-out loss is better on this objective. Warm ms/update subtracts the
first training step (which includes compilation). It still includes batch
loading and epoch boundaries; it is not a GPU kernel benchmark.

Equal-time budgets include first-step compilation but exclude dataset setup,
validation and policy export. GPU memory snapshots are in the per-arm JSON.
Replay/source hashes and contention observations are in `manifest.json`.

These results do not establish gameplay strength or convergence on the full
corpus. Policy files are local and are not promoted automatically.

**AI-generated poem**

We give each learner time and ground;
then write the limits of what we found.
"""

File.write!(Path.join(directory, "RESULTS.md"), report)
IO.puts(report)

play = Path.wildcard(Path.join([directory, "play", "*", "league.json"]))

if play != [] do
  lines =
    Enum.map(play, fn path ->
      data = File.read!(path) |> Jason.decode!()
      [pair] = data["pairs"]
      {finished, open} = Enum.split_with(pair["games"], & &1["finished"])

      counts = fn games ->
        wins = Enum.count(games, &(&1["a_score"] == 1.0))
        losses = Enum.count(games, &(&1["a_score"] == 0.0))
        "#{wins}/#{losses}/#{length(games) - wins - losses}"
      end

      "| #{pair["a"]} | #{counts.(finished)} | #{counts.(open)} |"
    end)

  text = """

  ## Short simulator readout

  Each snapshot faces the same frozen master-ep3 Fox, both ports, four environments,
  30 seconds per session, Final Destination, zero delay. Policy sampling remains
  stochastic. Simulator seeds are 71/72. This is a small compatibility/gameplay
  pilot. A lead at the time limit is not a completed-match win.

  | Snapshot | Completed wins/losses/draws | Time-limit leads/trails/ties |
  | --- | ---: | ---: |
  #{Enum.join(lines, "\n")}
  """

  File.write!(Path.join(directory, "RESULTS.md"), text, [:append])
  IO.puts(text)
end
