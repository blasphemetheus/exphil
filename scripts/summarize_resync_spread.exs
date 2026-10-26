Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep
root = List.first(System.argv()) || "eval_runs/resync_spread_20261026"
manifest = Sweep.read(Path.join(root, "manifest.json"))
true = manifest["status"] == "complete" and length(manifest["runs"]) == 2

lines =
  for run <- manifest["runs"] do
    true = run["competing_gpu_pids"] == []
    client = run["client"]
    report = Sweep.read(Path.join(root, "mamba_#{client}.json"))

    raw =
      Path.join(root, report["samples_file"]) |> File.read!() |> :zlib.gunzip() |> Jason.decode!()

    true = length(raw) == 16

    for {result, recorded} <- Enum.zip(raw, report["results"]) do
      true = Map.delete(result, "streams") == recorded
      samples = Enum.concat(result["streams"])
      true = length(samples) == result["calls"]
      true = Enum.all?(result["streams"], &(length(&1) == report["seconds"] * 60))
      response = samples |> Enum.map(&Enum.at(&1, 2)) |> Enum.sort()
      n = length(response)
      true = result["deadline_misses"] == Enum.count(response, &(&1 > 1_000_000 / 60))
      true = result["response"]["p95_us"] == Enum.at(response, ceil(n * 0.95) - 1)
      true = result["response"]["p99_us"] == Enum.at(response, ceil(n * 0.99) - 1)
      ms = fn value -> :erlang.float_to_binary(value / 1000, decimals: 2) end

      "| #{client} | #{result["mode"]} | #{result["schedule"]} | #{result["round"]} | " <>
        "#{if result["contention"], do: "matmul", else: "idle"} | " <>
        "#{ms.(result["response"]["p95_us"])} / #{ms.(result["response"]["p99_us"])} | " <>
        "#{result["deadline_misses"]} / #{n} |"
    end
  end

header = [
  "# Mamba resync schedules — AI-generated measurements",
  "",
  "Each row is a separate #{manifest["seconds"]}-second phase at 60 Hz. One late batch affects eight bots.",
  "Round two reverses schedule order. Setup and carry validation are outside the timed phases.",
  "",
  "| Client | Mode | Schedule | Round | Load | Response p95 / p99 ms | Missed / offered calls |",
  "|---|---|---|---:|---|---:|---:|"
]

File.write!(Path.join(root, "TABLES.md"), Enum.join(header ++ List.flatten(lines), "\n") <> "\n")
IO.puts("Verified raw samples and deadline counts for both clients")
