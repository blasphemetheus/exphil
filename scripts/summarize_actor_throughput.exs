# devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_actor_throughput.exs [DIRECTORY]
Code.require_file("lib/benchmark_sweep.exs", __DIR__)

defmodule ActorThroughputSummary do
  alias ExPhil.BenchmarkSweep, as: Sweep

  def stats(values) do
    values = Enum.sort(values)
    n = length(values)

    %{
      "samples" => n,
      "median_us" => Enum.at(values, div(n, 2)),
      "p95_us" => Enum.at(values, ceil(n * 0.95) - 1),
      "p99_us" => Enum.at(values, ceil(n * 0.99) - 1),
      "max_us" => List.last(values)
    }
  end

  def ms(us), do: :erlang.float_to_binary(us / 1000, decimals: 2)

  def run(root, allow_partial) do
    manifest = Sweep.read(Path.join(root, "manifest.json"))

    if allow_partial do
      true = Sweep.hashes(Map.keys(manifest["hashes"])) == manifest["hashes"]
      true = manifest["runs"] != []
    else
      true = manifest["hashes_verified_after"] and length(manifest["runs"]) == 8
    end

    true = Enum.all?(manifest["runs"], &(&1["competing_gpu_pids"] == []))
    labels = MapSet.new(manifest["runs"], & &1["label"])

    lines =
      for client <- ["host", "cuda"],
          arch <- ["min_gru", "gru", "mamba", "mamba2"],
          MapSet.member?(labels, "#{arch}_#{client}") do
        report = Sweep.read(Path.join(root, "#{arch}_#{client}.json"))

        raw =
          Path.join(root, report["samples_file"])
          |> File.read!()
          |> :zlib.gunzip()
          |> Jason.decode!()

        true = length(raw) == 12 and length(report["results"]) == 12

        for {result, recorded} <- Enum.zip(raw, report["results"]) do
          true = Map.delete(result, "streams") == recorded
          samples = Enum.concat(result["streams"])
          true = length(samples) == result["calls"]
          true = Enum.map(result["streams"], &length/1) == result["worker_calls"]
          true = stats(Enum.map(samples, &Enum.at(&1, 1))) == result["service"]
          true = stats(Enum.map(samples, &Enum.at(&1, 2))) == result["response"]

          true =
            Enum.all?(samples, fn [_, service, response] ->
              response >= service and service > 0
            end)

          true = result["elapsed_us"] >= report["seconds"] * 1_000_000
          multiplier = if result["mode"] == "batch", do: result["rows"], else: 1
          true = result["actions"] == result["calls"] * multiplier

          if result["rate"] == "paced" do
            true = Enum.all?(result["worker_calls"], &(&1 == report["seconds"] * 60))

            true =
              result["deadline_misses"] == Enum.count(samples, &(Enum.at(&1, 2) > 1_000_000 / 60))
          end
        end

        for mode <- ["single", "actors", "batch"], load <- [false, true] do
          cases =
            report["results"]
            |> Enum.filter(&(&1["mode"] == mode and &1["contention"] == load))
            |> Map.new(&{&1["rate"], &1})

          a = cases["saturated"]
          b = cases["paced"]

          "| #{client} | #{arch} | #{mode} | #{if load, do: "matmul", else: "idle"} | " <>
            "#{round(a["actions_per_second"])} | #{ms(a["service"]["p95_us"])} | " <>
            "#{ms(b["response"]["p95_us"])} / #{ms(b["response"]["p99_us"])} | " <>
            "#{b["deadline_misses"]} / #{b["calls"]} |"
        end
      end

    header = [
      "# Continuous actor throughput screen#{if length(manifest["runs"]) < 8, do: " (partial)", else: ""}",
      "",
      "Throughput counts individual controller actions, including every row of a batch.",
      "Paced response latency includes delay beyond the scheduled frame time; service latency does not.",
      "Misses finish more than 16.67 ms after scheduling (one batch miss affects all eight rows).",
      "Short stress screens, not deployment guarantees. See RESULTS.md for methods and limitations.",
      "",
      "| Client | Policy | Mode | GPU load | Saturated actions/s | Saturated service p95 ms | 60 Hz response p95 / p99 ms | Missed calls / offered |",
      "|---|---|---|---|---:|---:|---:|---:|"
    ]

    File.write!(
      Path.join(root, "TABLES.md"),
      Enum.join(header ++ List.flatten(lines), "\n") <> "\n"
    )

    IO.puts("Verified all raw samples, percentiles, deadlines, counts and artifact consistency")
  end
end

{opts, paths, []} = OptionParser.parse(System.argv(), strict: [allow_partial: :boolean])

ActorThroughputSummary.run(
  List.first(paths) || "eval_runs/actor_throughput_20261026",
  opts[:allow_partial] || false
)
