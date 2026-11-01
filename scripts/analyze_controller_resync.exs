# AI-generated. Authorship marker and method: the sweep's TABLES.md report.
# Classify recorded service times using the separately validated replay cadence.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep

defmodule ControllerResyncAnalysis do
  def stats([]), do: %{samples: 0}

  def stats(samples) do
    service = Enum.map(samples, &Enum.at(&1, 1)) |> Enum.sort()
    n = length(service)

    %{
      samples: n,
      service_median_us: Enum.at(service, div(n, 2)),
      service_p99_us: Enum.at(service, ceil(n * 0.99) - 1),
      missed_deadlines: Enum.count(samples, &(Enum.at(&1, 2) > 1_000_000 / 60))
    }
  end

  def run(root) do
    manifest = Sweep.read(Path.join(root, "manifest.json"))
    true = manifest["status"] == "complete"

    comparisons =
      for run <- manifest["runs"], String.ends_with?(run["label"], "_resync") do
        true = run["competing_gpu_pids"] == []
        report = Sweep.read(Path.join(root, run["label"] <> ".json"))
        period = report["resync"] + 1

        schedules =
          Map.new(report["resync_validation"], fn validation ->
            true = validation["replay_cadence_validated"]

            events =
              validation["events"]
              |> Enum.take(period)
              |> Map.new(&{&1["tick"], &1["rows"]})

            {validation["schedule"], events}
          end)

        raw =
          Path.join(root, report["samples_file"])
          |> File.read!()
          |> :zlib.gunzip()
          |> Jason.decode!()

        for {result, recorded} <- Enum.zip(raw, report["results"]) do
          true = Map.delete(result, "streams") == recorded
          events = Map.fetch!(schedules, result["schedule"])

          classified =
            for {stream, row} <- Enum.with_index(result["streams"]),
                {sample, tick} <- Enum.with_index(stream) do
              replay_rows = Map.fetch!(events, rem(tick, period))

              replay? =
                if result["mode"] == "batch", do: replay_rows != [], else: row in replay_rows

              {replay?, sample}
            end

          Map.take(result, ["mode", "schedule", "round", "contention"])
          |> Map.put("client", report["client"])
          |> Map.put("replay", stats(for {true, sample} <- classified, do: sample))
          |> Map.put("ordinary", stats(for {false, sample} <- classified, do: sample))
        end
      end

    Sweep.write(Path.join(root, "resync_service.json"), %{
      method:
        "Service times classified by validated replay schedule; response includes backlog and is not a causal attribution",
      results: List.flatten(comparisons)
    })

    IO.puts("Classified replay and ordinary calls without adding work to the timed path")
  end
end

[root] = System.argv()
ControllerResyncAnalysis.run(root)
