# AI-generated; authorship and method: docs/guides/MAMBA_CARRY_RESOLUTION.md.
# devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_mamba_accuracy.exs ROOT OUT.json
[root, out] = System.argv()

reports =
  for path <- Path.wildcard(Path.join(root, "*/report.json")) do
    report = path |> File.read!() |> Jason.decode!()

    comparisons =
      for comparison <- [
            "f32_batch_vs_single",
            "f64_batch_vs_single",
            "batch32_vs_reference",
            "single32_vs_reference"
          ],
          into: %{} do
        metrics =
          for leaf <- ["conv", "output", "h"], into: %{} do
            rows = Enum.map(report["rows"], & &1[comparison][leaf])

            {leaf,
             %{
               failed_values: Enum.sum(Enum.map(rows, & &1["failed_values"])),
               max_abs: Enum.max(Enum.map(rows, & &1["max_abs"])),
               max_error_over_bound: Enum.max(Enum.map(rows, & &1["max_error_over_bound"]))
             }}
          end

        {comparison, metrics}
      end

    %{
      file: Path.relative_to(path, root),
      passed: report["reference_checks_passed"],
      client: report["client"],
      mode: report["mode"],
      ticks: report["ticks"],
      comparisons: comparisons
    }
  end

timings =
  for path <- Path.wildcard(Path.join(root, "*/capture_metadata.json")) do
    metadata = path |> File.read!() |> Jason.decode!()

    groups =
      metadata["agent_timings"]
      |> Enum.drop(160)
      |> Enum.group_by(fn frame ->
        if Enum.any?(frame["reset"], &(&1 != 0)), do: "resync", else: "ordinary"
      end)

    stats =
      Map.new(groups, fn {name, frames} ->
        times = frames |> Enum.map(& &1["agent_us"]) |> Enum.sort()
        n = length(times)

        {name,
         %{
           count: n,
           median_us: Enum.at(times, div(n, 2)),
           p95_us: Enum.at(times, ceil(n * 0.95) - 1),
           max_us: List.last(times)
         }}
      end)

    %{
      file: Path.relative_to(path, root),
      mode: metadata["capture_mode"] || "baseline",
      client: metadata["client"],
      stagger: metadata["stagger"],
      warmup_frames: 160,
      stats: stats
    }
  end

File.write!(out, Jason.encode!(%{reports: reports, agent_timings: timings}, pretty: true) <> "\n")

ExPhil.Training.Output.puts(
  "Wrote #{out}: #{length(reports)} numerical reports, #{length(timings)} timing captures"
)
