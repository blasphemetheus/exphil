# Preserve compact, reviewable results; large detailed reports stay local.
reports = ~w(human_interpreter human_jit_fixed human_jit_nofma nmsub_cpu_smoke nmsub_longest nmsub_matched human_option)
summary = Map.new(reports, fn name ->
  path = Path.join([__DIR__, name, "report.json"])
  report = path |> File.read!() |> JSON.decode!()
  runs = for run <- report["runs"] do
    Map.take(run, ~w(slp frame pass error replay_dir prefix_audit timing_valid accurate_nmsub))
  end
  {name, %{runs: runs, exact: Enum.count(runs, &get_in(&1, ["prefix_audit", "valid"])),
    total: length(runs)}}
end)
File.write!(Path.join(__DIR__, "human_checks_summary.json"), JSON.encode!(summary) <> "\n")
IO.inspect(Map.new(summary, fn {k, v} -> {k, Map.take(v, [:exact, :total])} end))
