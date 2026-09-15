root = __DIR__
read = fn path -> root |> Path.join(path) |> File.read!() |> JSON.decode!() end
teacher = read.("teacher.json")
cpu = Enum.reject(teacher["runs"], &String.contains?(&1["slp"], "local_multishine"))
held = read.("manifest_heldout.json")["entries"] |> Map.new(&{{&1["slp"], &1["frame"]}, &1})
live = Map.new(~w(orig_cold orig_warm orig_interruptions cov_train cov_heldout cov_heldout_replay), fn name ->
  report = read.("round21/#{name}.json")
  runs = report["runs"]
  failures = for run <- runs, run["details"]["max_chain"] < 10 do
    Map.take(run, ~w(slp frame run))
    |> Map.merge(Map.take(run["details"], ~w(max_chain reentry_frame)))
  end
  {name, %{
    runs: length(runs), chain_ge10: Enum.count(runs, &(&1["details"]["max_chain"] >= 10)),
    scenario_pass: Enum.count(runs, & &1["pass"]),
    audited_prefixes: Enum.count(runs, &is_map(&1["prefix_audit"])),
    exact_prefixes: Enum.count(runs, &get_in(&1, ["prefix_audit", "valid"])),
    errors: report["errored_runs"], rejected_prefixes: report["diverged_runs"],
    invalid_timing: report["invalid_timing_runs"], chain_failures: failures
  }}
end)
pressure = read.("round21/cov_heldout_replay.json")["runs"] |> Map.new(&{&1["replay_dir"], &1})
hit_recovery = read.("round21/cov_heldout_replay_recovery.json")["results"] |> Enum.filter(fn r ->
  run = Map.fetch!(pressure, r["directory"])
  Map.fetch!(held, {run["slp"], run["frame"]})["class"] == "hit"
end)
fit = read.("round21/fit_heldout.json")["cases"] |> Enum.reject(&(&1["case"] == "canonical"))
accuracy = Enum.map(fit, & &1["first18"]["tf_argmax_correct"])
summary = %{
  teacher: %{runs: length(teacher["runs"]), exact: Enum.count(teacher["runs"], &get_in(&1, ["prefix_audit", "valid"])),
    cpu_runs: length(cpu), cpu_exact: Enum.count(cpu, &get_in(&1, ["prefix_audit", "valid"])),
    qualified: length(read.("teacher_qualified.json")["runs"])},
  split: %{train: length(read.("manifest_train.json")["entries"]), heldout: map_size(held)},
  fit: %{train_ready: read.("round21/fit_gate.json")["ready"],
    heldout_first18_mean: Enum.sum(accuracy) / length(accuracy), heldout_first18_min: Enum.min(accuracy)},
  live: live,
  heldout_hit_recovery_under_pressure: %{
    runs: length(hit_recovery), valid: Enum.count(hit_recovery, & &1["valid"]),
    resumed_within60_ready_frames: Enum.count(hit_recovery, & &1["resumed_within60_ready_frames"]),
    outcomes: Enum.frequencies_by(hit_recovery, &get_in(&1, ["initial_hit_episode", "outcome"]))
  }
}
File.write!(Path.join(root, "summary.json"), JSON.encode!(summary) <> "\n")
IO.inspect(summary, limit: :infinity, charlists: :as_lists)
