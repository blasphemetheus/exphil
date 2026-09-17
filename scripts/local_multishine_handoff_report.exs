# Replay audit of scenario responses; retain the original scenario gate result.
# elixir -pa '_build/dev/lib/*/ebin' scripts/local_multishine_handoff_report.exs REPORT.json
[path] = System.argv()
report = path |> File.read!() |> JSON.decode!()

runs = Enum.map(report["runs"], fn run ->
  [replay_path] = Path.wildcard(Path.join(run["replay_dir"], "**/*.slp"))
  {:ok, replay} = ExPhil.Data.Peppi.parse(replay_path)
  # One handoff context state, then the recorded response window. Exclude quit inputs.
  rows = ExPhil.Eval.MultishineBenchmark.rows(replay, 1)
    |> Enum.filter(&(&1.frame >= run["frame"] - 1 and &1.frame < run["frame"] + run["window"]))
  metrics = ExPhil.Eval.MultishineBenchmark.score(rows)
  %{frame: run["frame"], source: run["slp"], replay: replay_path,
    replay_sha256: :crypto.hash(:sha256, File.read!(replay_path)) |> Base.encode16(case: :lower),
    original_gate_pass: run["pass"], original_details: run["details"],
    prefix_valid: get_in(run, ["prefix_audit", "valid"]), timing_valid: run["timing_valid"],
    truncated: run["truncated"], metrics: metrics}
end)

IO.puts(Jason.encode!(%{source_report: path, metrics_semantics: "slippi_state_flag_hitstun_v2",
  readiness: "grounded locomotion or reflector, not an exact actionable-frame oracle", runs: runs}, pretty: true))
