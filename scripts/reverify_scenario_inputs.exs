alias ExPhil.Eval.ScenarioInputTiming

{opts, _, []} =
  OptionParser.parse(System.argv(), strict: [scores: :string, profile: :string, out: :string])

profile =
  case Keyword.fetch!(opts, :profile) do
    "pipe_v1" -> :pipe_v1
    "pipe_v2" -> :pipe_v2
  end

scores = opts |> Keyword.fetch!(:scores) |> File.read!() |> Jason.decode!(keys: :atoms)

rows =
  Enum.map(scores.runs, fn run ->
    timing =
      ScenarioInputTiming.verify_directory(
        run.replay_dir,
        run.policy_inputs,
        scores.response_delay,
        0,
        profile: profile
      )

    %{frame: run.frame, run: run.run, timing: timing}
  end)

report = %{
  source_scores: opts[:scores],
  profile: profile,
  runs: rows,
  valid_runs: Enum.count(rows, & &1.timing.valid),
  total_runs: length(rows)
}

File.write!(Keyword.fetch!(opts, :out), Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(Map.drop(report, [:runs]))
