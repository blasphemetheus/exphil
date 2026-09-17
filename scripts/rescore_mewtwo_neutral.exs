# Re-score preserved games after a scorer correction; never reruns gameplay.
alias ExPhil.Data.Peppi
alias ExPhil.Eval.MewtwoNeutralBenchmark, as: Benchmark
[out] = System.argv()
protocol = File.read!(Path.join(out, "protocol.json")) |> Jason.decode!()

runs =
  for mode <- ["stand", "cpu"], n <- 1..3 do
    dir = Path.join(out, "#{mode}_#{n}")
    session = File.read!(Path.join(dir, "session.json")) |> Jason.decode!()
    original = File.read!(Path.join(dir, "result.json")) |> Jason.decode!()
    [path] = Path.wildcard(Path.join(dir, "**/*.slp"))
    {:ok, replay} = Peppi.parse(path)
    metrics = Benchmark.score(replay, session["last_frame"])

    valid =
      session["status"] == "ok" and session["errors"] == 0 and
        session["measured_latency"] == 1 and metrics.frames == 1800 and
        (original["exit_status"] || 0) == 0

    result = %{
      mode: mode,
      run: n,
      valid: valid,
      metrics: metrics,
      session: session,
      replay: path,
      replay_sha256: :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
    }

    File.write!(Path.join(dir, "result_rescored.json"), Jason.encode!(result, pretty: true))

    IO.inspect(%{
      mode: mode,
      run: n,
      valid: valid,
      openings: metrics.neutral.outcomes,
      waves: metrics.wavedashes,
      shield: metrics.max_shield_action_streak
    })

    result
  end

criteria = Benchmark.qualification(runs)

File.write!(
  Path.join(out, "summary_rescored.json"),
  Jason.encode!(
    %{
      protocol: protocol,
      runs: runs,
      criteria: criteria,
      automated_passed: Enum.all?(criteria, fn {_, v} -> v end),
      graphical_validated: false,
      reason:
        "Correct first-frame initialization and exclude ongoing grabs/throws from fresh neutral exchanges; original reports preserved"
    },
    pretty: true
  )
)
