# Run from devenv with plain Elixir; child Mix jobs are strictly sequential.
alias ExPhil.Data.Peppi
alias ExPhil.Eval.MewtwoNeutralBenchmark
[policy, out] = System.argv()
File.mkdir_p!(Path.dirname(out))
File.mkdir!(out)
sha = fn path -> :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower) end

protocol = %{
  policy: Path.expand(policy),
  sha256: sha.(policy),
  temperature: 1.0,
  deterministic: false,
  stage: :final_destination,
  opponent: :fox,
  runs_per_mode: 3,
  seconds: 30,
  cpu_level: 6,
  criteria: "docs/planning/MEWTWO_NEUTRAL_BASELINE_V1.md"
}

File.write!(Path.join(out, "protocol.json"), Jason.encode!(protocol, pretty: true))

runs =
  for mode <- ["stand", "cpu"], n <- 1..3 do
    dir = Path.join(out, "#{mode}_#{n}")
    File.mkdir!(dir)
    report = Path.join(dir, "session.json")

    args = [
      "run",
      "scripts/play_dolphin.exs",
      "--policy",
      policy,
      "--dolphin",
      Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
      "--iso",
      Path.expand("~/isos/melee.iso"),
      "--replay-dir",
      dir,
      "--session-report",
      report,
      "--character",
      "mewtwo",
      "--stage",
      "final_destination",
      "--dummy",
      mode,
      "--dummy-character",
      "fox",
      "--dummy-cpu-level",
      if(mode == "cpu", do: "6", else: "0"),
      "--seconds",
      "30",
      "--reaction-delay",
      "0",
      "--live-af",
      "--temperature",
      "1.0",
      "--headless",
      "--no-audio",
      "--emulation-speed",
      "0",
      "--blocking-input"
    ]

    File.write!(Path.join(dir, "args.json"), Jason.encode!(args))
    IO.puts("Starting #{mode}_#{n}")

    {_, status} =
      System.cmd("timeout", ["270", "mix" | args],
        into: File.stream!(Path.join(dir, "play.log"), [:write]),
        stderr_to_stdout: true,
        env: [{"EXLA_TARGET", "cuda"}, {"EXPHIL_GPU_MEMORY_FRACTION", "0.15"}]
      )

    result =
      try do
        session = File.read!(report) |> Jason.decode!()
        [path] = Path.wildcard(Path.join(dir, "**/*.slp"))
        {:ok, replay} = Peppi.parse(path)
        metrics = MewtwoNeutralBenchmark.score(replay, session["last_frame"])

        valid =
          status == 0 and session["status"] == "ok" and session["errors"] == 0 and
            session["measured_latency"] == 1 and metrics.frames == 1800

        %{
          valid: valid,
          metrics: metrics,
          replay: path,
          replay_sha256: sha.(path),
          session: session
        }
      rescue
        e -> %{valid: false, error: Exception.message(e), exit_status: status}
      end
      |> Map.merge(%{mode: mode, run: n})

    File.write!(Path.join(dir, "result.json"), Jason.encode!(result, pretty: true))
    IO.inspect(Map.drop(result, [:session]))
    result
  end

valid = Enum.all?(runs, & &1.valid)
criteria = MewtwoNeutralBenchmark.qualification(runs)
passed = valid and Enum.all?(criteria, fn {_, v} -> v end)

File.write!(
  Path.join(out, "summary.json"),
  Jason.encode!(
    %{
      protocol: protocol,
      runs: runs,
      criteria: criteria,
      automated_passed: passed,
      graphical_validated: false
    },
    pretty: true
  )
)

unless passed, do: System.halt(1)
