# Run with the same bare-VM dependencies as sweep_actor_throughput.exs.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep

{opts, [], []} = OptionParser.parse(System.argv(), strict: [out: :string, seconds: :integer])
out = opts[:out] || "eval_runs/resync_spread_20261026"
seconds = opts[:seconds] || 10
true = seconds >= 2
File.mkdir_p!(out)
{_, policy, replay, character} = Enum.find(Sweep.policies(), &(elem(&1, 0) == "mamba"))

sources =
  [
    __ENV__.file |> Path.relative_to_cwd(),
    "scripts/lib/benchmark_sweep.exs",
    "scripts/lib/resync_spread.exs",
    "scripts/profile_actor_throughput.exs",
    policy,
    replay
  ] ++ Sweep.inference_sources()

hashes = Sweep.hashes(sources)
path = Path.join(out, "manifest.json")

manifest =
  if File.exists?(path) do
    previous = Sweep.read(path)
    true = previous["hashes"] == hashes and previous["seconds"] == seconds
    previous
  else
    %{"hashes" => hashes, "seconds" => seconds, "runs" => []}
  end

manifest =
  Enum.reduce(["host", "cuda"], manifest, fn client, manifest ->
    if Enum.any?(manifest["runs"], &(&1["client"] == client)) do
      manifest
    else
      Sweep.wait_for_gpu(10)
      output = Path.join(out, "mamba_#{client}.json")

      args = [
        "-pa",
        "_build/dev/lib/*/ebin",
        "scripts/run_recurrent_validation.exs",
        "scripts/profile_actor_throughput.exs",
        "--policy",
        policy,
        "--replay",
        replay,
        "--character",
        character,
        "--rows",
        "8",
        "--seconds",
        to_string(seconds),
        "--resync-spread",
        "--out",
        output
      ]

      env = [{"RECURRENT_CLIENT", client}, {"EDIFICE_FUSED_CUSTOM_CALL", "0"}]
      IO.puts("Starting Mamba resync schedules on #{client}")
      conflicts = Sweep.run(args, env, Path.join(out, "mamba_#{client}.log"))
      true = conflicts == []
      true = Sweep.hashes(sources) == hashes
      report = Sweep.read(output)
      true = length(report["results"]) == 16
      true = length(report["resync_validation"]) == 2

      record = %{
        "client" => client,
        "command" => args,
        "env" => Map.new(env),
        "competing_gpu_pids" => conflicts
      }

      manifest = Map.update!(manifest, "runs", &(&1 ++ [record]))
      Sweep.write(path, manifest)
      IO.puts("Completed #{client}; replay cadence validated and carry differences recorded")
      manifest
    end
  end)

Sweep.write(path, Map.put(manifest, "status", "complete"))
