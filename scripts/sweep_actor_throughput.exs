# devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_actor_throughput.exs
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [out: :string, seconds: :integer, import_existing: :boolean]
  )

out = opts[:out] || "eval_runs/actor_throughput_20261026"
seconds = opts[:seconds] || 3
true = seconds >= 1
File.mkdir_p!(out)

sources =
  [
    __ENV__.file |> Path.relative_to_cwd(),
    "scripts/lib/benchmark_sweep.exs",
    "scripts/lib/resync_spread.exs",
    "scripts/profile_actor_throughput.exs"
  ] ++ Sweep.inference_sources() ++ Sweep.inputs()

hashes = Sweep.hashes(sources)
path = Path.join(out, "manifest.json")

{hardware, 0} =
  System.cmd("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"])

manifest =
  if File.exists?(path) do
    previous = Sweep.read(path)
    true = previous["seconds"] == seconds

    if previous["hashes"] != hashes and opts[:import_existing] do
      # Retain valid measurements when replacing only the orchestration language.
      # Every measured Elixir source and checkpoint/replay must remain identical.
      shared = Map.delete(previous["hashes"], "scripts/profile_actor_throughput.py")
      true = shared == Map.take(hashes, Map.keys(shared))
      true = Enum.all?(previous["runs"], &(&1["competing_gpu_pids"] == []))
      File.cp!(path, Path.join(out, "pre_elixir_manifest.json"))

      previous
      |> Map.put("hashes", hashes)
      |> Map.put("orchestration_migration", %{
        "prior_manifest" => "pre_elixir_manifest.json",
        "preserved_labels" => Enum.map(previous["runs"], & &1["label"]),
        "measured_sources_and_inputs_unchanged" => true
      })
    else
      true = previous["hashes"] == hashes
      previous
    end
  else
    %{
      "hashes" => hashes,
      "seconds" => seconds,
      "started_at" => System.system_time(:second),
      "hardware" => String.trim(hardware),
      "runs" => [],
      "discarded_runs" => []
    }
  end

Sweep.write(path, manifest)

manifest =
  Enum.reduce(
    for(client <- ["host", "cuda"], policy <- Sweep.policies(), do: {client, policy}),
    manifest,
    fn {client, {arch, policy, replay, character}}, manifest ->
      label = "#{arch}_#{client}"
      output = Path.join(out, label <> ".json")

      if Enum.any?(manifest["runs"], &(&1["label"] == label)) do
        true = File.exists?(output)
        manifest
      else
        Sweep.wait_for_gpu(60)

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
          "--out",
          output
        ]

        env = [
          {"RECURRENT_CLIENT", client},
          {"EDIFICE_FUSED_CUSTOM_CALL", "0"},
          {"SAMPLING_REFERENCE", "0"}
        ]

        IO.puts("Starting #{label}")
        conflicts = Sweep.run(args, env, Path.join(out, label <> ".log"))
        report = Sweep.read(output)
        true = report["client"] == client and report["seconds"] == seconds
        true = length(report["results"]) == 12

        for result <- report["results"] do
          workers = if result["mode"] == "actors", do: 8, else: 1
          true = length(result["worker_calls"]) == workers and result["calls"] > 0

          if result["rate"] == "paced",
            do: true = result["worker_calls"] == List.duplicate(60 * seconds, workers)

          if result["contention"], do: true = result["competitor"]["iterations"] > 0
        end

        record = %{
          "label" => label,
          "command" => ["elixir" | args],
          "env" => Map.new(env),
          "competing_gpu_pids" => conflicts
        }

        if conflicts != [] do
          archive = Path.join([out, "contended", "#{label}_#{System.system_time(:second)}"])
          File.mkdir_p!(archive)

          for suffix <- [".json", ".json.samples.json.gz", ".log"],
              do:
                File.rename!(Path.join(out, label <> suffix), Path.join(archive, label <> suffix))

          failed =
            Map.update!(
              manifest,
              "discarded_runs",
              &(&1 ++ [Map.put(record, "archive_dir", archive)])
            )

          Sweep.write(path, failed)
          raise("Competing GPU process detected; archived run. Resume this command when free.")
        end

        true = Sweep.hashes(sources) == hashes
        manifest = Map.update!(manifest, "runs", &(&1 ++ [record]))
        Sweep.write(path, manifest)
        IO.puts("Completed #{label}")
        manifest
      end
    end
  )

true = Sweep.hashes(sources) == hashes

Sweep.write(
  path,
  Map.merge(manifest, %{
    "status" => "complete",
    "finished_at" => System.system_time(:second),
    "hashes_verified_after" => true
  })
)
