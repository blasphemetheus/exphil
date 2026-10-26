# devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_sampling_kernels.exs --client cuda --out /tmp/sampling-pairs
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep

{opts, [], []} =
  OptionParser.parse(System.argv(), strict: [client: :string, out: :string, dry_run: :boolean])

client = Keyword.fetch!(opts, :client)
true = client in ["host", "cuda"]
out = opts[:out] || "eval_runs/sampling_kernels_20261026"

plans =
  for {arch, policy, replay, character} <- Sweep.policies(),
      rows <- [1, 8],
      kernel <- ["rolled", "unrolled"] do
    label = "#{arch}_#{client}_#{rows}_#{kernel}"
    file = Path.join(out, label <> ".json")

    args = [
      "-pa",
      "_build/dev/lib/*/ebin",
      "scripts/run_recurrent_validation.exs",
      "scripts/profile_recurrent_policy.exs",
      "--policy",
      policy,
      "--replay",
      replay,
      "--character",
      character,
      "--batch",
      to_string(rows),
      "--iterations",
      "100",
      "--stochastic",
      "--out",
      file
    ]

    %{label: label, file: file, args: args, rows: rows, kernel: kernel}
  end

if opts[:dry_run] do
  IO.puts(Jason.encode!(plans, pretty: true))
else
  File.mkdir_p!(out)

  sources =
    [
      "scripts/sweep_sampling_kernels.exs",
      "scripts/lib/benchmark_sweep.exs",
      "scripts/profile_recurrent_policy.exs"
    ] ++ Sweep.inference_sources() ++ Sweep.inputs()

  hashes = Sweep.hashes(sources)
  path = Path.join(out, "#{client}_manifest.json")

  manifest =
    if File.exists?(path) do
      existing = Sweep.read(path)

      unless existing["hashes"] == hashes,
        do: raise("Source/input changed; use a new output directory")

      existing
    else
      %{
        "client" => client,
        "hashes" => hashes,
        "started_at" => System.system_time(:second),
        "runs" => []
      }
    end

  manifest =
    Enum.reduce(plans, manifest, fn plan, manifest ->
      if not File.exists?(plan.file) do
        if client == "cuda", do: Sweep.wait_for_gpu()
        IO.puts("Starting #{plan.label}")
        Sweep.write(path, manifest)

        env = [
          {"RECURRENT_CLIENT", client},
          {"EDIFICE_FUSED_CUSTOM_CALL", "0"},
          {"SAMPLING_REFERENCE", if(plan.kernel == "rolled", do: "1", else: "0")}
        ]

        conflicts = Sweep.run(plan.args, env, Path.join(out, plan.label <> ".log"))

        if client == "cuda" and conflicts != [] do
          archive = Path.join([out, "contended", "#{plan.label}_#{System.system_time(:second)}"])
          File.mkdir_p!(archive)
          File.rename!(plan.file, Path.join(archive, Path.basename(plan.file)))

          File.rename!(
            Path.join(out, plan.label <> ".log"),
            Path.join(archive, plan.label <> ".log")
          )

          Sweep.write(Path.join(archive, "conflicts.json"), conflicts)
          raise("Competing GPU process detected; archived run. Resume when free.")
        end
      end

      report = Sweep.read(plan.file)
      true = report["sampling_kernel"] == plan.kernel and report["client"] == client
      true = report["rows"] == plan.rows and report["stochastic"]
      true = Enum.all?(report["results"], &(&1["samples"] == 100))

      manifest =
        if Enum.any?(manifest["runs"], &(&1["file"] == plan.file)) do
          manifest
        else
          record = %{
            "file" => plan.file,
            "command" => ["elixir" | plan.args],
            "kernel" => plan.kernel
          }

          Map.update!(manifest, "runs", &(&1 ++ [record]))
        end

      Sweep.write(path, manifest)
      manifest
    end)

  for [old, new] <- Enum.chunk_every(plans, 2) do
    true =
      Sweep.read(old.file)["priming_controller_sha256"] ==
        Sweep.read(new.file)["priming_controller_sha256"]
  end

  true = Sweep.hashes(sources) == hashes

  Sweep.write(
    path,
    Map.merge(manifest, %{
      "finished_at" => System.system_time(:second),
      "hashes_verified_after" => true,
      "priming_controllers_exact" => true
    })
  )
end
