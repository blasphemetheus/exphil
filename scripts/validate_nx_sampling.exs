# AI-generated. Method and authorship marker: docs/research/NX_CURRENT_VALIDATION.md.
# Run with this project's existing beams; candidate Edifice build paths are added
# only in fresh worker VMs, leaving the running application and its NIFs alone.
if List.first(System.argv()) == "--worker" do
  [_, build | args] = System.argv()
  Code.prepend_paths(Path.wildcard(Path.join(build, "lib/*/ebin")))
  IO.puts("Nx: #{:code.which(Nx)}; EXLA: #{:code.which(EXLA)}")
  System.argv(args)
  Code.require_file("scripts/run_recurrent_validation.exs")
else
  Code.require_file("lib/benchmark_sweep.exs", __DIR__)
  alias ExPhil.BenchmarkSweep, as: Sweep

  {opts, [], []} =
    OptionParser.parse(System.argv(),
      strict: [
        edifice_build: :string,
        out: :string,
        rounds: :integer,
        models: :string,
        rows: :string
      ]
    )

  build = opts |> Keyword.fetch!(:edifice_build) |> Path.expand()
  out = opts |> Keyword.fetch!(:out) |> Path.expand()
  models = String.split(opts[:models] || "mamba,mamba2", ",")
  rows = Enum.map(String.split(opts[:rows] || "1,8", ","), &String.to_integer/1)
  rounds = opts[:rounds] || 2
  true = rounds > 0 and Enum.all?(rows, &(&1 > 0))
  File.mkdir_p!(out)
  policies = Enum.filter(Sweep.policies(), fn {arch, _, _, _} -> arch in models end)
  true = length(policies) == length(models)
  hashes = Sweep.hashes(Sweep.inference_sources() ++ Sweep.inputs() ++ [__ENV__.file])
  beam_hashes = Sweep.hashes(Path.wildcard(Path.join(build, "lib/{nx,exla,edifice}/ebin/*.beam")))

  manifest = %{
    source_hashes: hashes,
    candidate_beam_hashes: beam_hashes,
    candidate_build: build,
    otp: System.otp_release(),
    elixir: System.version(),
    candidate_native_hash: Sweep.hashes([Path.join(build, "lib/exla/priv/libexla.so")]),
    desktop: Sweep.gpu_processes(),
    runs: []
  }

  manifest =
    Enum.reduce(1..rounds, manifest, fn round, manifest ->
      Enum.reduce(policies, manifest, fn {arch, policy, replay, character}, manifest ->
        Enum.reduce(rows, manifest, fn rows, manifest ->
          order = if rem(round, 2) == 1, do: ["nx", "helper"], else: ["helper", "nx"]

          reports =
            Enum.map(order, fn implementation ->
              label = "#{arch}_b#{rows}_r#{round}_#{implementation}"
              file = Path.join(out, label <> ".json")
              if File.exists?(file), do: raise("Use a fresh output directory: #{file}")
              Sweep.wait_for_gpu(2)

              args = [
                "-pa",
                "_build/dev/lib/*/ebin",
                __ENV__.file,
                "--worker",
                build,
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

              conflicts =
                Sweep.run(
                  args,
                  [
                    {"RECURRENT_CLIENT", "cuda"},
                    {"EDIFICE_FUSED_CUSTOM_CALL", "0"},
                    {"SAMPLING_REFERENCE", if(implementation == "nx", do: "1", else: "0")}
                  ],
                  Path.join(out, label <> ".log")
                )

              if conflicts != [], do: raise("GPU contention: #{inspect(conflicts)}")
              report = Sweep.read(file)

              %{
                file: file,
                implementation: implementation,
                round: round,
                arch: arch,
                rows: rows,
                controller_sha256: report["priming_controller_sha256"],
                results:
                  Enum.map(
                    report["results"],
                    &Map.take(&1, ["kind", "median_us", "p95_us", "samples"])
                  )
              }
            end)

          [left, right] = reports
          true = left.controller_sha256 == right.controller_sha256
          updated = Map.update!(manifest, :runs, &(&1 ++ reports))
          Sweep.write(Path.join(out, "manifest.json"), updated)
          updated
        end)
      end)
    end)

  Sweep.write(Path.join(out, "manifest.json"), manifest)
end
