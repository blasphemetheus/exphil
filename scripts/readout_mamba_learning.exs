# AI-generated. Small fixed-opponent readout; see the pilot README.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep
alias ExPhil.Training.Output

{opts, [directory], []} =
  OptionParser.parse(System.argv(),
    strict: [
      frames: :integer,
      envs: :integer,
      seed: :integer,
      budgets: :string,
      direct: :boolean
    ]
  )

frames = opts[:frames] || 1800
envs = opts[:envs] || 4
seed = opts[:seed] || 71
budgets = String.split(opts[:budgets] || "equal_steps,equal_time", ",")
true = frames > 0 and envs > 0 and Enum.all?(budgets, &(&1 in ["equal_steps", "equal_time"]))
manifest = Sweep.read(Path.join(directory, "manifest.json"))
true = manifest["status"] == "complete"
opponent = "checkpoints/coh_fox_mamba_master_ep3/model_best_policy.bin"

source_hashes =
  Sweep.hashes(
    [opponent, "scripts/league.exs", "scripts/run_recurrent_validation.exs", __ENV__.file] ++
      Sweep.inference_sources()
  )

candidate = fn row, budget ->
  %{
    name: row["label"] <> "_" <> budget,
    policy: Path.join([directory, row["label"], budget <> "_policy.bin"]),
    character: "fox",
    line: "controlled training comparison"
  }
end

cases =
  for row <- manifest["runs"], budget <- budgets do
    bot = candidate.(row, budget)

    {bot.name,
     [bot, %{name: "master_ep3", policy: opponent, character: "fox", line: "frozen opponent"}]}
  end

direct =
  if opts[:direct] do
    for s <- manifest["seeds"], budget <- budgets do
      a =
        Enum.find(manifest["runs"], &(&1["label"] == "mamba_n16_s#{s}")) ||
          raise("Missing Mamba-1 arm")

      b =
        Enum.find(manifest["runs"], &(&1["label"] == "mamba2_full_n16_s#{s}")) ||
          raise("Missing Mamba-2 arm")

      {"direct_s#{s}_#{budget}", [candidate.(a, budget), candidate.(b, budget)]}
    end
  else
    []
  end

for {label, bots} <- cases ++ direct do
  out = Path.join([directory, "play", label])
  File.mkdir_p!(out)
  roster = Path.join(out, "roster.json")
  expected = Map.merge(source_hashes, Sweep.hashes(Enum.map(bots, & &1.policy)))
  settings = %{"frames" => frames, "envs" => envs, "seed" => seed}
  proof = Path.join(out, "provenance.json")

  if File.exists?(proof) do
    true = Sweep.read(proof)["hashes"] == expected
    true = Sweep.read(proof)["settings"] == settings
  end

  Sweep.write(proof, %{
    hashes: expected,
    settings: settings,
    scope: "batched parity simulator; zero delay; FD; both ports"
  })

  Sweep.write(roster, bots)

  unless File.exists?(Path.join(out, "league.json")) do
    Sweep.wait_for_gpu(3)
    Output.puts("Playing #{label}; log #{out}.log")

    conflicts =
      Sweep.run(
        [
          "-pa",
          "_build/dev/lib/*/ebin",
          "scripts/run_recurrent_validation.exs",
          "scripts/league.exs",
          "--roster",
          roster,
          "--envs",
          to_string(envs),
          "--frames",
          to_string(frames),
          "--seed",
          to_string(seed),
          "--stage",
          "final_destination",
          "--out",
          out
        ],
        [{"EDIFICE_FUSED_CUSTOM_CALL", "0"}],
        out <> ".log"
      )

    report = Sweep.read(Path.join(out, "league.json"))
    true = report["unsupported"] == []
    true = source_hashes == Sweep.hashes(Map.keys(source_hashes))
    Sweep.write(Path.join(out, "contention.json"), conflicts)
    Output.success("Played #{label}")
  end
end
