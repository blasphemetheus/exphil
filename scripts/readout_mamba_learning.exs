# AI-generated. Small fixed-opponent readout; see the pilot README.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep
alias ExPhil.Training.Output
[directory] = System.argv()
manifest = Sweep.read(Path.join(directory, "manifest.json"))
true = manifest["status"] == "complete"
opponent = "checkpoints/coh_fox_mamba_master_ep3/model_best_policy.bin"

source_hashes =
  Sweep.hashes(
    [opponent, "scripts/league.exs", "scripts/run_recurrent_validation.exs", __ENV__.file] ++
      Sweep.inference_sources()
  )

for row <- manifest["runs"], budget <- ["equal_steps", "equal_time"] do
  label = row["label"] <> "_" <> budget
  policy = Path.join([directory, row["label"], budget <> "_policy.bin"])
  out = Path.join([directory, "play", label])
  File.mkdir_p!(out)
  roster = Path.join(out, "roster.json")
  expected = Map.merge(source_hashes, Sweep.hashes([policy]))
  proof = Path.join(out, "provenance.json")
  if File.exists?(proof), do: true = Sweep.read(proof)["hashes"] == expected

  Sweep.write(proof, %{
    hashes: expected,
    scope: "batched parity simulator; zero delay; FD; fixed opponent; both ports"
  })

  Sweep.write(roster, [
    %{name: label, policy: policy, character: "fox", line: "learning pilot"},
    %{name: "master_ep3", policy: opponent, character: "fox", line: "frozen opponent"}
  ])

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
          "4",
          "--frames",
          "1800",
          "--seed",
          "71",
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
