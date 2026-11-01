# AI-generated. Method and poem: eval_runs/mamba_learning_20261101/README.md.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep
alias ExPhil.Training.Output

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [out: :string, seeds: :string, files: :integer, steps: :integer, seconds: :integer]
  )

out = opts[:out] || "eval_runs/mamba_learning_20261101/pilot"
File.mkdir_p!(out)
seeds = String.split(opts[:seeds] || "905,906", ",")

settings = %{
  files: opts[:files] || 32,
  steps: opts[:steps] || 300,
  seconds: opts[:seconds] || 120
}

sources =
  ["scripts/compare_mamba_learning.exs", __ENV__.file] ++
    Path.wildcard("lib/exphil/training/**/*.ex") ++
    Path.wildcard("_build/dev/lib/{nx,exla,axon,edifice,exphil}/ebin/*.{beam,app}")

hashes = Sweep.hashes(sources)
path = Path.join(out, "manifest.json")

manifest =
  if File.exists?(path),
    do: Sweep.read(path),
    else: %{
      "hashes" => hashes,
      "settings" => Map.new(settings, fn {k, v} -> {to_string(k), v} end),
      "seeds" => seeds,
      "runs" => [],
      "started_at" => DateTime.to_iso8601(DateTime.utc_now()),
      "gpu" =>
        elem(
          System.cmd("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"]),
          0
        )
    }

true = manifest["hashes"] == hashes and manifest["seeds"] == seeds
true = manifest["settings"] == Map.new(settings, fn {k, v} -> {to_string(k), v} end)
Sweep.write(path, manifest)
cases = [{"mamba", 16}, {"mamba2_full", 16}, {"mamba2_full", 128}]

manifest =
  Enum.reduce(Enum.with_index(seeds), manifest, fn {seed, i}, manifest ->
    order = if rem(i, 2) == 0, do: cases, else: Enum.reverse(cases)

    Enum.reduce(order, manifest, fn {backbone, size}, manifest ->
      label = "#{backbone}_n#{size}_s#{seed}"

      if Enum.any?(manifest["runs"], &(&1["label"] == label)) do
        manifest
      else
        Sweep.wait_for_gpu(3)
        dir = Path.join(out, label)
        Output.puts("Starting #{label}; progress in #{dir}.log")

        argv = [
          "-pa",
          "_build/dev/lib/*/ebin",
          "scripts/compare_mamba_learning.exs",
          "--backbone",
          backbone,
          "--state-size",
          to_string(size),
          "--seed",
          seed,
          "--out",
          dir,
          "--files",
          to_string(settings.files),
          "--steps",
          to_string(settings.steps),
          "--seconds",
          to_string(settings.seconds)
        ]

        started = System.monotonic_time(:millisecond)
        conflicts = Sweep.run(argv, [{"EDIFICE_FUSED_CUSTOM_CALL", "1"}], dir <> ".log")
        true = Sweep.hashes(sources) == hashes
        split = Sweep.read(Path.join(dir, "split.json"))
        replay_hashes = Sweep.hashes(split["train"] ++ split["validation"])

        row = %{
          "label" => label,
          "conflicts" => conflicts,
          "argv" => argv,
          "replay_hashes" => replay_hashes,
          "batch_fingerprints" => Sweep.read(Path.join(dir, "batch_fingerprints.json")),
          "process_seconds" => (System.monotonic_time(:millisecond) - started) / 1000,
          "initial" => Sweep.read(Path.join(dir, "initial.json")),
          "equal_steps" => Sweep.read(Path.join(dir, "equal_steps.json")),
          "equal_time" => Sweep.read(Path.join(dir, "equal_time.json"))
        }

        if manifest["runs"] != [],
          do: true = hd(manifest["runs"])["replay_hashes"] == replay_hashes

        peers = Enum.filter(manifest["runs"], &String.ends_with?(&1["label"], "_s#{seed}"))

        Enum.each(peers, fn peer ->
          true = peer["batch_fingerprints"] == row["batch_fingerprints"]
        end)

        manifest = Map.put(manifest, "runs", manifest["runs"] ++ [row])
        Sweep.write(path, manifest)
        Output.success("Completed #{label}")
        manifest
      end
    end)
  end)

Sweep.write(path, Map.put(manifest, "status", "complete"))
