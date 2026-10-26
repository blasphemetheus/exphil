# devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_sampling_kernels.exs [DIRECTORY]
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep
root = List.first(System.argv()) || "eval_runs/sampling_kernels_20261026"

lines =
  for client <- ["host", "cuda"] do
    manifest = Sweep.read(Path.join(root, "#{client}_manifest.json"))
    true = manifest["hashes_verified_after"] and manifest["priming_controllers_exact"]

    for arch <- ["min_gru", "gru", "mamba", "mamba2"], rows <- [1, 8] do
      [old, new] =
        for kernel <- ["rolled", "unrolled"],
            do: Sweep.read(Path.join(root, "#{arch}_#{client}_#{rows}_#{kernel}.json"))

      true = old["priming_controller_sha256"] == new["priming_controller_sha256"]
      true = length(old["results"]) == length(new["results"])

      for {a, b} <- Enum.zip(old["results"], new["results"]) do
        true = {a["kind"], a["rows"]} == {b["kind"], b["rows"]}
        speedup = :erlang.float_to_binary(a["median_us"] / b["median_us"], decimals: 2)

        "| #{client} | #{arch} | #{rows} | #{a["kind"]} | #{a["rows"]} | " <>
          "#{a["median_us"]} / #{a["p95_us"]} | #{b["median_us"]} / #{b["p95_us"]} | #{speedup}x |"
      end
    end
  end

header = [
  "# Paired stochastic sampling timings",
  "",
  "Times are microseconds per full controller call (per batch for eight rows).",
  "100 fresh-key calls per scenario; compilation and explicit restore/reset calls excluded.",
  "Rolled and unrolled variants use identical checkpoints and sampler logic.",
  "",
  "| Client | Policy | Batch | Scenario | Replay rows | Rolled median / p95 | Unrolled median / p95 | Median speedup |",
  "|---|---|---:|---|---:|---:|---:|---:|"
]

File.write!(Path.join(root, "TABLES.md"), Enum.join(header ++ List.flatten(lines), "\n") <> "\n")
