# AI-generated; method and attribution: docs/guides/MAMBA_BATCH_PARITY.md.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep
{opts, [], []} = OptionParser.parse(System.argv(), strict: [out: :string, frames: :integer])
out = opts[:out] || "eval_runs/mamba_batch_parity_20261026/dolphin"
frames = opts[:frames] || 600
File.mkdir_p!(out)

sources =
  Sweep.inference_sources() ++
    [
      "scripts/measure_dolphin_resync.exs",
      "scripts/sweep_dolphin_resync.exs",
      "lib/exphil/embeddings/player.ex",
      "lib/exphil/embeddings/game/spatial.ex",
      "checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin",
      Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
      Path.expand("~/isos/melee.iso")
    ]

hashes = Sweep.hashes(sources)

runs =
  for client <- ["host", "cuda"],
      round <- [1, 2],
      phase <- if(round == 1, do: [0, 15], else: [15, 0]) do
    Sweep.wait_for_gpu(3)
    name = "#{client}_r#{round}_phase#{phase}"
    path = Path.join(out, name <> ".json")
    IO.puts("Starting Dolphin #{name}")

    args = [
      "-pa",
      "_build/dev/lib/*/ebin",
      "scripts/run_recurrent_validation.exs",
      "scripts/measure_dolphin_resync.exs",
      "--phase",
      to_string(phase),
      "--frames",
      to_string(frames),
      "--out",
      path
    ]

    conflicts =
      Sweep.run(
        args,
        [{"RECURRENT_CLIENT", client}, {"ERL_FLAGS", "+S 4:4"}],
        Path.join(out, name <> ".log")
      )

    unless conflicts == [], do: raise("GPU contention during #{name}")
    report = Sweep.read(path)
    unless report["frames"] == frames, do: raise("incomplete #{name}")
    summary = Map.drop(report, ["samples"]) |> Map.merge(%{"round" => round, "file" => path})

    IO.puts(
      "Completed #{name}: p95 #{summary["pair_p95_us"]} us, budget exceeds #{summary["pair_over_budget"]}"
    )

    summary
  end

unless hashes == Sweep.hashes(sources), do: raise("sources changed during sweep")
Sweep.write(Path.join(out, "summary.json"), %{runs: runs, hashes: hashes, status: "complete"})
