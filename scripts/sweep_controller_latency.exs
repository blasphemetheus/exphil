# AI-generated. Method, limits and authorship poem are in the generated report.
# Run from ExPhil's devenv shell with its existing compiled dependency tree.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
alias ExPhil.BenchmarkSweep, as: Sweep

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      replay: :string,
      character: :string,
      out: :string,
      seconds: :integer,
      rows: :integer,
      clients: :string
    ]
  )

policy = Keyword.fetch!(opts, :policy)
replay = Keyword.fetch!(opts, :replay)
character = Keyword.fetch!(opts, :character)
out = Keyword.fetch!(opts, :out)
seconds = opts[:seconds] || 10
rows = opts[:rows] || 8
clients = String.split(opts[:clients] || "cuda,host", ",")
true = seconds >= 2 and rows >= 2 and Enum.all?(clients, &(&1 in ["host", "cuda"]))
File.mkdir_p!(out)

sources =
  [
    __ENV__.file,
    "scripts/lib/benchmark_sweep.exs",
    "scripts/lib/resync_spread.exs",
    "scripts/profile_actor_throughput.exs",
    policy,
    replay
  ] ++
    Sweep.inference_sources() ++
    Path.wildcard("_build/dev/lib/{nx,exla,axon,edifice,exphil}/ebin/*.{beam,app}")

hashes = Sweep.hashes(sources)
manifest_path = Path.join(out, "manifest.json")

manifest =
  if File.exists?(manifest_path),
    do: Sweep.read(manifest_path),
    else: %{
      "hashes" => hashes,
      "seconds" => seconds,
      "rows" => rows,
      "clients" => clients,
      "policy" => policy,
      "replay" => replay,
      "character" => character,
      "runs" => [],
      "started_at" => DateTime.to_iso8601(DateTime.utc_now()),
      "desktop_gpu_processes" => Enum.filter(Sweep.gpu_processes(), &Sweep.desktop_process?/1),
      "hardware" =>
        elem(
          System.cmd("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"]),
          0
        )
    }

true =
  manifest["hashes"] == hashes and manifest["seconds"] == seconds and manifest["rows"] == rows

true = manifest["clients"] == clients and manifest["character"] == character
Sweep.write(manifest_path, manifest)

manifest =
  Enum.reduce(
    for(client <- clients, spread <- [false, true], do: {client, spread}),
    manifest,
    fn {client, spread}, manifest ->
      label = "#{client}_#{if spread, do: "resync", else: "baseline"}"

      if Enum.any?(manifest["runs"], &(&1["label"] == label)) do
        manifest
      else
        Sweep.wait_for_gpu(5)
        output = Path.join(out, label <> ".json")

        args =
          [
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
            to_string(rows),
            "--seconds",
            to_string(seconds),
            "--paced-only",
            "--out",
            output
          ] ++ if(spread, do: ["--resync-spread"], else: [])

        env = [
          {"RECURRENT_CLIENT", client},
          {"EDIFICE_FUSED_CUSTOM_CALL", "0"},
          {"SAMPLING_REFERENCE", "0"}
        ]

        IO.puts("Starting #{label}; progress in #{Path.join(out, label <> ".log")}")
        conflicts = Sweep.run(args, env, Path.join(out, label <> ".log"))

        record = %{
          "label" => label,
          "command" => args,
          "env" => Map.new(env),
          "competing_gpu_pids" => conflicts
        }

        if conflicts != [] do
          Sweep.write(Path.join(out, label <> ".contended.json"), record)
          raise("Competing compute process detected; this run cannot support idle timing claims")
        end

        true = Sweep.hashes(sources) == hashes
        result = Sweep.read(output)
        true = length(result["results"]) == if(spread, do: 16, else: 6)
        manifest = Map.update!(manifest, "runs", &(&1 ++ [record]))
        Sweep.write(manifest_path, manifest)
        IO.puts("Completed #{label}")
        manifest
      end
    end
  )

lines =
  for run <- manifest["runs"] do
    report = Sweep.read(Path.join(out, run["label"] <> ".json"))

    raw =
      Path.join(out, report["samples_file"]) |> File.read!() |> :zlib.gunzip() |> Jason.decode!()

    true = length(raw) == length(report["results"])

    for {raw_result, result} <- Enum.zip(raw, report["results"]) do
      true = Map.delete(raw_result, "streams") == result
      samples = Enum.concat(raw_result["streams"])
      true = Enum.all?(raw_result["streams"], &(length(&1) == 60 * seconds))
      true = length(samples) == result["calls"]
      response = Enum.map(samples, &Enum.at(&1, 2)) |> Enum.sort()
      service = Enum.map(samples, &Enum.at(&1, 1)) |> Enum.sort()
      n = length(response)
      true = Enum.count(response, &(&1 > 1_000_000 / 60)) == result["deadline_misses"]

      for {name, position} <- [
            {"median_us", div(n, 2)},
            {"p95_us", ceil(n * 0.95) - 1},
            {"p99_us", ceil(n * 0.99) - 1},
            {"max_us", n - 1}
          ] do
        true = result["response"][name] == Enum.at(response, position)
        true = result["service"][name] == Enum.at(service, position)
      end

      ms = fn value -> :erlang.float_to_binary(value / 1000, decimals: 2) end

      "| #{report["client"]} | #{result["mode"]} | #{result["schedule"] || "aligned"} | #{result["round"] || 0} | " <>
        "#{if result["contention"], do: "matmul", else: "desktop only"} | #{ms.(result["service"]["median_us"])} | " <>
        "#{ms.(result["service"]["p99_us"])} | #{ms.(result["response"]["p99_us"])} | " <>
        "#{ms.(result["response"]["max_us"])} | #{result["deadline_misses"]}/#{n} |"
    end
  end

header = [
  "# Complete controller latency — AI-generated",
  "",
  "Checkpoint: `#{policy}`. Replay: `#{replay}`. #{seconds}-second phases, #{rows} bots, 60 Hz.",
  "Agent input through returned controller, including encoding, recurrent carry/replay and stochastic sampling.",
  "Dolphin transport, emulator work, rendering and configured game-frame delay are excluded.",
  "Warm execution only: compilation/history fill are recorded as setup, not included in percentiles.",
  "Service measures request-to-return time; response also includes lateness relative to the scheduled frame.",
  "One late batch affects all rows. The competing matmul is synthetic GPU load, not a training simulation.",
  "Desktop GPU processes remain. Other compute-process observations invalidate a run.",
  "Resync uses the benchmark's validated phase assignment, with schedule order reversed in round two.",
  "Carry differences are recorded, not treated as policy-quality or batch/single equivalence proof.",
  "",
  "| Client | Mode | Schedule | Round | Load | Service median ms | Service p99 ms | Response p99 ms | Response max ms | Missed/offered calls |",
  "|---|---|---|---:|---|---:|---:|---:|---:|---:|"
]

File.write!(
  Path.join(out, "TABLES.md"),
  Enum.join(
    header ++
      List.flatten(lines) ++
      [
        "",
        "**AI-generated poem**",
        "",
        "A frame arrives, a choice returns;  ",
        "We count the time while memory turns."
      ],
    "\n"
  ) <> "\n"
)

Sweep.write(manifest_path, Map.put(manifest, "status", "complete"))
IO.puts("Verified raw samples, percentiles and deadline counts; wrote #{out}/TABLES.md")
