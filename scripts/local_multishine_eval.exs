# Run via devenv shell -- elixir -pa '_build/dev/lib/*/ebin' this_file --policy ... --out NEW_DIR
# Each match owns its replay directory. No global Dolphin cleanup or shared replay scan.
alias ExPhil.Data.Peppi
alias ExPhil.Eval.MultishineBenchmark, as: Benchmark
{opts, [], []} = OptionParser.parse(System.argv(), strict: [
  policy: :string, out: :string, stages: :string, opponents: :string,
  runs: :integer, seconds: :integer, mode: :string, level: :integer, dolphin: :string
])
out = Keyword.fetch!(opts, :out)
File.mkdir_p!(Path.dirname(out))
File.mkdir!(out)
policy = Keyword.fetch!(opts, :policy)
stages = String.split(opts[:stages] || "final_destination", ",")
opponents = String.split(opts[:opponents] || "fox", ",")
mode = opts[:mode] || "stand"
unless mode in ["stand", "cpu"], do: raise("mode must be stand or cpu")
seconds = opts[:seconds] || 30
count = opts[:runs] || 1
unless seconds > 0 and count > 0, do: raise("seconds and runs must be positive")
unless Enum.all?(stages, &(&1 in ~w(final_destination battlefield dreamland yoshis_story fountain_of_dreams pokemon_stadium))),
  do: raise("Unsupported demo stage")
unless Enum.all?(opponents, &(&1 in ~w(fox falco marth peach samus))),
  do: raise("Unsupported demo opponent")
unless (opts[:level] || 1) in 1..9, do: raise("CPU level must be 1..9")
sha = fn p -> :crypto.hash(:sha256, File.read!(p)) |> Base.encode16(case: :lower) end
protocol = %{policy: policy, policy_sha256: sha.(policy), stages: stages, opponents: opponents,
  runs_per_cell: count, seconds: seconds, mode: mode, cpu_level: opts[:level] || 1,
  reaction_delay: 0, expected_latency: 1, deterministic: true, frozen_stadium: true,
  metrics_semantics: "slippi_state_flag_hitstun_v2",
  source_hashes: Map.new(~w(scripts/play_dolphin.exs lib/exphil/eval/multishine_benchmark.ex native/exphil_peppi/src/lib.rs), &{&1, sha.(&1)})}
File.write!(Path.join(out, "protocol.json"), JSON.encode!(protocol))

runs = for stage <- stages, opponent <- opponents, run <- 1..count do
  id = "#{stage}_#{opponent}_#{run}"
  directory = Path.join(out, id)
  File.mkdir!(directory)
  report = Path.join(directory, "session.json")
  args = ["run", "scripts/play_dolphin.exs", "--policy", policy,
    "--dolphin", opts[:dolphin] || Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
    "--iso", Path.expand("~/isos/melee.iso"), "--replay-dir", directory,
    "--session-report", report, "--character", "fox", "--stage", stage,
    "--dummy", mode, "--dummy-character", opponent, "--dummy-cpu-level", to_string(if(mode == "cpu", do: opts[:level] || 1, else: 0)),
    "--seconds", to_string(seconds), "--reaction-delay", "0", "--live-af",
    "--headless", "--no-audio", "--emulation-speed", "0", "--blocking-input", "--deterministic"]
  File.write!(Path.join(directory, "args.json"), JSON.encode!(args))
  IO.puts("Starting #{id}")
  {_, status} = System.cmd("timeout", [to_string(seconds + 240), "mix" | args],
    into: File.stream!(Path.join(directory, "play.log"), [:write]), stderr_to_stdout: true,
    env: [{"EXLA_TARGET", "cuda"}, {"EXPHIL_GPU_MEMORY_FRACTION", "0.15"}])
  result = try do
    session = report |> File.read!() |> JSON.decode!()
    [replay_path] = Path.wildcard(Path.join(directory, "**/*.slp"))
    {:ok, replay} = Peppi.parse(replay_path)
    true = Melee.Enums.Stage.from_external(replay.metadata.stage) == String.to_existing_atom(stage)
    if stage == "pokemon_stadium", do: true = replay.metadata.frozen_stadium == true
    opponent_meta = Enum.find(replay.metadata.players, &(&1.port == 2))
    true = String.downcase(opponent_meta.character_name) == opponent
    true = is_integer(session["last_frame"])
    rows = Benchmark.rows(replay, 1) |> Enum.filter(&(&1.frame <= session["last_frame"]))
    metrics = Benchmark.score(rows)
    valid = status == 0 and session["status"] == "ok" and session["errors"] == 0 and
      session["measured_latency"] == 1 and length(rows) == seconds * 60
    %{valid: valid, session: session, metrics: metrics, replay: replay_path,
      replay_sha256: sha.(replay_path), chain_gate: valid and metrics.max_chain >= 30}
  rescue
    e -> %{valid: false, error: Exception.message(e), exit_status: status, chain_gate: false}
  end
  result = Map.merge(result, %{id: id, stage: stage, opponent: opponent, run: run})
  File.write!(Path.join(directory, "result.json"), Jason.encode!(result, pretty: true))
  IO.puts("#{id}: valid=#{result.valid} chain=#{get_in(result, [:metrics, :max_chain])} error=#{result[:error]}")
  result
end
File.write!(Path.join(out, "summary.json"), Jason.encode!(%{protocol: protocol, runs: runs}, pretty: true))
unless Enum.all?(runs, & &1.valid), do: System.halt(1)
