# Curate a BC archive by initiation richness (r17 data-quality lever).
#
# r16 took `--bc-sample 20` RANDOM archive games. But some archive Mewtwos
# are campy — cloning them teaches the very passivity gate-10 measures.
# This scores every replay on the behaviors we want cloned (armed
# approaches/min = initiation, opener diversity, conversion rate) and
# selects the games rich in going-in, emitting a ready-to-use --bc-replays
# value.
#
#   mix run scripts/curate_bc.exs 'corpus/archive/mewtwo/*.slp' --top 20 \
#     --out logs/bc_curation.json --list-out logs/bc_selected.txt
#
# Then: --bc-replays "$(cat logs/bc_selected.txt)"  (drop --bc-sample)
#
# Options:
#   --port N          Learner port (default 1)
#   --char NAME       Autodetect the learner port per file by character
#   --top N           Select the top-N by score (default 20)
#   --min-armed F     Also require armed/min >= F to be selected (default 0)
#   --min-frames N    Skip replays under N frames (default 600)
#   --out PATH        Full ranked report JSON (default logs/bc_curation_<ts>.json)
#   --list-out PATH   Comma-joined selected paths for --bc-replays
#   --quiet

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Eval.{FailureScan, NeutralScan, ScenarioScan}
alias ExPhil.Interp.ReplayStats
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [
      port: :integer,
      char: :string,
      top: :integer,
      min_armed: :float,
      min_frames: :integer,
      out: :string,
      list_out: :string,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

min_frames = opts[:min_frames] || 600
top_n = opts[:top] || 20

slp_paths =
  paths
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.uniq()

if slp_paths == [] do
  Output.error("No replays. Usage: mix run scripts/curate_bc.exs <glob> [--top N]")
  System.halt(1)
end

# Learner port per file: explicit --port, or --char autodetect.
learner_port = fn path ->
  case opts[:char] do
    nil ->
      opts[:port] || 1

    char ->
      want = String.downcase(char)

      case Peppi.metadata(path) do
        {:ok, %{players: players}} when is_list(players) ->
          case Enum.find(players, fn p ->
                 String.contains?(String.downcase(p.character_name || ""), want)
               end) do
            %{port: p} -> p
            _ -> nil
          end

        _ ->
          nil
      end
  end
end

Output.banner("BC Archive Curation")

Output.config([
  {"Replays", length(slp_paths)},
  {"Learner", opts[:char] || "port #{opts[:port] || 1}"},
  {"Select top", top_n},
  {"Min armed/min", opts[:min_armed] || 0.0}
])

scored =
  slp_paths
  |> Enum.map(fn path ->
    with port when is_integer(port) <- learner_port.(path),
         {:ok, %{frames: slim}} <- ScenarioScan.load(path),
         true <- length(slim) >= min_frames do
      opp = if port == 1, do: 2, else: 1
      rs = ReplayStats.load(path)
      me = rs[if(port == 1, do: :p1, else: :p2)]
      them = rs[if(port == 1, do: :p2, else: :p1)]

      approach = ReplayStats.approach_stats(me, them)
      conv = ReplayStats.conversion_stats(me, them)
      ns = NeutralScan.summary(me.actions, them.actions)

      # FailureScan passivity on the learner (flip if port 2).
      fs_frames = if port == 2, do: FailureScan.flip(slim), else: slim
      passivity = FailureScan.scan(fs_frames, types: [:passivity_window]) |> length()

      conv_rate = if conv.approaches > 0, do: conv.conversions / conv.approaches, else: 0.0

      # Initiation-richness score: armed/min is the r16-missing signal;
      # opener entropy rewards diverse neutral; passivity/1000f penalizes
      # camping. Deliberately interpretable, not tuned.
      entropy = ns.entropy_bits || 0.0
      passive_rate = passivity / max(length(slim) / 1000, 1)
      score = approach.armed_per_min * (1.0 + 0.3 * entropy) - 0.1 * passive_rate

      %{
        slp: path,
        port: port,
        frames: length(slim),
        armed_per_min: Float.round(approach.armed_per_min * 1.0, 3),
        approaches: approach.approaches,
        conv_rate: Float.round(conv_rate, 3),
        openers: ns.openers,
        opener_entropy: entropy,
        top_opener_share: ns.top_share,
        passivity: passivity,
        score: Float.round(score, 3)
      }
    else
      _ -> nil
    end
  end)
  |> Enum.reject(&is_nil/1)
  |> Enum.sort_by(&(-&1.score))

if scored == [] do
  Output.error("No scorable replays (character absent / too short / unparseable)")
  System.halt(1)
end

selected =
  scored
  |> Enum.filter(fn s -> s.armed_per_min >= (opts[:min_armed] || 0.0) end)
  |> Enum.take(top_n)

agg = fn key -> Enum.sum(Enum.map(scored, &Map.fetch!(&1, key))) / length(scored) end

Output.puts("")
Output.puts("Corpus (#{length(scored)} scorable): mean armed/min=#{Float.round(agg.(:armed_per_min), 3)} " <>
  "conv_rate=#{Float.round(agg.(:conv_rate), 3)} opener_entropy=#{Float.round(agg.(:opener_entropy), 3)}")
Output.puts("")
Output.puts("Top selected (#{length(selected)}):")

for s <- Enum.take(selected, 12) do
  Output.puts(
    "  armed/min=#{String.pad_trailing(to_string(s.armed_per_min), 5)} " <>
      "conv=#{s.conv_rate} openers=#{s.openers} H=#{Float.round(s.opener_entropy, 2)} " <>
      "passive=#{s.passivity}  #{Path.basename(s.slp)}"
  )
end

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
out_path = opts[:out] || "logs/bc_curation_#{ts}.json"
File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(
    %{
      timestamp: ts,
      corpus_size: length(scored),
      selected: length(selected),
      aggregate: %{
        mean_armed_per_min: Float.round(agg.(:armed_per_min), 3),
        mean_conv_rate: Float.round(agg.(:conv_rate), 3),
        mean_opener_entropy: Float.round(agg.(:opener_entropy), 3)
      },
      ranked: scored,
      selected_paths: Enum.map(selected, & &1.slp)
    },
    pretty: true
  )
)

Output.success("ranked report -> #{out_path}")

if opts[:list_out] do
  File.write!(opts[:list_out], Enum.map_join(selected, ",", & &1.slp))
  Output.success("--bc-replays list (#{length(selected)}) -> #{opts[:list_out]}")
end
