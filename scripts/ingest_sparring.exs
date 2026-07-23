# Sparring capture (DATA_FLYWHEEL_DESIGN 2026-07-23, stage B2).
#
# Every human-vs-bot game is (a) a labeled record of where the bot fails
# against real humans and (b) on-distribution human corrections. This
# ingests a session's .slp files so neither evaporates:
#
#   1. copies them to corpus/sparring/<date>/ with a session.json sidecar
#      ({human_port, human_tag, bot_port, bot_checkpoint, games})
#   2. runs coach_report.exs on the set (gaps -> ledger) unless --no-report
#   3. registers the HUMAN port for BC ingestion via the sidecar — the
#      bot's port is NEVER BC'd (those are policy actions)
#
#   mix run scripts/ingest_sparring.exs --bot-port 2 \
#     --checkpoint checkpoints/mewtwo_combo_newera_r16_r13_policy.bin \
#     ~/Slippi/Game_20260723T2*.slp
#
# Options:
#   --bot-port N       Which port the BOT played (required unless --bot-tag)
#   --bot-tag TAG      Detect the bot by in-game name tag (parser exposes
#                      name_tag, NOT the netplay code — see parser-netplay-code)
#   --human-tag TAG    Style tag for the human port (default human_blewf)
#   --checkpoint PATH  Bot checkpoint that played (recorded in the sidecar;
#                      the .slp does not carry it)
#   --date YYYY-MM-DD  Session date dir (default today, local)
#   --out-root DIR     Corpus root (default corpus/sparring)
#   --gaps PATH        Gap ledger for the coach report (default scenarios/gaps.json)
#   --no-report        Skip the coach report step
#   --quiet
#
# BC usage after ingest (human side only):
#   --bc-replays 'corpus/sparring/<date>/*.slp' with --port from the
#   sidecar's human_port (port-honoring ingestion hook is a follow-on;
#   until then pass it explicitly).

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [
      bot_port: :integer,
      bot_tag: :string,
      human_tag: :string,
      checkpoint: :string,
      date: :string,
      out_root: :string,
      gaps: :string,
      no_report: :boolean,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

slp_paths =
  paths
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.uniq()

if slp_paths == [] do
  Output.error("No replays. Usage: mix run scripts/ingest_sparring.exs --bot-port N <slp...>")
  System.halt(1)
end

unless opts[:bot_port] || opts[:bot_tag] do
  Output.error("--bot-port (or --bot-tag) is required — which side was the bot?")
  System.halt(1)
end

# Resolve the bot port once per file (tag lookup) or globally (explicit).
detect_bot_port = fn path ->
  cond do
    opts[:bot_port] ->
      opts[:bot_port]

    true ->
      case Peppi.metadata(path) do
        {:ok, %{players: players}} when is_list(players) ->
          case Enum.find(players, fn p -> p.tag == opts[:bot_tag] end) do
            %{port: p} -> p
            _ -> nil
          end

        _ ->
          nil
      end
  end
end

date = opts[:date] || Date.to_iso8601(Date.utc_today())
out_dir = Path.join(opts[:out_root] || "corpus/sparring", date)
File.mkdir_p!(out_dir)

Output.banner("Sparring Ingest")

Output.config([
  {"Games", length(slp_paths)},
  {"Bot", opts[:bot_port] || "tag=#{opts[:bot_tag]}"},
  {"Human tag", opts[:human_tag] || "human_blewf"},
  {"Checkpoint", opts[:checkpoint] || "(unrecorded)"},
  {"Out", out_dir}
])

games =
  Enum.flat_map(slp_paths, fn path ->
    case detect_bot_port.(path) do
      nil ->
        Output.warning("skipping #{Path.basename(path)}: bot port undetectable")
        []

      bot_port ->
        dest = Path.join(out_dir, Path.basename(path))
        File.cp!(path, dest)

        [
          %{
            "file" => Path.basename(path),
            "bot_port" => bot_port,
            "human_port" => if(bot_port == 1, do: 2, else: 1)
          }
        ]
    end
  end)

if games == [] do
  Output.error("No games ingested")
  System.halt(1)
end

# One sidecar per session. Mixed bot ports across games are legal (port
# assignment can flip between netplay games) — per-game ports live in
# "games"; the top-level fields describe the session.
bot_ports = games |> Enum.map(& &1["bot_port"]) |> Enum.uniq()

sidecar = %{
  "date" => date,
  "human_tag" => opts[:human_tag] || "human_blewf",
  "bot_checkpoint" => opts[:checkpoint],
  "bot_ports" => bot_ports,
  "games" => games,
  "bc_note" => "BC the HUMAN port only (see per-game human_port); the bot port is policy output, not demonstration"
}

sidecar_path = Path.join(out_dir, "session.json")

# Merge if the session dir already has games from an earlier ingest today.
sidecar =
  case File.read(sidecar_path) do
    {:ok, bin} ->
      case Jason.decode(bin) do
        {:ok, prev} ->
          seen = MapSet.new(prev["games"] || [], & &1["file"])
          fresh = Enum.reject(games, &MapSet.member?(seen, &1["file"]))
          Map.put(sidecar, "games", (prev["games"] || []) ++ fresh)

        _ ->
          sidecar
      end

    _ ->
      sidecar
  end

File.write!(sidecar_path, Jason.encode!(sidecar, pretty: true))
Output.success("#{length(games)} game(s) -> #{out_dir} (session.json written)")

# Coach report on the ingested copies. Per-game bot-port variance is rare;
# the report runs once per distinct bot port with just that port's games.
unless opts[:no_report] do
  for bp <- bot_ports do
    set = for g <- games, g["bot_port"] == bp, do: Path.join(out_dir, g["file"])

    args =
      ["run", "--no-compile", "scripts/coach_report.exs", "--bot-port", to_string(bp)] ++
        (if opts[:gaps], do: ["--gaps", opts[:gaps]], else: []) ++ set

    Output.puts("Running coach report (bot port #{bp}, #{length(set)} game(s))...")

    case System.cmd("mix", args, into: IO.stream(:stdio, :line), stderr_to_stdout: true) do
      {_, 0} -> :ok
      {_, code} -> Output.warning("coach_report exited #{code} — run it manually on #{out_dir}")
    end
  end
end

Output.puts("")
Output.puts("BC the human side with e.g.:")

Output.puts(
  "  --bc-replays '#{out_dir}/*.slp'  (human_port per session.json; " <>
    "pass the port explicitly until sidecar-honoring ingestion lands)"
)
