# Post-set coach report (DATA_FLYWHEEL_DESIGN 2026-07-23, stage A2).
#
# Read the .slp files of one set (a Yeti exhibition, a Direct sparring
# night, a probe fan-out), and emit ONE report: where the bot lost
# neutral, what it never converted, how it died, plus the top drill
# candidates appended to the gap ledger. This is the artifact to read
# after every human-vs-bot set, and the primary LLM-coach ingest.
#
#   mix run scripts/coach_report.exs --bot-port 2 corpus/sparring/2026-07-24/*.slp
#   mix run scripts/coach_report.exs probes/newera8/r16/**/*.slp   # bot=P1 default
#
# Options:
#   --bot-port N        Bot's port (RELIABLE; default 1 = probe convention)
#   --bot-tag TAG       Detect the bot by in-game NAME TAG instead (note:
#                       the parser exposes name_tag, NOT the netplay connect
#                       code EXPH#288 — so --bot-port is the robust path)
#   --gaps PATH         Gap ledger (default scenarios/gaps.json)
#   --top N             Gap candidates to append per set (default 10)
#   --out DIR           Report dir (default logs/coach/<ts>)
#   --char NAME         Character pack for stats context (default mewtwo)
#   --today YYYY-MM-DD  created date override
#   --quiet
#
# UNTESTED as of 2026-07-23 (written while r16 held mix) — test plan in
# docs/planning/DATA_FLYWHEEL_DESIGN_2026-07-23.md.

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Eval.{FailureScan, GapLedger}
alias ExPhil.Interp.ReplayStats
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [
      bot_port: :integer,
      bot_tag: :string,
      gaps: :string,
      top: :integer,
      out: :string,
      char: :string,
      today: :string,
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
  Output.error("No replays. Usage: mix run scripts/coach_report.exs <slp...>")
  System.halt(1)
end

top_n = opts[:top] || 10
gaps_path = opts[:gaps] || "scenarios/gaps.json"
ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
out_dir = opts[:out] || "logs/coach/#{ts}"

# Detect the bot's port for one replay: explicit override wins; else name-tag
# match; else default 1 (probe convention).
detect_bot_port = fn path ->
  cond do
    opts[:bot_port] ->
      opts[:bot_port]

    opts[:bot_tag] ->
      case Peppi.metadata(path) do
        {:ok, %{players: players}} when is_list(players) ->
          case Enum.find(players, fn p -> p.tag == opts[:bot_tag] end) do
            %{port: p} -> p
            _ -> 1
          end

        _ ->
          1
      end

    true ->
      1
  end
end

Output.banner("Coach Report")

Output.config([
  {"Set", "#{length(slp_paths)} game(s)"},
  {"Bot port", opts[:bot_port] || opts[:bot_tag] || "1 (default)"},
  {"Char", opts[:char] || "mewtwo"},
  {"Gap ledger", gaps_path}
])

# ---- per-game analysis -----------------------------------------------------

analyze = fn path ->
  bot_port = detect_bot_port.(path)
  opp_port = if bot_port == 1, do: 2, else: 1

  with {:ok, frames} <- FailureScan.load(path) do
    frames = if bot_port == 2, do: FailureScan.flip(frames), else: frames

    # Combat aggregate via ReplayStats (its load is port-1 canonical; select
    # the bot's sub-map so the numbers describe the BOT).
    rs = ReplayStats.load(path)
    bot_sub = rs[if(bot_port == 1, do: :p1, else: :p2)]
    opp_sub = rs[if(bot_port == 1, do: :p2, else: :p1)]

    approach = ReplayStats.approach_stats(bot_sub, opp_sub)
    conv = ReplayStats.conversion_stats(bot_sub, opp_sub)

    fails = FailureScan.scan(frames)
    by_type = Enum.frequencies_by(fails, & &1.type)

    %{
      path: path,
      bot_port: bot_port,
      opp_port: opp_port,
      armed_per_min: approach.armed_per_min,
      approaches: approach.approaches,
      conversions: conv.conversions,
      conv_rate: if(conv.approaches > 0, do: conv.conversions / conv.approaches, else: 0.0),
      neutral_losses: Map.get(by_type, :neutral_loss, 0),
      dropped_punishes: Map.get(by_type, :dropped_punish, 0),
      deaths: Map.get(by_type, :death_sequence, 0),
      passivity: Map.get(by_type, :passivity_window, 0),
      candidates: fails
    }
  else
    {:error, reason} ->
      Output.warning("skipping #{Path.basename(path)}: #{inspect(reason)}")
      nil
  end
end

games = slp_paths |> Enum.map(analyze) |> Enum.reject(&is_nil/1)

if games == [] do
  Output.error("No parseable games in the set")
  System.halt(1)
end

# ---- aggregate -------------------------------------------------------------

sum = fn key -> Enum.sum(Enum.map(games, &Map.fetch!(&1, key))) end
mean = fn key -> Enum.sum(Enum.map(games, &Map.fetch!(&1, key))) / length(games) end

agg = %{
  games: length(games),
  mean_armed_per_min: Float.round(mean.(:armed_per_min) * 1.0, 2),
  approaches: sum.(:approaches),
  conversions: sum.(:conversions),
  neutral_losses: sum.(:neutral_losses),
  dropped_punishes: sum.(:dropped_punishes),
  deaths: sum.(:deaths),
  passivity: sum.(:passivity)
}

# Top-N drill candidates across the set, appended to the ledger.
all_candidates =
  Enum.flat_map(games, fn g ->
    Enum.map(g.candidates, fn c ->
      %{
        source: "coach",
        type: c.type,
        slp: g.path,
        frame: c.frame,
        note: c.note,
        evidence: %{"set_ts" => ts, "bot_port" => g.bot_port}
      }
    end)
  end)

# Prioritize the pathologies that map to the gate-10 story first.
priority = %{passivity_window: 0, dropped_punish: 1, neutral_loss: 2, death_sequence: 3}

top_candidates =
  all_candidates
  |> Enum.sort_by(fn a -> {Map.get(priority, a.type, 9), a.frame} end)
  |> Enum.take(top_n)

ledger = GapLedger.load(gaps_path)
{ledger, added} = GapLedger.append(ledger, top_candidates, today: opts[:today])
GapLedger.save(ledger, gaps_path)

# ---- existing-drill cross-reference ---------------------------------------

existing_types =
  case File.read("scenarios/manifest.json") do
    {:ok, bin} ->
      case Jason.decode(bin) do
        {:ok, %{"entries" => entries}} -> entries |> Enum.map(& &1["type"]) |> Enum.uniq()
        _ -> []
      end

    _ ->
      []
  end

# ---- write report ----------------------------------------------------------

File.mkdir_p!(out_dir)

md = """
# Coach Report — #{ts}

Set: #{length(games)} game(s). Bot port: #{opts[:bot_port] || opts[:bot_tag] || "1 (default)"}.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | #{agg.mean_armed_per_min} |
| Approaches (total) | #{agg.approaches} |
| Conversions (total) | #{agg.conversions} |
| Neutral losses (opened up) | #{agg.neutral_losses} |
| Dropped punishes | #{agg.dropped_punishes} |
| Death sequences | #{agg.deaths} |
| Passivity windows | #{agg.passivity} |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
#{Enum.map_join(games, "\n", fn g -> "| #{Path.basename(g.path)} | #{Float.round(g.armed_per_min * 1.0, 2)} | #{g.approaches} | #{g.conversions} | #{g.neutral_losses} | #{g.dropped_punishes} | #{g.deaths} | #{g.passivity} |" end)}

## Top drill candidates (appended to gap ledger)

#{if top_candidates == [], do: "_none_", else: Enum.map_join(top_candidates, "\n", fn c -> "- **#{c.type}** #{Path.basename(c.slp)}:#{c.frame} — #{c.note}" end)}

Appended #{added} new gap(s) to `#{gaps_path}` (#{length(ledger["gaps"])} total).

## Drills that already exist for these gaps

Existing scenario-manifest types: #{if existing_types == [], do: "_none_", else: Enum.join(existing_types, ", ")}.
Gap types this set surfaced: #{agg |> Map.take([:neutral_losses, :dropped_punishes, :deaths, :passivity]) |> Enum.filter(fn {_k, v} -> v > 0 end) |> Enum.map_join(", ", fn {k, _} -> to_string(k) end)}.
"""

md_path = Path.join(out_dir, "report.md")
json_path = Path.join(out_dir, "report.json")
File.write!(md_path, md)

File.write!(
  json_path,
  Jason.encode!(
    %{
      timestamp: ts,
      aggregate: agg,
      games:
        Enum.map(games, fn g ->
          Map.drop(g, [:candidates]) |> Map.put(:candidate_count, length(g.candidates))
        end),
      gaps_appended: added,
      gaps_path: gaps_path
    },
    pretty: true
  )
)

Output.puts("")
Output.divider()

Output.puts(
  "SCORE: armed/min=#{agg.mean_armed_per_min} conv=#{agg.conversions}/#{agg.approaches} " <>
    "neutral_loss=#{agg.neutral_losses} dropped=#{agg.dropped_punishes} " <>
    "deaths=#{agg.deaths} passive=#{agg.passivity}"
)

Output.success("report -> #{md_path}")
Output.success("json   -> #{json_path}")
