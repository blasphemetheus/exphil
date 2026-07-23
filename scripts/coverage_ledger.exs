# Coverage ledger: bot visitation vs corpus visitation (DATA_FLYWHEEL_DESIGN
# 2026-07-23, stage A3). Where does human play go that the bot never goes?
# Diffs situation-occupancy distributions and appends the most
# under-visited buckets to the gap ledger as coverage gaps.
#
#   # Compute a corpus ledger once, cache it:
#   mix run scripts/coverage_ledger.exs --corpus 'corpus/archive/mewtwo/*.slp' \
#     --save-corpus cache/coverage_corpus_mewtwo.json --bot 'probes/newera8/r16/**/*.slp'
#
#   # Reuse the cached corpus next time:
#   mix run scripts/coverage_ledger.exs --corpus-ledger cache/coverage_corpus_mewtwo.json \
#     --bot 'probes/newera8/r17/**/*.slp'
#
# Options:
#   --bot GLOB[,GLOB]      Bot replays (required)
#   --corpus GLOB[,GLOB]   Corpus replays (or use --corpus-ledger)
#   --corpus-ledger PATH   Cached corpus ledger JSON (skips corpus parse)
#   --save-corpus PATH     Write the computed corpus ledger here for reuse
#   --bot-port N           Bot's port in the bot replays (default 1)
#   --top K                Under-visited buckets to append (default 20)
#   --alpha F              Laplace pseudocount (default 1.0)
#   --min-corpus-frac F    Drop buckets the corpus barely visits (default 0.002)
#   --gaps PATH            Gap ledger (default scenarios/gaps.json)
#   --out PATH             Diff report JSON (default logs/coverage_<ts>.json)
#   --today YYYY-MM-DD
#   --quiet
#
# UNTESTED as of 2026-07-23 (written while r16 held mix) — VERIFICATION
# PENDING, see docs/planning/DATA_FLYWHEEL_DESIGN_2026-07-23.md.

require Logger

alias ExPhil.Eval.{Coverage, GapLedger, ScenarioScan}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      bot: :string,
      corpus: :string,
      corpus_ledger: :string,
      save_corpus: :string,
      bot_port: :integer,
      top: :integer,
      alpha: :float,
      min_corpus_frac: :float,
      gaps: :string,
      out: :string,
      today: :string,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

unless opts[:bot] do
  Output.error("--bot is required")
  System.halt(1)
end

unless opts[:corpus] || opts[:corpus_ledger] do
  Output.error("one of --corpus / --corpus-ledger is required")
  System.halt(1)
end

expand = fn spec ->
  (spec || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.uniq()
end

bot_port = opts[:bot_port] || 1

# Build an occupancy ledger from replay globs (p1 = subject; flip when the
# subject played port 2).
build_ledger = fn paths, flip? ->
  Enum.reduce(paths, {%{}, 0, 0}, fn path, {acc, ok, bad} ->
    case ScenarioScan.load(path) do
      {:ok, %{frames: frames}} ->
        frames = if flip?, do: Enum.map(frames, fn f -> %{f | p1: f.p2, p2: f.p1} end), else: frames
        {Coverage.merge(acc, Coverage.ledger(frames)), ok + 1, bad}

      {:error, _} ->
        {acc, ok, bad + 1}
    end
  end)
end

Output.banner("Coverage Ledger")

bot_paths = expand.(opts[:bot])

if bot_paths == [] do
  Output.error("no bot replays matched --bot")
  System.halt(1)
end

Output.puts("Building bot ledger from #{length(bot_paths)} replay(s)...")
{bot_ledger, bot_ok, bot_bad} = build_ledger.(bot_paths, bot_port == 2)
Output.puts("  bot: #{bot_ok} parsed, #{bot_bad} skipped, #{map_size(bot_ledger)} buckets")

{corpus_ledger, corpus_src} =
  cond do
    opts[:corpus_ledger] ->
      {Coverage.load(opts[:corpus_ledger]), opts[:corpus_ledger]}

    true ->
      corpus_paths = expand.(opts[:corpus])
      Output.puts("Building corpus ledger from #{length(corpus_paths)} replay(s)...")
      {cl, cok, cbad} = build_ledger.(corpus_paths, false)
      Output.puts("  corpus: #{cok} parsed, #{cbad} skipped, #{map_size(cl)} buckets")
      if opts[:save_corpus], do: Coverage.save(cl, opts[:save_corpus])
      {cl, opts[:corpus] || "computed"}
  end

diff =
  Coverage.diff(bot_ledger, corpus_ledger,
    alpha: opts[:alpha] || 1.0,
    min_corpus_frac: opts[:min_corpus_frac] || 0.002
  )

top_n = opts[:top] || 20
under = Enum.take(diff, top_n)

# Coverage gaps: no single slp/frame — the bucket key IS the gap identity.
gap_attrs =
  Enum.map(under, fn r ->
    %{
      source: "coverage",
      type: r.key,
      slp: nil,
      frame: nil,
      note:
        "under-visited: bot #{Float.round(r.bot_frac * 100, 3)}% vs corpus " <>
          "#{Float.round(r.corpus_frac * 100, 3)}% (ratio #{Float.round(r.ratio, 3)})",
      evidence: %{
        "bot_count" => r.bot_count,
        "corpus_count" => r.corpus_count,
        "ratio" => Float.round(r.ratio, 4)
      }
    }
  end)

gaps_path = opts[:gaps] || "scenarios/gaps.json"
ledger = GapLedger.load(gaps_path)
{ledger, added} = GapLedger.append(ledger, gap_attrs, today: opts[:today])
GapLedger.save(ledger, gaps_path)

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
out_path = opts[:out] || "logs/coverage_#{ts}.json"
File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(
    %{
      timestamp: ts,
      corpus_source: corpus_src,
      bot_buckets: map_size(bot_ledger),
      corpus_buckets: map_size(corpus_ledger),
      under_visited: under
    },
    pretty: true
  )
)

Output.puts("")
Output.puts("Most under-visited buckets (bot vs corpus):")

for r <- Enum.take(under, 10) do
  Output.puts(
    "  #{String.pad_trailing(r.key, 38)} bot=#{Float.round(r.bot_frac * 100, 2)}% " <>
      "corpus=#{Float.round(r.corpus_frac * 100, 2)}% ratio=#{Float.round(r.ratio, 3)}"
  )
end

Output.puts("")
Output.puts("appended #{added} coverage gap(s) to #{gaps_path}")
Output.success("diff report -> #{out_path}")
