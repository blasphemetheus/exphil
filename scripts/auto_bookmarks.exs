# Automatic gap detection over the bot's own games (DATA_FLYWHEEL_DESIGN
# 2026-07-23, stage A1). Runs the FailureScan outcome detectors across
# replay globs and appends drill handoffs to the shared gap ledger — so
# playing the bot GENERATES its own drill backlog (manual bookmarks via
# scan_bookmarks.exs become the high-signal layer on top).
#
#   mix run scripts/auto_bookmarks.exs probes/newera8/r16/**/*.slp
#   mix run scripts/auto_bookmarks.exs --port 2 --manifest-out /tmp/m.json \
#     corpus/sparring/2026-07-24/*.slp
#
# Options:
#   --port N            Bot's port in these replays (default 1, probe
#                       convention; use 2 for sparring where you were P1)
#   --types a,b         Only these FailureScan types
#   --min-frame N       Earliest handoff frame (default 300)
#   --gap N             Min frames between same-type handoffs (default 240)
#   --gaps PATH         Gap ledger to append to (default scenarios/gaps.json)
#   --manifest-out PATH Also write handoffs as scenario-manifest entries
#   --today YYYY-MM-DD  Override the created date (default today, UTC)
#   --quiet
#
# UNTESTED as of 2026-07-23 (written while r16 held mix) — see the test
# plan in docs/planning/DATA_FLYWHEEL_DESIGN_2026-07-23.md.

require Logger

alias ExPhil.Eval.{FailureScan, GapLedger}
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [
      port: :integer,
      types: :string,
      min_frame: :integer,
      gap: :integer,
      gaps: :string,
      manifest_out: :string,
      today: :string,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

port = opts[:port] || 1

scan_opts =
  [min_frame: opts[:min_frame] || 300, gap: opts[:gap] || 240]
  |> then(fn o ->
    case opts[:types] do
      nil ->
        o

      s ->
        wanted =
          s
          |> String.split(",", trim: true)
          |> Enum.map(&String.to_existing_atom/1)

        Keyword.put(o, :types, wanted)
    end
  end)

slp_paths =
  paths
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.uniq()

if slp_paths == [] do
  Output.error("No replays. Usage: mix run scripts/auto_bookmarks.exs <slp...>")
  System.halt(1)
end

gaps_path = opts[:gaps] || "scenarios/gaps.json"
today = opts[:today]

Output.banner("Auto Bookmarks (FailureScan)")

Output.config([
  {"Replays", length(slp_paths)},
  {"Bot port", port},
  {"Types", opts[:types] || "all"},
  {"Gap ledger", gaps_path}
])

# One flat list of gap-attr maps (loose keys; GapLedger normalizes + dedupes).
{gap_attrs, per_type} =
  Enum.reduce(slp_paths, {[], %{}}, fn path, {acc, counts} ->
    case FailureScan.load(path) do
      {:error, reason} ->
        Output.warning("skipping #{Path.basename(path)}: #{inspect(reason)}")
        {acc, counts}

      {:ok, frames} ->
        # FailureScan's P1 = the bot; flip if the bot played port 2.
        frames = if port == 2, do: FailureScan.flip(frames), else: frames
        cands = FailureScan.scan(frames, scan_opts)

        attrs =
          Enum.map(cands, fn c ->
            %{
              source: "detector",
              type: c.type,
              slp: path,
              frame: c.frame,
              note: c.note,
              evidence: %{"detector" => to_string(c.type), "port" => port}
            }
          end)

        counts =
          Enum.reduce(cands, counts, fn c, m -> Map.update(m, c.type, 1, &(&1 + 1)) end)

        {acc ++ attrs, counts}
    end
  end)

ledger = GapLedger.load(gaps_path)
{ledger, added} = GapLedger.append(ledger, gap_attrs, today: today)
GapLedger.save(ledger, gaps_path)

Output.puts("")
Output.puts("detected:  #{length(gap_attrs)} handoff(s) — #{inspect(per_type)}")
Output.puts("appended:  #{added} new (#{length(gap_attrs) - added} already in ledger)")
Output.success("gap ledger -> #{gaps_path} (#{length(ledger["gaps"])} total)")

if opts[:manifest_out] do
  entries =
    Enum.map(gap_attrs, fn a ->
      %{"slp" => a.slp, "frame" => a.frame, "type" => to_string(a.type), "note" => a.note}
    end)

  File.mkdir_p!(Path.dirname(opts[:manifest_out]))
  File.write!(opts[:manifest_out], Jason.encode!(%{"entries" => entries}, pretty: true))
  Output.success("manifest (#{length(entries)}) -> #{opts[:manifest_out]}")
end
