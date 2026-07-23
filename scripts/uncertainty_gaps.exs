# Uncertainty -> gap ledger (DATA_FLYWHEEL_DESIGN 2026-07-23, stage A4
# consumer). Reads per-frame confidence JSONL written by the agent's
# --uncertainty-log, finds the LOW-confidence clusters — states the
# training data doesn't cover, a gap signal that fires before failures
# do — and appends them to the gap ledger.
#
#   mix run scripts/uncertainty_gaps.exs --log logs/uncertainty_r16.jsonl \
#     --slp probes/newera8/r16/r13/plain/p1/Game_X.slp
#
# Options:
#   --log PATH[,PATH]   Confidence JSONL file(s) (required)
#   --slp PATH          Replay the session recorded (attached to gaps so
#                       they're drillable; omit for a report-only run)
#   --percentile F      Keep frames below this confidence percentile
#                       (default 5.0)
#   --metric NAME       overall|buttons|main|c|shoulder (default overall)
#   --gap-frames N      Min frames between kept gaps (default 240)
#   --min-frame N       Earliest frame (default 300, prefix room)
#   --top K             Cap gaps appended (default 10)
#   --gaps PATH         Gap ledger (default scenarios/gaps.json)
#   --quiet

require Logger

alias ExPhil.Eval.GapLedger
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      log: :string,
      slp: :string,
      percentile: :float,
      metric: :string,
      gap_frames: :integer,
      min_frame: :integer,
      top: :integer,
      gaps: :string,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

unless opts[:log] do
  Output.error("--log is required")
  System.halt(1)
end

metric = opts[:metric] || "overall"

unless metric in ~w(overall buttons main c shoulder) do
  Output.error("--metric must be one of overall|buttons|main|c|shoulder")
  System.halt(1)
end

percentile = opts[:percentile] || 5.0
gap_frames = opts[:gap_frames] || 240
min_frame = opts[:min_frame] || 300
top_n = opts[:top] || 10
gaps_path = opts[:gaps] || "scenarios/gaps.json"

log_paths =
  opts[:log]
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)

entries =
  log_paths
  |> Enum.flat_map(fn path ->
    path
    |> File.stream!()
    |> Enum.flat_map(fn line ->
      case Jason.decode(String.trim(line)) do
        {:ok, %{"f" => f} = e} when is_integer(f) -> [e]
        _ -> []
      end
    end)
  end)
  |> Enum.filter(fn e -> is_number(e[metric]) and e["f"] >= min_frame end)

if entries == [] do
  Output.error("No usable entries in #{inspect(log_paths)}")
  System.halt(1)
end

Output.banner("Uncertainty Gaps")

values = entries |> Enum.map(& &1[metric]) |> Enum.sort()
cutoff = Enum.at(values, max(trunc(length(values) * percentile / 100) - 1, 0))

low =
  entries
  |> Enum.filter(fn e -> e[metric] <= cutoff end)
  |> Enum.sort_by(& &1["f"])

# Space clusters out — consecutive uncertain frames are one situation.
spaced =
  Enum.reduce(low, [], fn e, acc ->
    case acc do
      [prev | _] -> if e["f"] - prev["f"] < gap_frames, do: acc, else: [e | acc]
      [] -> [e | acc]
    end
  end)
  |> Enum.reverse()
  |> Enum.sort_by(& &1[metric])
  |> Enum.take(top_n)

Output.config([
  {"Frames", length(entries)},
  {"Metric", metric},
  {"P#{percentile} cutoff", Float.round(cutoff * 1.0, 3)},
  {"Clusters kept", length(spaced)},
  {"Replay", opts[:slp] || "(none — report only)"}
])

for e <- spaced do
  Output.puts("  f=#{e["f"]}  #{metric}=#{Float.round(e[metric] * 1.0, 3)}")
end

if opts[:slp] do
  gap_attrs =
    Enum.map(spaced, fn e ->
      %{
        source: "uncertainty",
        type: "low_confidence_#{metric}",
        slp: Path.expand(opts[:slp]),
        frame: e["f"],
        note: "#{metric}=#{Float.round(e[metric] * 1.0, 3)} (p#{percentile} cutoff #{Float.round(cutoff * 1.0, 3)})",
        evidence: Map.take(e, ~w(overall buttons main c shoulder))
      }
    end)

  ledger = GapLedger.load(gaps_path)
  {ledger, added} = GapLedger.append(ledger, gap_attrs)
  GapLedger.save(ledger, gaps_path)
  Output.success("appended #{added} uncertainty gap(s) -> #{gaps_path}")
else
  Output.puts("")
  Output.puts("(no --slp given — nothing appended to the ledger)")
end
