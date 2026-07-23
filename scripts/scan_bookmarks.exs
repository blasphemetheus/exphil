# Input-signature bookmark scanner (drills design 2026-07-23, feature C).
#
# During play (netplay OR console — the marker travels inside the .slp),
# press an inert D-pad direction to bookmark "the bot needs to learn this
# moment". This scans replays for those signatures and emits handoff
# candidates for the scenario suite / TM-CE savestate import / drill
# generation.
#
#   mix run scripts/scan_bookmarks.exs ~/Slippi/Game_2026*.slp
#   mix run scripts/scan_bookmarks.exs --port 1 --classify \
#     --manifest-out /tmp/bookmark_manifest.json ~/Slippi/Game_X.slp
#
# Options:
#   --port N             Only scan this port (default: all occupied ports)
#   --signature S        d_down (default) | d_left | d_right  (d_up is taunt)
#   --min-hold N         Signature must be held >= N frames (default 2)
#   --cluster N          Merge marks within N frames, keep first (default 120)
#   --lookback N         handoff = mark - N (default 120; the situation
#                        started before you reacted)
#   --min-frame N        Drop handoffs earlier than N (default 300 — the
#                        suite needs prefix room; see SCENARIOS.md)
#   --classify           Match bookmarks against ScenarioScan candidates
#   --classify-window N  Max |candidate - handoff| to accept (default 240)
#   --out PATH           Bookmarks JSON (default logs/bookmarks_<ts>.json)
#   --manifest-out PATH  Also write classified marks as manifest entries
#   --quiet
#
# UNTESTED as of 2026-07-23 (written while r16 held mix) — see the test
# plan in docs/planning/DRILLS_DESIGN_2026-07-23.md.

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ScenarioScan
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [
      port: :integer,
      signature: :string,
      min_hold: :integer,
      cluster: :integer,
      lookback: :integer,
      min_frame: :integer,
      classify: :boolean,
      classify_window: :integer,
      out: :string,
      manifest_out: :string,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

signature_key =
  case opts[:signature] || "d_down" do
    "d_down" -> :button_d_down
    "d_left" -> :button_d_left
    "d_right" -> :button_d_right
    other ->
      Output.error("Unknown signature #{inspect(other)} (d_down|d_left|d_right)")
      System.halt(1)
  end

min_hold = opts[:min_hold] || 2
cluster = opts[:cluster] || 120
lookback = opts[:lookback] || 120
min_frame = opts[:min_frame] || 300
classify_window = opts[:classify_window] || 240

slp_paths =
  paths
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.uniq()

if slp_paths == [] do
  Output.error("No replays. Usage: mix run scripts/scan_bookmarks.exs <slp...>")
  System.halt(1)
end

Output.banner("Bookmark Scanner")

Output.config([
  {"Replays", length(slp_paths)},
  {"Signature", signature_key},
  {"Min hold", min_hold},
  {"Cluster", cluster},
  {"Lookback", lookback},
  {"Classify", opts[:classify] || false}
])

# Rising-edge scan: frames where the signature button transitions
# false->true and stays held >= min_hold frames.
detect_marks = fn frames, port ->
  frames
  |> Enum.map(fn f ->
    case f.players[port] do
      %{controller: c} when c != nil -> {f.frame_number, Map.get(c, signature_key, false)}
      _ -> {f.frame_number, false}
    end
  end)
  |> Enum.chunk_by(&elem(&1, 1))
  |> Enum.filter(fn [{_f, held} | _] = run -> held and length(run) >= min_hold end)
  |> Enum.map(fn [{frame, _} | _] -> frame end)
end

# Merge marks within `cluster` frames of the previous kept mark.
cluster_marks = fn marks ->
  marks
  |> Enum.sort()
  |> Enum.reduce([], fn m, acc ->
    case acc do
      [prev | _] when m - prev <= cluster -> acc
      _ -> [m | acc]
    end
  end)
  |> Enum.reverse()
end

bookmarks =
  Enum.flat_map(slp_paths, fn path ->
    case Peppi.parse(path) do
      {:error, reason} ->
        Output.warning("skipping #{Path.basename(path)}: #{inspect(reason)}")
        []

      {:ok, replay} ->
        ports =
          case opts[:port] do
            nil ->
              replay.frames
              |> Enum.take(1)
              |> Enum.flat_map(fn f -> Map.keys(f.players) end)

            p ->
              [p]
          end

        # Classification candidates once per replay (only if requested).
        candidates =
          if opts[:classify] do
            frames = ScenarioScan.load(path)
            ScenarioScan.scan(frames)
          else
            []
          end

        for port <- ports,
            mark <- cluster_marks.(detect_marks.(replay.frames, port)),
            handoff = mark - lookback,
            handoff >= min_frame do
          match =
            candidates
            |> Enum.filter(fn c -> abs(c.frame - handoff) <= classify_window end)
            |> Enum.min_by(fn c -> abs(c.frame - handoff) end, fn -> nil end)

          %{
            "slp" => path,
            "port" => port,
            "mark_frame" => mark,
            "frame" => if(match, do: match.frame, else: handoff),
            "type" => if(match, do: to_string(match.type), else: nil),
            "note" =>
              if match do
                "bookmark@#{mark} -> #{match.note}"
              else
                "bookmark@#{mark} (unclassified, handoff=mark-#{lookback})"
              end
          }
        end
    end
  end)

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
out_path = opts[:out] || "logs/bookmarks_#{ts}.json"
File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Jason.encode!(%{"bookmarks" => bookmarks}, pretty: true))

classified = Enum.filter(bookmarks, & &1["type"])

Output.puts("")
Output.puts("bookmarks found:  #{length(bookmarks)} across #{length(slp_paths)} replay(s)")
Output.puts("classified:       #{length(classified)}")
Output.success("bookmarks -> #{out_path}")

# Classified marks in manifest-entry format (drop scanner-only keys) for
# direct use with scenario_suite --manifest.
if opts[:manifest_out] do
  entries = Enum.map(classified, &Map.drop(&1, ["port", "mark_frame"]))
  File.mkdir_p!(Path.dirname(opts[:manifest_out]))
  File.write!(opts[:manifest_out], Jason.encode!(%{"entries" => entries}, pretty: true))
  Output.success("manifest (#{length(entries)} classified) -> #{opts[:manifest_out]}")
end
