# Situation retrieval v1 — bucket-grep (DATA_FLYWHEEL_DESIGN 2026-07-23,
# stage B1 v1). Given a gap (a moment, a bucket key, or a ledger gap id),
# find human-played instances of the SAME situation in a corpus and emit
# them for BC oversampling / TM-CE practice import / drill manifests.
#
# The Fox unlock: ~95k CC0 replays of world-class Fox exist — the
# bottleneck is finding the right 200 frames, not data volume. (v2 will
# use embedding-ANN; this cheap bucket match runs today without #33.)
#
#   # From a moment the bot flubbed:
#   mix run scripts/find_situations.exs --like ~/Slippi/Game_X.slp:4180 \
#     --corpus 'corpus/archive/fox/*.slp' --char-filter fox --max 40
#
#   # From a coverage gap id:
#   mix run scripts/find_situations.exs --gap g_1a2b3c4d \
#     --corpus 'corpus/archive/mewtwo/*.slp'
#
#   # From a raw bucket key:
#   mix run scripts/find_situations.exs --bucket 'd15-30|zmid|aboth|p40-80|ftoward' \
#     --corpus 'corpus/**/*.slp'
#
# Options (exactly one target selector):
#   --like SLP:FRAME     Bucket that moment (subject = --port, default 1)
#   --bucket KEY         Use this bucket key directly
#   --gap ID             Read the gap from --gaps; bucket its moment, or use
#                        its bucket-key type for coverage gaps
#   --corpus GLOB[,GLOB] Corpus to search (required)
#   --char-filter NAME   Keep only replays with a player of this character
#   --port N             Subject port for --like / matches (default 1)
#   --min-frame N        Earliest match frame (default 300, prefix room)
#   --gap-frames N       Min frames between matches in one replay (default 240)
#   --max N              Cap total matches (default 100)
#   --gaps PATH          Gap ledger (default scenarios/gaps.json)
#   --out PATH           Situations JSON (default logs/situations_<ts>.json)
#   --manifest-out PATH  Also write matches as scenario-manifest entries
#   --quiet
#
# UNTESTED as of 2026-07-23 (written while r16 held mix) — VERIFICATION
# PENDING, see docs/planning/DATA_FLYWHEEL_DESIGN_2026-07-23.md.

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Eval.{Coverage, GapLedger, ScenarioScan}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      like: :string,
      bucket: :string,
      gap: :string,
      corpus: :string,
      char_filter: :string,
      port: :integer,
      min_frame: :integer,
      gap_frames: :integer,
      max: :integer,
      gaps: :string,
      out: :string,
      manifest_out: :string,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

port = opts[:port] || 1
min_frame = opts[:min_frame] || 300
gap_frames = opts[:gap_frames] || 240
max_matches = opts[:max] || 100
gaps_path = opts[:gaps] || "scenarios/gaps.json"

selectors = Enum.count([opts[:like], opts[:bucket], opts[:gap]], & &1)

if selectors != 1 do
  Output.error("pass exactly one of --like / --bucket / --gap")
  System.halt(1)
end

unless opts[:corpus] do
  Output.error("--corpus is required")
  System.halt(1)
end

# Bucket a specific (slp, frame) moment for the subject port.
bucket_moment = fn slp, frame_no ->
  case ScenarioScan.load(slp) do
    {:ok, %{frames: frames}} ->
      subj = if port == 2, do: fn f -> %{f | p1: f.p2, p2: f.p1} end, else: & &1

      case Enum.find(frames, fn f -> f.frame == frame_no end) do
        nil ->
          Output.error("frame #{frame_no} not found in #{Path.basename(slp)}")
          System.halt(1)

        f ->
          sf = subj.(f)
          Coverage.bucket(sf.p1, sf.p2)
      end

    {:error, reason} ->
      Output.error("cannot load #{slp}: #{inspect(reason)}")
      System.halt(1)
  end
end

# Resolve the target bucket + the gap id to mark mined (if any).
{target_bucket, mark_gap_id} =
  cond do
    opts[:bucket] ->
      {opts[:bucket], nil}

    opts[:like] ->
      case String.split(opts[:like], ":", parts: 2) do
        [slp, frame_s] -> {bucket_moment.(Path.expand(slp), String.to_integer(frame_s)), nil}
        _ -> Output.error("--like must be SLP:FRAME") && System.halt(1)
      end

    opts[:gap] ->
      ledger = GapLedger.load(gaps_path)

      case Enum.find(ledger["gaps"] || [], &(&1["id"] == opts[:gap])) do
        nil ->
          Output.error("gap #{opts[:gap]} not in #{gaps_path}")
          System.halt(1)

        gap ->
          bucket =
            cond do
              gap["slp"] && gap["frame"] -> bucket_moment.(Path.expand(gap["slp"]), gap["frame"])
              # coverage gaps store the bucket key as the type
              String.contains?(gap["type"] || "", "|") -> gap["type"]
              true -> Output.error("gap #{opts[:gap]} has no moment and no bucket-key type") && System.halt(1)
            end

          {bucket, gap["id"]}
      end
  end

corpus_paths =
  opts[:corpus]
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.uniq()

Output.banner("Find Situations (bucket-grep)")

Output.config([
  {"Target bucket", target_bucket},
  {"Corpus", "#{length(corpus_paths)} replay(s)"},
  {"Char filter", opts[:char_filter] || "none"},
  {"Max matches", max_matches}
])

# Optional character filter via replay metadata.
char_ok? = fn path ->
  case opts[:char_filter] do
    nil ->
      true

    name ->
      want = String.downcase(name)

      case Peppi.metadata(path) do
        {:ok, %{players: players}} when is_list(players) ->
          Enum.any?(players, fn p ->
            String.contains?(String.downcase(p.character_name || ""), want)
          end)

        _ ->
          false
      end
  end
end

# Space matches within one replay so successive frames of the same held
# situation don't all get returned.
space_out = fn matches ->
  matches
  |> Enum.sort()
  |> Enum.reduce([], fn m, acc ->
    case acc do
      [prev | _] when m - prev < gap_frames -> acc
      _ -> [m | acc]
    end
  end)
  |> Enum.reverse()
end

matches =
  corpus_paths
  |> Enum.filter(char_ok?)
  |> Enum.reduce_while([], fn path, acc ->
    if length(acc) >= max_matches do
      {:halt, acc}
    else
      case ScenarioScan.load(path) do
        {:ok, %{frames: frames}} ->
          hits =
            frames
            |> Enum.filter(fn f -> f.frame >= min_frame and Coverage.bucket(f.p1, f.p2) == target_bucket end)
            |> Enum.map(& &1.frame)
            |> space_out.()
            |> Enum.map(fn frame -> %{"slp" => path, "frame" => frame} end)

          {:cont, acc ++ hits}
      {:error, _} ->
          {:cont, acc}
      end
    end
  end)
  |> Enum.take(max_matches)

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
out_path = opts[:out] || "logs/situations_#{ts}.json"
File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(%{"bucket" => target_bucket, "count" => length(matches), "situations" => matches}, pretty: true)
)

# Mark the source gap mined.
if mark_gap_id do
  ledger = GapLedger.load(gaps_path)
  ledger |> GapLedger.set_status(mark_gap_id, "mined") |> GapLedger.save(gaps_path)
  Output.puts("marked gap #{mark_gap_id} -> mined")
end

Output.puts("")
Output.puts("matches: #{length(matches)} across #{length(corpus_paths)} corpus replay(s)")
Output.success("situations -> #{out_path}")

if opts[:manifest_out] do
  entries =
    Enum.map(matches, fn m ->
      %{"slp" => m["slp"], "frame" => m["frame"], "type" => "retrieved", "note" => "bucket=#{target_bucket}"}
    end)

  File.mkdir_p!(Path.dirname(opts[:manifest_out]))
  File.write!(opts[:manifest_out], Jason.encode!(%{"entries" => entries}, pretty: true))
  Output.success("manifest (#{length(entries)}) -> #{opts[:manifest_out]}")
end
