# Self-destruct scan — is a corpus poisoned with "run off the LEFT edge"?
#
# Bradley's hypothesis (2026-08-29 evening): the higher mode-of-N decodes
# run off to the LEFT repeatedly (never the right), and the training set
# may contain scripted/SD games where Fox walks off left several times per
# game — a mode-seeking decode would amplify exactly that.
#
# For every stock loss of the subject port:
#   SD?        no hitstun and no percent gain in the preceding --lookback
#              frames (default 90) — nobody hit them, they died on their own
#   side       which blastzone: :left / :right / :bottom / :top, from the
#              last observed position vs Situations.geometry(stage).blast
#   walk_off?  the last on-ground frame's action was walk/dash/run and the
#              player then left the stage past the edge (ran off, no jump)
#
# Reports per-corpus totals, left:right ratio, walk-off share, and the games
# with >= 2 / >= 3 / >= 4 left SDs (the "poisoned batch" signature), plus
# the top offender files. Scans the OPPONENT port too for symmetry.
#
# Usage:
#   mix run scripts/sd_scan.exs --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --port 1 --out eval_runs/0829_sd_scan/erickfm.md
#   mix run scripts/sd_scan.exs --replays 'eval_runs/0829_mode_of_n/mode16/r*.slp' --port 1
#
# Options: --port N (default 1) · --limit-files N · --lookback N (90)
#          --concurrency N (8) · --out FILE.md · --top N (15)
require Logger
Logger.configure(level: :warning)

alias ExPhil.Eval.ScenarioScan
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, port: :integer, limit_files: :integer, lookback: :integer,
             concurrency: :integer, out: :string, top: :integer]
  )

glob = opts[:replays] || raise("--replays GLOB required")
port = opts[:port] || 1
opp = if port == 1, do: 2, else: 1
lookback = opts[:lookback] || 90
top_n = opts[:top] || 15

files = glob |> Path.wildcard() |> Enum.sort()
files = if opts[:limit_files], do: Enum.take(files, opts[:limit_files]), else: files
if files == [], do: raise("no replays matched #{glob}")

Output.banner("Self-destruct scan")
Output.config([{"Replays", "#{length(files)} files (#{glob})"}, {"Subject port", port},
               {"Lookback", "#{lookback} frames"}])

# Mirrors FailureScan / ReplayStats.
hitstun = MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
# WALK_SLOW..WALK_FAST 15-17, TURNING 18, DASH 20, RUN 21, RUN_BRAKE 23
ground_move = MapSet.new([15, 16, 17, 18, 20, 21, 23])

pk = fn f, p -> Map.fetch!(f, p) end

scan_port = fn frames, arr, n, p, geo ->
  # stock losses of p
  losses =
    0..(n - 2)
    |> Enum.filter(fn i ->
      a = pk.(elem(arr, i), p).stock
      b = pk.(elem(arr, i + 1), p).stock
      is_integer(a) and is_integer(b) and b < a
    end)

  Enum.map(losses, fn i ->
    lo = max(0, i - lookback)
    window = for j <- lo..i, do: pk.(elem(arr, j), p)

    hit? =
      Enum.any?(window, fn q -> MapSet.member?(hitstun, q.action) end) or
        (List.first(window).percent < List.last(window).percent)

    # last frame with a sane position before the death animation (dying
    # actions are <= 0x0A; positions there are the blastzone crossing)
    last =
      Enum.reverse(window) |> Enum.find(fn q -> q.action > 0x0A end) || List.last(window)

    side =
      case geo.blast do
        {left, right, top, bottom} ->
          cond do
            last.y <= bottom + 25 and abs(last.x) < geo.edge -> :bottom
            last.x <= 0 and last.x <= left + 60 -> :left
            last.x > 0 and last.x >= right - 60 -> :right
            last.y >= top - 25 -> :top
            last.x < 0 -> :left_ish
            true -> :right_ish
          end

        nil ->
          if last.x < 0, do: :left_ish, else: :right_ish
      end

    # walk-off: last grounded frame in the window was a ground move and the
    # player left past the edge on that side without a jump
    grounded = Enum.reverse(window) |> Enum.find(fn q -> q.on_ground end)

    walk_off? =
      grounded != nil and MapSet.member?(ground_move, grounded.action) and
        abs(grounded.x) > geo.edge - 15 and side in [:left, :right, :left_ish, :right_ish]

    %{frame: elem(arr, i + 1).frame, sd: not hit?, side: side, walk_off: walk_off?}
  end)
end

scan_file = fn path ->
  case ScenarioScan.load(path) do
    {:ok, %{frames: frames, meta: meta}} when frames != [] ->
      arr = List.to_tuple(frames)
      n = tuple_size(arr)
      stage = meta && Map.get(meta, :stage)
      geo = Situations.geometry(stage)

      %{
        path: path,
        stage: stage,
        frames: n,
        subject: scan_port.(frames, arr, n, if(port == 1, do: :p1, else: :p2), geo),
        opponent: scan_port.(frames, arr, n, if(opp == 1, do: :p1, else: :p2), geo)
      }

    _ ->
      %{path: path, error: true}
  end
end

results =
  files
  |> Task.async_stream(scan_file, max_concurrency: opts[:concurrency] || 8, timeout: :infinity, ordered: false)
  |> Stream.with_index(1)
  |> Enum.map(fn {{:ok, r}, i} ->
    if rem(i, 50) == 0, do: Output.progress_bar(i, length(files), label: "replays")
    r
  end)

Output.progress_done()

ok = Enum.reject(results, & &1[:error])
errors = length(results) - length(ok)

summarize = fn key ->
  deaths = Enum.flat_map(ok, & &1[key])
  sds = Enum.filter(deaths, & &1.sd)
  by_side = Enum.frequencies_by(sds, & &1.side)
  walk = Enum.filter(sds, & &1.walk_off)
  walk_by_side = Enum.frequencies_by(walk, & &1.side)
  left = Map.get(by_side, :left, 0) + Map.get(by_side, :left_ish, 0)
  right = Map.get(by_side, :right, 0) + Map.get(by_side, :right_ish, 0)

  per_game_left =
    Enum.map(ok, fn r -> {r.path, Enum.count(r[key], &(&1.sd and &1.side in [:left, :left_ish]))} end)

  per_game_right =
    Enum.map(ok, fn r -> {r.path, Enum.count(r[key], &(&1.sd and &1.side in [:right, :right_ish]))} end)

  ge = fn list, k -> Enum.count(list, fn {_, c} -> c >= k end) end

  %{
    deaths: length(deaths), sds: length(sds), by_side: by_side, walk: length(walk),
    walk_by_side: walk_by_side, left: left, right: right,
    left_games: {ge.(per_game_left, 2), ge.(per_game_left, 3), ge.(per_game_left, 4)},
    right_games: {ge.(per_game_right, 2), ge.(per_game_right, 3), ge.(per_game_right, 4)},
    top_left: per_game_left |> Enum.sort_by(fn {_, c} -> -c end) |> Enum.take(top_n) |> Enum.filter(fn {_, c} -> c > 0 end)
  }
end

s = summarize.(:subject)
o = summarize.(:opponent)

fmt_side = fn m -> m |> Enum.sort_by(fn {k, _} -> to_string(k) end) |> Enum.map_join(", ", fn {k, v} -> "#{k} #{v}" end) end
pct = fn a, b -> if b > 0, do: Float.round(a / b * 100, 1), else: 0.0 end
{l2, l3, l4} = s.left_games
{r2, r3, r4} = s.right_games
{ol2, ol3, ol4} = o.left_games
{or2, or3, or4} = o.right_games
games = length(ok)

table = """
| | subject (port #{port}) | opponent (port #{opp}) |
|---|---|---|
| games scanned / errors | #{games} / #{errors} | same |
| deaths | #{s.deaths} | #{o.deaths} |
| self-destructs (no hit in #{lookback}f) | #{s.sds} (#{pct.(s.sds, s.deaths)}% of deaths) | #{o.sds} (#{pct.(o.sds, o.deaths)}%) |
| SD by side | #{fmt_side.(s.by_side)} | #{fmt_side.(o.by_side)} |
| **SD left : right** | **#{s.left} : #{s.right}** | #{o.left} : #{o.right} |
| walk-off SDs (ran/walked off, no jump) | #{s.walk} — #{fmt_side.(s.walk_by_side)} | #{o.walk} — #{fmt_side.(o.walk_by_side)} |
| games with ≥2 / ≥3 / ≥4 LEFT SDs | #{l2} / #{l3} / #{l4} | #{ol2} / #{ol3} / #{ol4} |
| games with ≥2 / ≥3 / ≥4 RIGHT SDs | #{r2} / #{r3} / #{r4} | #{or2} / #{or3} / #{or4} |
"""

offenders =
  s.top_left
  |> Enum.map_join("\n", fn {p, c} -> "| #{Path.basename(p)} | #{c} |" end)

report = """
# Self-destruct scan — #{glob}

#{length(files)} files, subject port #{port}, lookback #{lookback} frames.
A "self-destruct" is a stock loss with no hitstun and no percent gain in the
lookback window. Side is the blastzone crossed. "left_ish/right_ish" = died
off-screen on that half without a clean blastzone read (stage geometry
unknown or ambiguous).

#{table}

## Games with the most LEFT self-destructs (subject)

| file | left SDs |
|---|---|
#{offenders}

Read: a poisoned batch shows up as many games with ≥3 left SDs AND a
left:right ratio far from 1 for the subject only. A ratio near 1 with few
repeat-offender games means no directional poison in this corpus.
"""

IO.puts("\n" <> table)
IO.puts("Top LEFT-SD games (subject):\n" <> offenders)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
