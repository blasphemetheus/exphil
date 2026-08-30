# E3 — expert pathology baselines (EVAL_DIRECTIONS).
#
# Pummels, taunts, laser-family specials, grabs, throws, d-up presses per
# game/minute in EXPERT play, so "too much" has a denominator. Per-file
# port resolution by character (erickfm masters sit on varying ports).
#
#   mix run scripts/expert_pathology.exs \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --char-id 2 --limit-files 400 --out eval_runs/0830_expert_pathology/RESULTS.md
#
# Options: --replays GLOB (required) · --char-id N (default 2 = Fox)
#          --port N (pin instead of auto) · --limit-files N (default 400)
#          --concurrency N (8) · --out FILE.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.LoopStats
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, char_id: :integer, port: :integer, limit_files: :integer,
             concurrency: :integer, out: :string]
  )

glob = opts[:replays] || raise "--replays required"
char_id = opts[:char_id] || 2
limit = opts[:limit_files] || 400
conc = opts[:concurrency] || 8

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit)
if files == [], do: raise("no replays matched #{glob}")

Output.banner("E3 — expert pathology baselines")
Output.config([{"Replays", "#{length(files)} files"}, {"Char", char_id}])

resolve_port = fn path ->
  if opts[:port] do
    {:ok, opts[:port]}
  else
    case Peppi.metadata(path) do
      {:ok, meta} ->
        case Enum.filter(meta.players, &(&1.character == char_id)) do
          [%{port: p}] -> {:ok, p}
          _ -> :skip
        end

      _ ->
        :skip
    end
  end
end

# entries into a state set (rising edges over the action stream)
count_entries = fn actions, set ->
  actions
  |> Enum.reduce({0, nil}, fn a, {n, prev} ->
    inside = MapSet.member?(set, a)
    was = prev != nil and MapSet.member?(set, prev)
    {if(inside and not was, do: n + 1, else: n), a}
  end)
  |> elem(0)
end

pummel = MapSet.new([217])
grab = MapSet.new([212, 214])
throws = MapSet.new([219, 220, 221, 222])
# char-specific special family (>= 341); for Fox the ground/air laser +
# side/up/down B all land here — reported as one "specials" family plus
# the 341..345 "neutral-B-ish" sub-band
specials = MapSet.new(341..400)
neutral_b = MapSet.new(341..345)

scan = fn path ->
  with {:ok, port} <- resolve_port.(path),
       {:ok, %{actions: actions, controllers: controllers, n: n}} when n > 1800 <-
         LoopStats.safe_load(path, bot_port: port) do
    t = LoopStats.taunt_stats(actions)
    d = LoopStats.dpad_stats(controllers)
    mins = n / 3600

    %{
      mins: mins,
      taunts: t.entries,
      dpad: d.presses,
      pummels: count_entries.(actions, pummel),
      grabs: count_entries.(actions, grab),
      throws: count_entries.(actions, throws),
      specials: count_entries.(actions, specials),
      neutral_b: count_entries.(actions, neutral_b)
    }
  else
    _ -> nil
  end
end

rows =
  files
  |> Task.async_stream(scan, max_concurrency: conc, timeout: :infinity, ordered: false)
  |> Stream.with_index(1)
  |> Enum.reduce([], fn {{:ok, r}, i}, acc ->
    if rem(i, 25) == 0, do: Output.progress_bar(i, length(files), label: "files")
    if r, do: [r | acc], else: acc
  end)

Output.progress_done()
if rows == [], do: raise("nothing scanned")

n = length(rows)
total_mins = rows |> Enum.map(& &1.mins) |> Enum.sum()

per_min = fn key -> Enum.sum(Enum.map(rows, &Map.fetch!(&1, key))) / total_mins end
per_game = fn key -> Enum.sum(Enum.map(rows, &Map.fetch!(&1, key))) / n end

pummels_per_grab =
  Enum.sum(Enum.map(rows, & &1.pummels)) / max(Enum.sum(Enum.map(rows, & &1.grabs)), 1)

throws_per_grab =
  Enum.sum(Enum.map(rows, & &1.throws)) / max(Enum.sum(Enum.map(rows, & &1.grabs)), 1)

f = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end

report = """
# E3 — expert pathology baselines

#{n} expert games (#{f.(total_mins)} minutes). Per-file port resolved by
character #{char_id}. These are the DENOMINATORS for the bot's pathology
metrics (loop_report): "too many taunts/pummels/lasers" is relative to these.

| metric | per game | per minute |
|---|---:|---:|
| taunts (executed, 264/265) | #{f.(per_game.(:taunts))} | #{f.(per_min.(:taunts))} |
| d-up presses | #{f.(per_game.(:dpad))} | #{f.(per_min.(:dpad))} |
| pummels (217) | #{f.(per_game.(:pummels))} | #{f.(per_min.(:pummels))} |
| grabs (212/214) | #{f.(per_game.(:grabs))} | #{f.(per_min.(:grabs))} |
| throws (219–222) | #{f.(per_game.(:throws))} | #{f.(per_min.(:throws))} |
| specials (char-specific ≥341) | #{f.(per_game.(:specials))} | #{f.(per_min.(:specials))} |
| neutral-B family (341–345) | #{f.(per_game.(:neutral_b))} | #{f.(per_min.(:neutral_b))} |

**Pummels per grab: #{f.(pummels_per_grab)} · throws per grab: #{f.(throws_per_grab)}**
(the bot's pummel-loop number divides by these).
"""

IO.puts(report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
