# Loop / taunt report — score the pathologies a HUMAN flags.
#
# The 08-28 human session flagged taunts, laser/shield-grab/pummel loops
# and dithering. `coach_report` scores none of them (it counts armed
# approaches, passivity, deaths, conversions), which is why two decode
# brackets came back null: the instrument was not pointed at the
# complaint. This script points at the complaint, and because every
# metric here is dense per game (button presses, frame-runs) rather than
# rare (conversions), it has the statistical power the approach-rate
# metric lacks.
#
# Usage:
#   mix run scripts/loop_report.exs eval_runs/0828_buttons_temp/btn*/r*.slp
#   mix run scripts/loop_report.exs --bot-port 2 --out logs/loops corpus/**/*.slp
#
# Games are GROUPED BY PARENT DIRECTORY (the sweep convention is
# <arm>/rN.slp), so an arm comparison falls out for free.
#
# Options:
#   --bot-port N   Bot's port (default 1, the probe/sweep convention)
#   --out DIR      Write report.md + report.json here (default: print only)
#   --per-game     Also print every game's row
#   --quiet

alias ExPhil.Interp.{ActionNames, LoopStats}
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    switches: [bot_port: :integer, out: :string, per_game: :boolean, quiet: :boolean],
    aliases: [o: :out, q: :quiet]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

slps =
  paths
  |> Enum.flat_map(fn p ->
    if String.contains?(p, "*"), do: Path.wildcard(p), else: [p]
  end)
  |> Enum.filter(&String.ends_with?(&1, ".slp"))
  |> Enum.uniq()
  |> Enum.sort()

if slps == [] do
  Output.error("No .slp files matched. Pass paths or globs.")
  System.halt(1)
end

bot_port = opts[:bot_port] || 1

Output.banner("Loop / Taunt Report")

Output.config([
  {"Replays", length(slps)},
  {"Bot port", bot_port},
  {"Groups", slps |> Enum.map(&(&1 |> Path.dirname() |> Path.basename())) |> Enum.uniq() |> length()}
])

# ---- score every game ------------------------------------------------------

reports =
  slps
  |> Enum.with_index(1)
  |> Enum.map(fn {path, i} ->
    if !opts[:quiet], do: Output.progress_bar(i, length(slps), label: "Scoring")

    case LoopStats.report(path, bot_port: bot_port) do
      %{} = r -> {path, r}
    end
  end)

unless opts[:quiet], do: Output.progress_done()

groups =
  reports
  |> Enum.group_by(fn {path, _} -> path |> Path.dirname() |> Path.basename() end)
  |> Enum.sort_by(fn {name, _} -> name end)

# ---- table -----------------------------------------------------------------

fmt = fn
  v when is_float(v) -> :erlang.float_to_binary(v, decimals: 2)
  v -> to_string(v)
end

cell = fn s -> "#{fmt.(s.mean)} [#{fmt.(s.min)}-#{fmt.(s.max)}]" end

headline = [
  {:taunts_per_min, "taunts/min"},
  {:dpad_per_min, "d_up press/min"},
  {:input_long_frac, "frozen-input frac"},
  {:action_long_frac, "held-action frac"},
  {:loops_per_min, "loops/min"},
  {:max_loop_repeats, "max loop reps"}
]

header =
  "| arm | n | " <> Enum.map_join(headline, " | ", fn {_, label} -> label end) <> " |"

sep = "|" <> String.duplicate("---|", 2 + length(headline))

rows =
  Enum.map(groups, fn {name, entries} ->
    agg = entries |> Enum.map(&elem(&1, 1)) |> LoopStats.aggregate()

    "| #{name} | #{agg.n} | " <>
      Enum.map_join(headline, " | ", fn {k, _} -> cell.(agg.stats[k]) end) <> " |"
  end)

table = Enum.join([header, sep | rows], "\n")

IO.puts("\n" <> table <> "\n")

Output.warning(
  "Mean [min-max]. Standing law: differences under 2x are UNRESOLVED, " <>
    "and a range that spans the other arm's mean is no difference at all."
)

# ---- per-game ---------------------------------------------------------------

per_game_lines =
  Enum.flat_map(groups, fn {name, entries} ->
    Enum.map(entries, fn {path, r} ->
      s = r.summary

      "| #{name} | #{Path.basename(path)} | #{fmt.(r.minutes)} | " <>
        "#{s.taunts_per_min |> fmt.()} | #{s.dpad_per_min |> fmt.()} | " <>
        "#{s.input_long_frac |> fmt.()} | #{s.action_long_frac |> fmt.()} | " <>
        "#{s.loops_per_min |> fmt.()} | #{s.max_loop_repeats} |"
    end)
  end)

per_game_table =
  Enum.join(
    [
      "| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |",
      "|---|---|---|---|---|---|---|---|---|" | per_game_lines
    ],
    "\n"
  )

if opts[:per_game], do: IO.puts("\n" <> per_game_table <> "\n")

# ---- top loop patterns ------------------------------------------------------

top_loops =
  reports
  |> Enum.flat_map(fn {_, r} -> r.cycles.episodes end)
  |> Enum.frequencies_by(& &1.pattern)
  |> Enum.sort_by(fn {_, c} -> -c end)
  |> Enum.take(10)

loop_lines =
  Enum.map(top_loops, fn {pattern, count} ->
    "| `#{ActionNames.cycle(pattern)}` | #{count} |"
  end)

loops_table =
  Enum.join(["| action cycle | episodes |", "|---|---|" | loop_lines], "\n")

IO.puts("\nMost repeated action cycles (action-state ids):\n")
IO.puts(loops_table <> "\n")

# ---- write ------------------------------------------------------------------

if out = opts[:out] do
  File.mkdir_p!(out)

  md = """
  # Loop / Taunt Report

  Bot port #{bot_port}, #{length(slps)} replays, #{length(groups)} groups.

  Metrics that score the human-flagged pathologies (taunts, loops,
  dithering) — none of which `coach_report` measures.

  ## By group

  #{table}

  Mean [min-max]. Differences under 2x are unresolved; a range spanning
  another arm's mean is no difference at all.

  ## Per game

  #{per_game_table}

  ## Most repeated action cycles

  #{loops_table}
  """

  File.write!(Path.join(out, "report.md"), md)

  json =
    Enum.map(groups, fn {name, entries} ->
      agg = entries |> Enum.map(&elem(&1, 1)) |> LoopStats.aggregate()

      %{
        arm: name,
        n: agg.n,
        stats: Map.new(agg.stats, fn {k, v} -> {k, Map.drop(v, [])} end),
        games:
          Enum.map(entries, fn {path, r} ->
            %{path: path, minutes: r.minutes, summary: r.summary}
          end)
      }
    end)

  File.write!(Path.join(out, "report.json"), Jason.encode!(json, pretty: true))
  Output.success("Wrote #{Path.join(out, "report.md")} and report.json")
end
