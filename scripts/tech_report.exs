# Tech / comboed / defensive-movement report over .slp sets (2026-10-14).
# The instrument the s906 ep2 live look was missing: tech rate 0.26 vs 0.45
# and wavedashes 2.7 vs 4.6 /min had no readout behind them, and "doesn't
# convert" (punish_quality.exs) is a different question from "gets comboed"
# (this script: the same opening detector on the subject's OWN percent).
# ExPhil.Eval.TechStats does the counting; this groups files into sets.
#
#   mix run scripts/tech_report.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' --expert expert \
#     --set bot='eval_runs/<live>/2026-10-Mainline/*.slp' \
#     --set human:2='eval_runs/<live>/2026-10-Mainline/*.slp' \
#     --out eval_runs/<dir>/tech_report.md
#
# --set NAME[:PORT]=GLOB (repeatable; PORT overrides --port for that set —
#   the human in a live session is a third column for free) · --expert NAME
#   (per-file port by --expert-char, default Fox = 2; --expert-port N pins
#   it) · --port N (bot sets, default 1) · --expert-limit N (400) ·
#   --limit-files N · --concurrency N (8) · --out FILE.md (+ FILE.json)
alias ExPhil.{Data.Peppi, Eval.TechStats, Training.Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer, expert_char: :integer,
             expert_limit: :integer, limit_files: :integer, concurrency: :integer, out: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

sets =
  Keyword.get_values(opts, :set)
  |> Enum.map(fn s ->
    [n, g] = String.split(s, "=", parts: 2)
    case String.split(n, ":", parts: 2) do
      [name, p] -> {name, String.to_integer(p), g}
      [name] -> {name, nil, g}
    end
  end)
if sets == [], do: raise("--set NAME[:PORT]=GLOB required")

expert = opts[:expert] || "expert"
port = opts[:port] || 1
expert_char = opts[:expert_char] || 2
conc = opts[:concurrency] || 8

files_for = fn {name, _, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(&(File.stat!(&1).size < 150_000))
  limit = if name == expert, do: opts[:expert_limit] || 400, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

resolve_port = fn name, set_port, path ->
  cond do
    set_port -> {:ok, set_port}
    name != expert -> {:ok, port}
    opts[:expert_port] -> {:ok, opts[:expert_port]}
    true ->
      case Peppi.metadata(path) do
        {:ok, meta} ->
          case Enum.filter(meta.players, &(&1.character == expert_char)) do
            [%{port: p}] -> {:ok, p}
            _ -> :skip
          end
        _ -> :skip
      end
  end
end

scan = fn path, p ->
  o = if p == 1, do: 2, else: 1
  try do
    {:ok, replay} = Peppi.parse(path, player_port: p)
    replay
    |> Peppi.to_training_frames(player_port: p, opponent_port: o)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> Enum.map(&%{own: &1.game_state.players[p], opp: &1.game_state.players[o]})
    |> TechStats.from_game()
  rescue
    e -> Output.warning("skip #{Path.basename(path)}: #{Exception.message(e)}"); nil
  end
end

Output.banner("Tech report")
Output.config(Enum.map(sets, fn {n, p, g} = s -> {n, "#{length(files_for.(s))} files, port #{p || (if n == expert, do: "char #{expert_char}", else: port)} (#{g})"} end))

agg =
  Map.new(sets, fn {name, set_port, _} = s ->
    pairs =
      files_for.(s)
      |> Enum.flat_map(fn f -> case resolve_port.(name, set_port, f) do {:ok, p} -> [{f, p}]; _ -> [] end end)

    Output.puts("Scanning #{name}: #{length(pairs)} files")

    {counts, games} =
      pairs
      |> Task.async_stream(fn {f, p} -> scan.(f, p) end, max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce({TechStats.empty(), 0}, fn {:ok, c}, {acc, k} ->
        if c, do: {TechStats.merge(acc, c), k + 1}, else: {acc, k}
      end)

    {name, %{games: games, summary: TechStats.summarize(counts)}}
  end)

names = Enum.map(sets, &elem(&1, 0))
fmt = fn
  nil -> "–"
  v when is_float(v) -> :erlang.float_to_binary(v, decimals: 3)
  v -> "#{v}"
end

sections = [
  {"Tech (own knockdown landings)",
   ~w(tech_situations_per_min tech_rate wall_techs_per_min tech_in_place_share tech_toward_share tech_away_share
      tech_punished_share tech_cost_per miss_punished_share miss_cost_per miss_hit_on_ground_share
      miss_getup_neutral_share miss_getup_attack_share miss_getup_toward_share miss_getup_away_share miss_lying_frames_mean)},
  {"Tech-chase (opponent's misses)", ~w(opp_tech_situations_per_min opp_tech_rate chase_punished_share chase_cost_per)},
  {"Comboed (openings against the subject)",
   ~w(openings_taken_per_min hits_per_opening_taken dmg_per_opening_taken opening_taken_death_share comboed_3plus_share)},
  {"Defensive movement",
   ~w(shields_per_min dash_shields_per_min shield_time_share spotdodges_per_min rolls_per_min wavedashes_per_min wavelands_per_min)}
]

table = fn keys ->
  header = "| metric | " <> Enum.join(names, " | ") <> " |"
  sep = "|---|" <> String.duplicate("---:|", length(names))
  rows = Enum.map(keys, fn k -> "| #{k} | " <> Enum.map_join(names, " | ", &fmt.(agg[&1].summary[k])) <> " |" end)
  Enum.join([header, sep | rows], "\n")
end

report = """
# Tech report

Sets: #{Enum.map_join(names, " · ", fn n -> "#{n}: #{agg[n].games} games, #{agg[n].summary["minutes"]} min" end)}.
Tech situation = a landing from a knockdown: tech (in place / forward / backward; wall techs separate) or
a miss (TECH_MISS_UP/DOWN). Toward/away is relative to the opponent. Punished = any hit within 90 f;
cost = damage taken in those 90 f. Comboed = punish_quality's opening detector (quiet 30 f, gap 45 f,
kill 90 f) on the subject's OWN percent. Wavedash = airdodge ≤ 4 f into a landing from a jumpsquat;
waveland = the same without one. Dash→shield = shield started within 5 f of dashing/running.

#{Enum.map_join(sections, "\n\n", fn {title, keys} -> "## #{title}\n\n" <> table.(keys) end)}
"""

IO.puts("\n" <> report)
Enum.each(names, fn n -> Output.puts("RESULT #{n}: " <> Enum.map_join(Enum.sort(agg[n].summary), "  ", fn {k, v} -> "#{k} #{fmt.(v)}" end)) end)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  File.write!(Path.rootname(out) <> ".json", Jason.encode!(agg, pretty: true))
  Output.success("wrote #{out}")
end
