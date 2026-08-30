# B1 — action-FAMILY match per situation (EVAL_DIRECTIONS).
#
# Softer than pass@1: in each situation, did the bot pick from the same
# CATEGORY of option (movement / ground_attack / aerial / grab / shield /
# special / defense / ledge) as the expert distribution? Collapses
# `Options.events` into families and reports family shares + TV per
# situation — robust where the exact-option histogram (A1/B2) is sparse.
#
#   mix run scripts/action_family_match.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' --expert expert \
#     --set B1='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp' \
#     --expert-limit 400 --out eval_runs/0830_family_match/RESULTS.md
#
# Options: same shape as situation_hist.exs (--set NAME=GLOB repeatable,
#   --expert NAME, --port, --expert-port, --expert-limit, --limit-files,
#   --min-n, --concurrency, --out)
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Options
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer,
             expert_limit: :integer, limit_files: :integer, min_n: :integer,
             concurrency: :integer, out: :string]
  )

sets =
  Keyword.get_values(opts, :set)
  |> Enum.map(fn s ->
    [name, glob] = String.split(s, "=", parts: 2)
    {name, glob}
  end)

if sets == [], do: raise("at least one --set NAME=GLOB")
expert = opts[:expert] || "expert"
unless Enum.any?(sets, fn {n, _} -> n == expert end), do: raise("--expert #{expert} is not a --set")

port = opts[:port] || 1
expert_port = opts[:expert_port] || 1
min_n = opts[:min_n] || 20
conc = opts[:concurrency] || 8

family = fn option ->
  case option do
    o when o in [:dash, :dashdance, :wavedash, :waveland, :double_jump, :airdodge,
                 :roll_forward, :roll_backward, :spotdodge] -> :movement
    :shield_on -> :shield
    o when o in [:grab, :throw] -> :grab
    o when o in [:jab, :dash_attack, :tilt, :smash] -> :ground_attack
    :aerial -> :aerial
    :special -> :special
    o when o in [:tech_in_place, :tech_roll, :missed_tech, :getup_attack, :getup_stand] -> :defense
    o when o in [:ledge_getup, :ledge_attack, :ledge_roll, :ledge_jump] -> :ledge
    _ -> :other
  end
end

labels = ~w(neutral approach advantage disadvantage conversion_open combo_active juggle
  tech_chase ledge_trap edgeguard shield_pressure_ours pummel_throw_decision
  shield_pressure_theirs being_edgeguarded recovery_low recovery_high cornered edge_danger
  offstage)a

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(fn f -> File.stat!(f).size < 150_000 end)
  limit = if name == expert, do: opts[:expert_limit] || 400, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

Output.banner("B1 — action-family match per situation")

scan_file = fn path, p ->
  opp = if p == 1, do: 2, else: 1

  try do
    with {:ok, replay} <- Peppi.parse(path, player_port: p) do
      states =
        replay
        |> Peppi.to_training_frames(player_port: p, opponent_port: opp)
        |> Enum.reject(&(&1.game_state.frame < 0))
        |> Enum.map(& &1.game_state)

      if length(states) < 300 do
        nil
      else
        sits = Situations.label_states(states, p, as: :set) |> List.to_tuple()
        events = Options.events(states, p)

        Enum.reduce(events, %{}, fn %{index: i, option: o}, acc ->
          fam = family.(o)
          ctx = elem(sits, max(i - 1, 0))

          Enum.reduce(MapSet.put(ctx, :_any), acc, fn l, a ->
            Map.update(a, {l, fam}, 1, &(&1 + 1))
          end)
        end)
      end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

agg =
  Map.new(sets, fn {name, _} = s ->
    files = files_for.(s)
    p = if name == expert, do: expert_port, else: port
    Output.puts("Scanning #{name}: #{length(files)} files (port #{p})")

    counts =
      files
      |> Task.async_stream(&scan_file.(&1, p), max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce(%{}, fn {:ok, r}, acc ->
        if r, do: Map.merge(acc, r, fn _, x, y -> x + y end), else: acc
      end)

    {name, counts}
  end)

set_names = Enum.map(sets, fn {n, _} -> n end)
others = Enum.reject(set_names, &(&1 == expert))
families = ~w(movement ground_attack aerial grab shield special defense ledge other)a

hist = fn name, label ->
  Map.new(families, fn fam -> {fam, Map.get(agg[name], {label, fam}, 0)} end)
end

shares = fn h ->
  t = h |> Map.values() |> Enum.sum()
  if t == 0, do: %{}, else: Map.new(h, fn {o, n} -> {o, n / t} end)
end

tv = fn p, q ->
  0.5 * Enum.sum(Enum.map(families, fn k -> abs(Map.get(p, k, 0.0) - Map.get(q, k, 0.0)) end))
end

f1 = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
f2 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end

section = fn label ->
  hs = Map.new(set_names, fn nm -> {nm, hist.(nm, label)} end)
  ns = Map.new(set_names, fn nm -> {nm, hs[nm] |> Map.values() |> Enum.sum()} end)
  sh = Map.new(set_names, fn nm -> {nm, shares.(hs[nm])} end)

  if ns[expert] < min_n do
    "### `#{label}` — expert n=#{ns[expert]} < #{min_n}, skipped\n"
  else
    header = "| family | " <> Enum.map_join(set_names, " | ", &"#{&1} (n=#{ns[&1]})") <> " |"
    sep = "|---|" <> String.duplicate("---:|", length(set_names))

    rows =
      Enum.map_join(families, "\n", fn fam ->
        "| #{fam} | " <> Enum.map_join(set_names, " | ", fn nm -> f1.(Map.get(sh[nm], fam, 0.0)) end) <> " |"
      end)

    dist =
      "| **family TV vs expert** | – | " <>
        Enum.map_join(others, " | ", fn nm -> if ns[nm] < min_n, do: "–", else: f2.(tv.(sh[nm], sh[expert])) end) <> " |"

    "### `#{label}`\n\n#{header}\n#{sep}\n#{rows}\n#{dist}\n"
  end
end

summary_rows =
  Enum.map_join([:_any | labels], "\n", fn label ->
    he = shares.(hist.(expert, label))
    ne = hist.(expert, label) |> Map.values() |> Enum.sum()

    cells =
      Enum.map_join(others, " | ", fn nm ->
        h = hist.(nm, label)
        nn = h |> Map.values() |> Enum.sum()
        if nn < min_n or ne < min_n, do: "– (#{nn})", else: "#{f2.(tv.(shares.(h), he))} (#{nn})"
      end)

    "| #{label} | #{ne} | #{cells} |"
  end)

report = """
# B1 — action-family shares per situation (family TV vs expert)

Families collapse `Options.events`: movement (dash/wd/jump/dodge/roll),
ground_attack (jab/da/tilt/smash), aerial, grab (grab+throw), shield,
special, defense (tech/getup), ledge. TV over 9 families — the softer
pass@1: 0 = same category mix as the expert.

## Summary — family TV per situation

| situation | expert n | #{Enum.map_join(others, " | ", &"#{&1} TV (n)")} |
|---|---:|#{String.duplicate("---:|", length(others))}
#{summary_rows}

## Per situation

#{Enum.map_join([:_any | labels], "\n", section)}
"""

IO.puts(report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
