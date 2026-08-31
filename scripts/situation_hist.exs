# Situation -> next-option histograms, bot vs expert (EVAL_DIRECTIONS A1),
# with per-situation distribution distance (B2).
#
# For every option EVENT (ExPhil.Options: dash, wavedash, grab, throw, jab,
# aerial, special, shield_on, spotdodge, ...) fired by the subject port,
# record the SITUATION labels active on the frame before it (ExPhil.
# Situations — the decision context). Aggregate per set -> label -> option.
#
# Output, per situation label: one table whose rows are options and whose
# columns are the sets' shares (%), with n events and events/min in that
# situation per set, and for every non-expert set the total-variation
# distance TV and KL(set || expert) from the expert's histogram. Then a
# summary matrix label x set -> TV. A decode that NARROWS the distribution
# (argmax, mode-of-N) shows up as high TV even when its pass@1 is high —
# this is the offline decode ranker that replaces master-match (L9).
#
#   mix run scripts/situation_hist.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' --expert expert \
#     --set ep10_cpu='eval_runs/0829_mode_of_n/base/r*.slp' \
#     --set mode16_cpu='eval_runs/0829_mode_of_n/mode16/r*.slp' \
#     --expert-limit 600 --out eval_runs/0829_situation_hist/RESULTS.md
#
# Options:
#   --set NAME=GLOB     repeatable; the corpora to compare
#   --expert NAME       which set is the baseline (default: "expert")
#   --port N            subject port for non-expert sets (default 1)
#   --expert-port N     subject port for the expert set (default 1)
#   --expert-limit N    sample N expert files (sorted, first N; default 600)
#   --limit-files N     cap per non-expert set
#   --labels a,b,c      situation labels to report (default: the decision list)
#   --min-n N           skip a (label, set) cell with fewer events (default 20)
#   --top N             options per table (default 12)
#   --concurrency N     (default 8)
#   --out FILE.md       write the report (also prints)
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Options
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer,
             expert_char: :integer, expert_limit: :integer, limit_files: :integer,
             labels: :string, min_n: :integer,
             top: :integer, concurrency: :integer, out: :string]
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
expert_char = opts[:expert_char]
min_n = opts[:min_n] || 20

# --expert-char N: resolve the expert's port PER FILE by character (the
# corpus has fox on varying ports — E1, eval_runs/0830_corpus_mix; a fixed
# port mixes ~43% opponent characters into the baseline). Files where the
# character is absent or ambiguous (dittos) are skipped.
port_for = fn name, path ->
  cond do
    name != expert -> port
    expert_char == nil -> expert_port
    true ->
      case Peppi.metadata(path) do
        {:ok, meta} ->
          case Enum.filter(meta.players, &(&1.character == expert_char)) do
            [%{port: pp}] -> pp
            _ -> nil
          end

        _ -> nil
      end
  end
end
top_n = opts[:top] || 12
conc = opts[:concurrency] || 8

default_labels = ~w(neutral approach retreat advantage disadvantage conversion_open combo_active
  juggle tech_chase ledge_trap edgeguard shield_pressure_ours pummel_throw_decision
  shield_pressure_theirs being_edgeguarded recovery_low recovery_high cornered edge_danger
  offstage ledge_hang respawn_invincible post_kill_neutral percent_lead percent_deficit)a

labels =
  case opts[:labels] do
    nil -> default_labels
    s -> s |> String.split(",") |> Enum.map(&String.to_atom(String.trim(&1)))
  end

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort()
  # drop CSS-restart stubs (~100 KB) from bot sets
  fs = Enum.reject(fs, fn f -> File.stat!(f).size < 150_000 end)
  limit = if name == expert, do: opts[:expert_limit] || 600, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

Output.banner("Situation histograms (A1) + distribution distance (B2)")

Output.config(
  Enum.map(sets, fn {n, g} = s -> {n, "#{length(files_for.(s))} files (#{g})"} end) ++
    [{"Expert set", expert}, {"Labels", length(labels)}, {"min n", min_n}]
)

# One file -> %{counts: %{{label, option} => n}, sit_frames: %{label => frames}, events: n}
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
        sits = Situations.label_states(states, p, as: :set)
        sit_arr = List.to_tuple(sits)
        events = Options.events(states, p)

        sit_frames =
          Enum.reduce(sits, %{}, fn set, acc ->
            Enum.reduce(set, acc, fn l, a -> Map.update(a, l, 1, &(&1 + 1)) end)
          end)

        counts =
          Enum.reduce(events, %{}, fn %{index: i, option: o}, acc ->
            ctx = elem(sit_arr, max(i - 1, 0))

            Enum.reduce(MapSet.put(ctx, :_any), acc, fn l, a ->
              Map.update(a, {l, o}, 1, &(&1 + 1))
            end)
          end)

        %{counts: counts, sit_frames: Map.put(sit_frames, :_any, length(states)), events: length(events)}
      end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

merge = fn a, b ->
  %{
    counts: Map.merge(a.counts, b.counts, fn _, x, y -> x + y end),
    sit_frames: Map.merge(a.sit_frames, b.sit_frames, fn _, x, y -> x + y end),
    events: a.events + b.events,
    files: a.files + 1
  }
end

empty = %{counts: %{}, sit_frames: %{}, events: 0, files: 0}

agg =
  Map.new(sets, fn {name, _} = s ->
    files = files_for.(s)

    pairs =
      files |> Enum.map(&{&1, port_for.(name, &1)}) |> Enum.reject(fn {_, p} -> is_nil(p) end)

    Output.puts("Scanning #{name}: #{length(pairs)}/#{length(files)} files" <>
      if(name == expert and expert_char, do: " (per-file char #{expert_char})", else: ""))

    total =
      pairs
      |> Task.async_stream(fn {f, p} -> scan_file.(f, p) end,
        max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Stream.with_index(1)
      |> Enum.reduce(empty, fn {{:ok, r}, i}, acc ->
        if rem(i, 25) == 0, do: Output.progress_bar(i, length(pairs), label: name)
        if r, do: merge.(acc, r), else: acc
      end)

    Output.progress_done()
    {name, total}
  end)

set_names = Enum.map(sets, fn {n, _} -> n end)
others = Enum.reject(set_names, &(&1 == expert))

hist = fn name, label ->
  agg[name].counts
  |> Enum.filter(fn {{l, _}, _} -> l == label end)
  |> Map.new(fn {{_, o}, n} -> {o, n} end)
end

shares = fn h ->
  t = h |> Map.values() |> Enum.sum()
  if t == 0, do: %{}, else: Map.new(h, fn {o, n} -> {o, n / t} end)
end

tv = fn p, q ->
  keys = MapSet.union(MapSet.new(Map.keys(p)), MapSet.new(Map.keys(q)))
  0.5 * Enum.sum(Enum.map(keys, fn k -> abs(Map.get(p, k, 0.0) - Map.get(q, k, 0.0)) end))
end

# KL(p || q) with additive smoothing so unseen expert options don't blow up
kl = fn p, q ->
  keys = MapSet.union(MapSet.new(Map.keys(p)), MapSet.new(Map.keys(q)))
  k = MapSet.size(keys)
  eps = 1.0e-3

  Enum.sum(
    Enum.map(keys, fn key ->
      pi = (Map.get(p, key, 0.0) + eps) / (1 + eps * k)
      qi = (Map.get(q, key, 0.0) + eps) / (1 + eps * k)
      pi * :math.log(pi / qi)
    end)
  )
end

f1 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 1) end
f2 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end

per_min = fn name, label ->
  h = hist.(name, label)
  n = h |> Map.values() |> Enum.sum()
  fr = Map.get(agg[name].sit_frames, label, 0)
  if fr > 0, do: n / (fr / 3600), else: 0.0
end

section = fn label ->
  hs = Map.new(set_names, fn n -> {n, hist.(n, label)} end)
  ns = Map.new(set_names, fn n -> {n, hs[n] |> Map.values() |> Enum.sum()} end)
  sh = Map.new(set_names, fn n -> {n, shares.(hs[n])} end)

  if ns[expert] < min_n do
    "### `#{label}` — expert n=#{ns[expert]} < #{min_n}, skipped\n"
  else
    options =
      sh[expert]
      |> Enum.sort_by(fn {_, s} -> -s end)
      |> Enum.map(&elem(&1, 0))

    extra =
      others
      |> Enum.flat_map(fn n -> sh[n] |> Enum.sort_by(fn {_, s} -> -s end) |> Enum.take(3) |> Enum.map(&elem(&1, 0)) end)

    options = (options ++ extra) |> Enum.uniq() |> Enum.take(top_n + 3)

    header =
      "| option | " <>
        Enum.map_join(set_names, " | ", fn n ->
          tag = if ns[n] < min_n, do: " ⚠", else: ""
          "#{n} (n=#{ns[n]}#{tag})"
        end) <> " |"

    sep = "|---|" <> String.duplicate("---:|", length(set_names))

    rows =
      Enum.map_join(options, "\n", fn o ->
        "| #{o} | " <> Enum.map_join(set_names, " | ", fn n -> f1.(Map.get(sh[n], o, 0.0) * 100) end) <> " |"
      end)

    rate_row =
      "| *options / min in situation* | " <> Enum.map_join(set_names, " | ", fn n -> f1.(per_min.(n, label)) end) <> " |"

    dist_rows =
      "| **TV vs expert** | – | " <>
        Enum.map_join(others, " | ", fn n -> if ns[n] < min_n, do: "–", else: f2.(tv.(sh[n], sh[expert])) end) <>
        " |\n| **KL(set‖expert)** | – | " <>
        Enum.map_join(others, " | ", fn n -> if ns[n] < min_n, do: "–", else: f2.(kl.(sh[n], sh[expert])) end) <> " |"

    "### `#{label}`\n\n#{header}\n#{sep}\n#{rows}\n#{rate_row}\n#{dist_rows}\n"
  end
end

sections = Enum.map_join([:_any | labels], "\n", section)

# Summary matrix: label x set -> TV (only cells with n >= min_n on both sides)
summary_header = "| situation | expert n | " <> Enum.map_join(others, " | ", &"#{&1} TV (n)") <> " |"
summary_sep = "|---|---:|" <> String.duplicate("---:|", length(others))

summary_rows =
  Enum.map_join([:_any | labels], "\n", fn label ->
    he = shares.(hist.(expert, label))
    ne = hist.(expert, label) |> Map.values() |> Enum.sum()

    cells =
      Enum.map_join(others, " | ", fn n ->
        h = hist.(n, label)
        nn = h |> Map.values() |> Enum.sum()
        if nn < min_n or ne < min_n, do: "– (#{nn})", else: "#{f2.(tv.(shares.(h), he))} (#{nn})"
      end)

    "| #{label} | #{ne} | #{cells} |"
  end)

mean_tv =
  Enum.map_join(others, " · ", fn n ->
    vals =
      labels
      |> Enum.map(fn label ->
        h = hist.(n, label)
        nn = h |> Map.values() |> Enum.sum()
        ne = hist.(expert, label) |> Map.values() |> Enum.sum()
        if nn >= min_n and ne >= min_n, do: tv.(shares.(h), shares.(hist.(expert, label))), else: nil
      end)
      |> Enum.reject(&is_nil/1)

    if vals == [], do: "#{n}: –", else: "#{n}: #{f2.(Enum.sum(vals) / length(vals))} over #{length(vals)} situations"
  end)

files_line = Enum.map_join(set_names, " · ", fn n -> "#{n}: #{agg[n].files} files, #{agg[n].events} events" end)

report = """
# Situation → next-option histograms (A1) and distribution distance (B2)

Sets: #{files_line}.
Subject port: expert #{expert_port}, others #{port}. Situation = labels active on the
frame BEFORE the option fired (`Situations`); option = `Options.events`. Shares are
% of that set's options in that situation. TV = total-variation distance to the
expert's histogram (0 = identical, 1 = disjoint); KL is smoothed (ε=1e-3).
Cells with n < #{min_n} are marked ⚠ / skipped. `_any` = all frames.

## Summary — TV distance to expert, per situation

#{summary_header}
#{summary_sep}
#{summary_rows}

**Mean TV over reported situations:** #{mean_tv}

## Per situation

#{sections}
"""

IO.puts(report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
