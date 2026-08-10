# Per-situation option statistics — the v0 "what's good" knowledge model
# (coach roadmap #4b).
#
# For every option EVENT (ExPhil.Options) in a replay corpus, record the
# SITUATION labels active when it was chosen (ExPhil.Situations, at the
# frame before initiation — the decision context) and its short-horizon
# OUTCOME (damage dealt/taken, stocks taken/lost over --horizon frames).
# Aggregated per matchup -> situation label -> option, this answers the
# coach's core query in words: "in edge_danger vs Falco, players chose
# dash 41%, wavedash 20%, ... and dash averaged +4.2 damage".
#
# Percent-reset handling: damage sums POSITIVE deltas only (a stock loss
# resets percent to 0; counting that as negative damage would corrupt
# every kill-adjacent option's outcome).
#
#   mix run scripts/situation_stats.exs \
#     --replays replays/fox_il_v1 --character fox \
#     --max-files 200 --out eval_runs/0811_fox_stats \
#     2>&1 | tee eval_runs/0811_fox_stats.log
#
# Options:
#   --replays PATH     Replay dir (recursive *.slp) [required]
#   --character NAME   Subject character; per-file port detection
#                      (files without it are skipped). Omit = port 1.
#   --max-files N      Limit files
#   --horizon N        Outcome window in frames [120]
#   --out DIR          Output dir: stats.json + summary printed [required]
#   --quiet            Errors/summary only

if "--quiet" in System.argv(), do: Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Options
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      replays: :string,
      character: :string,
      max_files: :integer,
      horizon: :integer,
      out: :string,
      quiet: :boolean
    ]
  )

replay_dir = Path.expand(opts[:replays] || raise("--replays is required"))
out_dir = opts[:out] || raise("--out is required")
horizon = opts[:horizon] || 120
character = opts[:character]

# External (CSS) ids — same table as build_corpus.exs
external_ids = %{
  "captainfalcon" => 0, "falcon" => 0, "dk" => 1, "donkeykong" => 1,
  "fox" => 2, "gameandwatch" => 3, "gnw" => 3, "kirby" => 4,
  "bowser" => 5, "link" => 6, "luigi" => 7, "mario" => 8,
  "marth" => 9, "mewtwo" => 10, "ness" => 11, "peach" => 12,
  "pikachu" => 13, "iceclimbers" => 14, "ics" => 14,
  "jigglypuff" => 15, "puff" => 15, "samus" => 16, "yoshi" => 17,
  "zelda" => 18, "sheik" => 19, "falco" => 20, "younglink" => 21,
  "doc" => 22, "drmario" => 22, "roy" => 23, "pichu" => 24,
  "ganondorf" => 25, "ganon" => 25
}

id_to_name =
  external_ids
  |> Enum.group_by(fn {_k, v} -> v end, fn {k, _v} -> k end)
  |> Map.new(fn {id, names} -> {id, Enum.max_by(names, &String.length/1)} end)

want_id =
  if character do
    key = character |> String.downcase() |> String.replace(~r/[^a-z0-9]/, "")
    external_ids[key] || raise "unknown character #{character}"
  end

pick_ports = fn meta ->
  ports = Enum.map(meta.players, & &1.port)

  cond do
    want_id == nil -> {1, 2}
    length(ports) != 2 -> nil
    true ->
      case Enum.find(meta.players, &(&1.character == want_id)) do
        nil -> nil
        p -> {p.port, Enum.find(ports, &(&1 != p.port))}
      end
  end
end

files = Path.wildcard(Path.join(replay_dir, "**/*.slp")) |> Enum.sort()
files = if opts[:max_files], do: Enum.take(files, opts[:max_files]), else: files
if files == [], do: raise("no .slp files under #{replay_dir}")

File.mkdir_p!(out_dir)
Output.banner("Situation-option statistics")
Output.config([
  {"Replays", "#{replay_dir} (#{length(files)} files)"},
  {"Character", character || "port 1"},
  {"Horizon", "#{horizon} frames"},
  {"Output", out_dir}
])

percent_bucket = fn pct ->
  cond do
    pct < 40 -> "0-39"
    pct < 80 -> "40-79"
    pct < 120 -> "80-119"
    true -> "120+"
  end
end

# Positive-delta damage over [i, i+horizon] (resets ignored)
outcome = fn players_arr, i, n ->
  last = min(i + horizon, n - 1)

  Enum.reduce((i + 1)..last//1, %{dealt: 0.0, taken: 0.0, takes: 0, losses: 0}, fn j, acc ->
    {own_p, opp_p} = elem(players_arr, j)
    {own_q, opp_q} = elem(players_arr, j - 1)

    %{
      acc
      | dealt: acc.dealt + max((opp_p.percent || 0.0) - (opp_q.percent || 0.0), 0.0),
        taken: acc.taken + max((own_p.percent || 0.0) - (own_q.percent || 0.0), 0.0),
        takes: acc.takes + if((opp_p.stock || 0) < (opp_q.stock || 0), do: 1, else: 0),
        losses: acc.losses + if((own_p.stock || 0) < (own_q.stock || 0), do: 1, else: 0)
    }
  end)
end

merge_cell = fn cell, out ->
  cell = cell || %{n: 0, dealt: 0.0, taken: 0.0, takes: 0, losses: 0}

  %{
    n: cell.n + 1,
    dealt: cell.dealt + out.dealt,
    taken: cell.taken + out.taken,
    takes: cell.takes + out.takes,
    losses: cell.losses + out.losses
  }
end

{stats, n_files, n_events} =
  files
  |> Enum.with_index(1)
  |> Enum.reduce({%{}, 0, 0}, fn {path, idx}, {stats, nf, ne} ->
    if rem(idx, 25) == 0, do: IO.write(:stderr, "\r  [#{idx}/#{length(files)}] #{ne} events\e[K")

    try do
      with {:ok, meta} <- Peppi.metadata(path),
           {port, opp_port} when is_integer(port) <- pick_ports.(meta) || :no_char,
           {:ok, replay} <- Peppi.parse(path, player_port: port) do
        states =
          replay
          |> Peppi.to_training_frames(player_port: port, opponent_port: opp_port)
          |> Enum.reject(&(&1.game_state.frame < 0))
          |> Enum.map(& &1.game_state)

        n = length(states)

        if n < 300 do
          {stats, nf, ne}
        else
          own_char = Enum.find(meta.players, &(&1.port == port)).character
          opp_char = Enum.find(meta.players, &(&1.port == opp_port)).character

          matchup =
            "#{id_to_name[own_char] || own_char}_vs_#{id_to_name[opp_char] || opp_char}"

          situations = Situations.label_states(states, port, as: :set)
          sit_arr = List.to_tuple(situations)

          players_arr =
            states
            |> Enum.map(fn gs -> {gs.players[port], gs.players[opp_port]} end)
            |> List.to_tuple()

          events = Options.events(states, port)

          stats =
            Enum.reduce(events, stats, fn %{index: i, option: option}, acc ->
              {own, _opp} = elem(players_arr, i)
              context = elem(sit_arr, max(i - 1, 0))
              out = outcome.(players_arr, i, n)
              bucket = percent_bucket.(own.percent || 0.0)

              keys =
                for label <- MapSet.to_list(context) do
                  {matchup, to_string(label), to_string(option)}
                end ++ [{matchup, "_any", to_string(option)}]

              acc =
                Enum.reduce(keys, acc, fn key, a ->
                  Map.update(a, key, merge_cell.(nil, out), &merge_cell.(&1, out))
                end)

              Map.update(
                acc,
                {matchup, "_pct_#{bucket}", to_string(option)},
                merge_cell.(nil, out),
                &merge_cell.(&1, out)
              )
            end)

          {stats, nf + 1, ne + length(events)}
        end
      else
        _ -> {stats, nf, ne}
      end
    rescue
      _ -> {stats, nf, ne}
    end
  end)

IO.write(:stderr, "\r\e[K")

# Serialize: matchup -> label -> option -> summary
tree =
  stats
  |> Enum.reduce(%{}, fn {{matchup, label, option}, cell}, acc ->
    summary = %{
      n: cell.n,
      dmg_dealt_avg: Float.round(cell.dealt / cell.n, 2),
      dmg_taken_avg: Float.round(cell.taken / cell.n, 2),
      dmg_delta_avg: Float.round((cell.dealt - cell.taken) / cell.n, 2),
      stock_take_rate: Float.round(cell.takes / cell.n, 4),
      stock_loss_rate: Float.round(cell.losses / cell.n, 4)
    }

    put_in(acc, [Access.key(matchup, %{}), Access.key(label, %{}), option], summary)
  end)

File.write!(
  Path.join(out_dir, "stats.json"),
  Jason.encode!(
    %{
      generated_from: replay_dir,
      character: character,
      files_used: n_files,
      events: n_events,
      horizon: horizon,
      tree: tree
    },
    pretty: true
  )
)

Output.success("#{n_events} events from #{n_files} files -> #{Path.join(out_dir, "stats.json")}")

# Console summary: option mix + outcomes for the evidence-backed labels
interesting = ~w(_any edge_danger conversion_open being_edgeguarded ledge_option_pending edgeguard cornered)

for {matchup, labels} <- Enum.sort(tree) do
  Output.puts("")
  Output.puts("== #{matchup}")

  for label <- interesting, cells = labels[label], cells != nil do
    total = cells |> Enum.map(fn {_o, s} -> s.n end) |> Enum.sum()

    top =
      cells
      |> Enum.sort_by(fn {_o, s} -> -s.n end)
      |> Enum.take(8)
      |> Enum.map_join("  ", fn {o, s} ->
        "#{o} #{Float.round(100.0 * s.n / total, 0) |> trunc()}% (Δ#{s.dmg_delta_avg})"
      end)

    Output.puts("  #{label} (#{total}): #{top}")
  end
end
