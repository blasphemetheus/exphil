# F2 — punishable-commitment rate (EVAL_DIRECTIONS F2, Bradley 08-31):
# "it puts itself into whiff punish." The MECHANISM behind C1's whiff-punish
# outcomes: how often does the subject initiate a committal (laggy) option
# while the opponent is in threat range AND actionable — and how often does
# that commitment get punished?
#
# Committal options (ExPhil.Options events): smash, dash_attack, grab,
# spotdodge, roll_forward, roll_backward.
# In-threat at the event frame: opponent NOT in hitstun, |dx| < 30, |dy| < 20.
# Punished: subject enters hitstun within 60 f of the commitment.
#
#   mix run scripts/commitment_scorecard.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' --expert expert \
#     --expert-char 2 --expert-limit 200 \
#     --set AR_human='eval_runs/0831_livelook_v11ar/2026-08-Mainline/*.slp' \
#     --out eval_runs/0831_session_score/commitment_scorecard.md
#
# Options mirror edge_scorecard (--set/--expert/--expert-char/--port/...).
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
             range_x: :integer, range_y: :integer, punish_window: :integer,
             concurrency: :integer, out: :string]
  )

sets = Keyword.get_values(opts, :set) |> Enum.map(fn s -> [n, g] = String.split(s, "=", parts: 2); {n, g} end)
if sets == [], do: raise("--set NAME=GLOB required")
expert = opts[:expert] || "expert"
port = opts[:port] || 1
expert_port = opts[:expert_port] || 1
range_x = opts[:range_x] || 30
range_y = opts[:range_y] || 20
pwin = opts[:punish_window] || 60
conc = opts[:concurrency] || 12

committal = [:smash, :dash_attack, :grab, :spotdodge, :roll_forward, :roll_backward]

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(&(File.stat!(&1).size < 150_000))
  limit = if name == expert, do: opts[:expert_limit] || 200, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

Output.banner("Punishable-commitment scorecard (F2)")
Output.config(Enum.map(sets, fn {n, g} = s -> {n, "#{length(files_for.(s))} files (#{g})"} end) ++
  [{"Threat range", "|dx|<#{range_x}, |dy|<#{range_y}"}, {"Punish window", "#{pwin} f"}])

scan = fn path, p ->
  o = if p == 1, do: 2, else: 1

  try do
    with {:ok, replay} <- Peppi.parse(path, player_port: p) do
      states =
        replay
        |> Peppi.to_training_frames(player_port: p, opponent_port: o)
        |> Enum.reject(&(&1.game_state.frame < 0))
        |> Enum.map(& &1.game_state)

      n = length(states)

      if n < 300 do
        nil
      else
        arr = List.to_tuple(states)
        my_sits = Situations.label_states(states, p, as: :set) |> List.to_tuple()
        opp_sits = Situations.label_states(states, o, as: :set) |> List.to_tuple()
        pl = fn i, who -> elem(arr, i).players[who] end
        my_hit = fn i -> MapSet.member?(elem(my_sits, i), :in_hitstun) end
        opp_hit = fn i -> MapSet.member?(elem(opp_sits, i), :in_hitstun) end

        events = Options.events(states, p) |> Enum.filter(&(&1.option in committal))

        rows =
          Enum.map(events, fn e ->
            i = e.index
            me = pl.(i, p)
            op = pl.(i, o)

            in_threat =
              not opp_hit.(i) and
                abs((me.x || 0.0) - (op.x || 0.0)) < range_x and
                abs((me.y || 0.0) - (op.y || 0.0)) < range_y

            punished =
              not my_hit.(i) and
                Enum.any?(i..min(i + pwin, n - 1), fn k -> my_hit.(k) and not my_hit.(max(k - 1, 0)) end)

            %{option: e.option, in_threat: in_threat, punished: punished}
          end)

        %{rows: rows, files: 1, frames: n}
      end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

expert_char = opts[:expert_char]

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

agg =
  Map.new(sets, fn {name, _} = s ->
    files = files_for.(s)
    pairs = files |> Enum.map(&{&1, port_for.(name, &1)}) |> Enum.reject(fn {_, p} -> is_nil(p) end)
    Output.puts("Scanning #{name}: #{length(pairs)}/#{length(files)} files")

    r =
      pairs
      |> Task.async_stream(fn {f, p} -> scan.(f, p) end,
        max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce(%{rows: [], files: 0, frames: 0}, fn {:ok, x}, acc ->
        if x, do: %{rows: x.rows ++ acc.rows, files: acc.files + 1, frames: acc.frames + x.frames}, else: acc
      end)

    {name, r}
  end)

names = Enum.map(sets, &elem(&1, 0))
f1 = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
per_min = fn count, frames -> :erlang.float_to_binary(count / max(frames / 3600, 1), decimals: 2) end

p_pun = fn rows ->
  if length(rows) < 10, do: "– (#{length(rows)})",
    else: f1.(Enum.count(rows, & &1.punished) / length(rows)) <> " (#{length(rows)})"
end

row = fn label, cell_fn -> "| #{label} | " <> Enum.map_join(names, " | ", cell_fn) <> " |" end
threat = fn n -> Enum.filter(agg[n].rows, & &1.in_threat) end

per_option =
  for opt <- committal do
    row.("in-threat #{opt}: P(punished) %", fn n ->
      p_pun.(threat.(n) |> Enum.filter(&(&1.option == opt)))
    end)
  end

table =
  [
    "| commitment | " <> Enum.map_join(names, " | ", fn n -> "#{n} (n=#{length(agg[n].rows)})" end) <> " |",
    "|---|" <> String.duplicate("---:|", length(names)),
    row.("committal options / min", fn n -> per_min.(length(agg[n].rows), agg[n].frames) end),
    row.("IN-THREAT committals / min", fn n -> per_min.(length(threat.(n)), agg[n].frames) end),
    row.("in-threat share of committals %", fn n ->
      r = agg[n].rows
      if r == [], do: "–", else: f1.(length(threat.(n)) / length(r))
    end),
    row.("P(punished | in-threat committal) %", fn n -> p_pun.(threat.(n)) end),
    row.("P(punished | out-of-threat committal) %", fn n ->
      p_pun.(Enum.reject(agg[n].rows, & &1.in_threat))
    end)
  ] ++ per_option
  |> Enum.join("\n")

report = """
# Punishable-commitment scorecard (F2)

Sets: #{Enum.map_join(names, " · ", fn n -> "#{n}: #{agg[n].files} files" end)}.
Committal = #{Enum.join(Enum.map(committal, &to_string/1), ", ")} (Options events).
In-threat = opponent not in hitstun, |dx| < #{range_x}, |dy| < #{range_y} at the
event frame. Punished = subject enters hitstun within #{pwin} f. Cells under
n=10 report "–".

#{table}

Read: the expert commits too — the difference that matters is
P(punished | in-threat), i.e. whether commitments are TIMED (opponent
committed/landing) or thrown out raw. High in-threat share + high
P(punished) = "puts itself into whiff punish" (C1's mechanism).
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
