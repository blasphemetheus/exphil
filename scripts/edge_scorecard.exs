# A2 — edgeguard / recovery scorecards, bot vs expert (EVAL_DIRECTIONS A2).
#
# EDGEGUARD episodes: the opponent goes offstage (their :offstage label turns
# on while the subject is not) — what does the subject do FIRST within the
# episode (first Options event: aerial / special / dash / shield / ledge /
# nothing), and how does it end within --horizon frames:
#   kill      opponent lost a stock
#   returned  opponent back on stage / on the ledge, subject alive
#   traded    subject lost a stock (went out and died)
# RECOVERY episodes: the subject goes offstage — route = first of
# {upB, sideB, double_jump, airdodge, ledge} used, and outcome:
#   back      subject on ground / ledge within --horizon
#   died      subject lost a stock
# Episodes end at the first outcome or at the horizon (→ "open").
#
#   mix run scripts/edge_scorecard.exs --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --set ep10_human='eval_runs/0828_livelook_btn05/2026-08-Mainline/*.slp' \
#     --expert expert --expert-limit 600 --out eval_runs/0829_edge_scorecard/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Options
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer,
             expert_char: :integer, expert_limit: :integer,
             limit_files: :integer, horizon: :integer, concurrency: :integer, out: :string]
  )

sets = Keyword.get_values(opts, :set) |> Enum.map(fn s -> [n, g] = String.split(s, "=", parts: 2); {n, g} end)
if sets == [], do: raise("--set NAME=GLOB required")
expert = opts[:expert] || "expert"
port = opts[:port] || 1
expert_port = opts[:expert_port] || 1
horizon = opts[:horizon] || 240
conc = opts[:concurrency] || 12

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(&(File.stat!(&1).size < 150_000))
  limit = if name == expert, do: opts[:expert_limit] || 600, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

Output.banner("Edgeguard / recovery scorecards (A2)")
Output.config(Enum.map(sets, fn {n, g} = s -> {n, "#{length(files_for.(s))} files (#{g})"} end) ++ [{"Horizon", "#{horizon} f"}])

firefox = [353, 354, 355, 356]
illusion = [350, 351, 352]
ledge_states = [252, 253]

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
        events = Options.events(states, p)
        ev_by_i = Enum.group_by(events, & &1.index)
        pl = fn i, who -> elem(arr, i).players[who] end
        off_me = fn i -> MapSet.member?(elem(my_sits, i), :offstage) end
        off_opp = fn i -> MapSet.member?(elem(opp_sits, i), :offstage) end
        act = fn i, who -> trunc(pl.(i, who).action || 0) end
        onstage = fn i, who -> (pl.(i, who).on_ground and not (if who == p, do: off_me.(i), else: off_opp.(i))) or act.(i, who) in ledge_states end

        first_event = fn i, j ->
          Enum.find_value(i..min(j, n - 1), fn k ->
            case Map.get(ev_by_i, k) do
              [e | _] -> e.option
              _ -> nil
            end
          end) || :nothing
        end

        # --- edgeguard episodes: opp goes offstage while I am not ---
        eg =
          1..(n - 1)
          |> Enum.filter(fn i -> off_opp.(i) and not off_opp.(i - 1) and not off_me.(i) end)
          |> Enum.map(fn i ->
            j = min(i + horizon, n - 1)
            my_stock = pl.(i, p).stock
            opp_stock = pl.(i, o).stock

            outcome =
              Enum.find_value(i..j, fn k ->
                cond do
                  is_integer(pl.(k, o).stock) and pl.(k, o).stock < opp_stock -> :kill
                  is_integer(pl.(k, p).stock) and pl.(k, p).stock < my_stock -> :traded
                  k > i + 20 and onstage.(k, o) -> :returned
                  true -> nil
                end
              end) || :open

            {first_event.(i, min(i + 60, n - 1)), outcome}
          end)

        # --- recovery episodes: I go offstage ---
        rc =
          1..(n - 1)
          |> Enum.filter(fn i -> off_me.(i) and not off_me.(i - 1) end)
          |> Enum.map(fn i ->
            j = min(i + horizon, n - 1)
            my_stock = pl.(i, p).stock

            route =
              Enum.find_value(i..j, fn k ->
                a = act.(k, p)
                cond do
                  a in firefox -> :upB
                  a in illusion -> :sideB
                  a == 25 -> :double_jump
                  a == 236 -> :airdodge
                  a in ledge_states -> :ledge_direct
                  true -> nil
                end
              end) || :none

            outcome =
              Enum.find_value(i..j, fn k ->
                cond do
                  is_integer(pl.(k, p).stock) and pl.(k, p).stock < my_stock -> :died
                  k > i + 10 and onstage.(k, p) -> :back
                  true -> nil
                end
              end) || :open

            {route, outcome}
          end)

        %{eg: eg, rc: rc, files: 1, frames: n}
      end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

# --expert-char N: per-file expert port by character (E1: fox sits on
# varying ports; a fixed port mixes opponent characters into the baseline).
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

    pairs =
      files |> Enum.map(&{&1, port_for.(name, &1)}) |> Enum.reject(fn {_, p} -> is_nil(p) end)

    Output.puts("Scanning #{name}: #{length(pairs)}/#{length(files)} files")

    r =
      pairs
      |> Task.async_stream(fn {f, p} -> scan.(f, p) end,
        max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce(%{eg: [], rc: [], files: 0, frames: 0}, fn {:ok, x}, acc ->
        if x, do: %{eg: x.eg ++ acc.eg, rc: x.rc ++ acc.rc, files: acc.files + 1, frames: acc.frames + x.frames}, else: acc
      end)

    {name, r}
  end)

names = Enum.map(sets, &elem(&1, 0))
f1 = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
share = fn list, key_fn, key -> if list == [], do: "–", else: f1.(Enum.count(list, &(key_fn.(&1) == key)) / length(list)) end

eg_table = fn ->
  header = "| edgeguard | " <> Enum.map_join(names, " | ", fn n -> "#{n} (n=#{length(agg[n].eg)})" end) <> " |"
  sep = "|---|" <> String.duplicate("---:|", length(names))
  per_min = "| episodes / min | " <> Enum.map_join(names, " | ", fn n -> :erlang.float_to_binary(length(agg[n].eg) / max(agg[n].frames / 3600, 1), decimals: 2) end) <> " |"
  outs = for k <- [:kill, :returned, :traded, :open], do: "| outcome: #{k} % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].eg, &elem(&1, 1), k) end) <> " |"
  firsts = for k <- [:aerial, :special, :dash, :double_jump, :wavedash, :shield_on, :grab, :jab, :smash, :tilt, :nothing], do: "| first option: #{k} % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].eg, &elem(&1, 0), k) end) <> " |"
  Enum.join([header, sep, per_min] ++ outs ++ firsts, "\n")
end

rc_table = fn ->
  header = "| recovery | " <> Enum.map_join(names, " | ", fn n -> "#{n} (n=#{length(agg[n].rc)})" end) <> " |"
  sep = "|---|" <> String.duplicate("---:|", length(names))
  per_min = "| episodes / min | " <> Enum.map_join(names, " | ", fn n -> :erlang.float_to_binary(length(agg[n].rc) / max(agg[n].frames / 3600, 1), decimals: 2) end) <> " |"
  outs = for k <- [:back, :died, :open], do: "| outcome: #{k} % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].rc, &elem(&1, 1), k) end) <> " |"
  routes = for k <- [:upB, :sideB, :double_jump, :airdodge, :ledge_direct, :none], do: "| first route: #{k} % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].rc, &elem(&1, 0), k) end) <> " |"
  died_by = for k <- [:upB, :sideB, :double_jump, :airdodge, :none], do: "| died % given route #{k} | " <> Enum.map_join(names, " | ", fn n -> sub = Enum.filter(agg[n].rc, &(elem(&1, 0) == k)); if length(sub) < 5, do: "– (#{length(sub)})", else: f1.(Enum.count(sub, &(elem(&1, 1) == :died)) / length(sub)) <> " (#{length(sub)})" end) <> " |"
  Enum.join([header, sep, per_min] ++ outs ++ routes ++ died_by, "\n")
end

report = """
# Edgeguard / recovery scorecards (A2)

Sets: #{Enum.map_join(names, " · ", fn n -> "#{n}: #{agg[n].files} files" end)}. Horizon #{horizon} f.
Edgeguard episode = opponent goes offstage while the subject is not; first option = the
subject's first `Options` event within 60 f. Recovery episode = subject goes offstage;
route = first of upB / sideB / double jump / airdodge / ledge within the horizon.

## Edgeguarding (subject on stage, opponent off)

#{eg_table.()}

## Recovering (subject off stage)

#{rc_table.()}
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
