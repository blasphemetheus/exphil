# C1 — neutral-exchange outcomes (EVAL_DIRECTIONS C1): the dense
# "harder to hit" number.
#
# An EXCHANGE starts when both players have been neutral (no hitstun,
# no knockdown lifecycle, no shield, both alive, not offstage) for
# --lead frames, and ends at the first HIT: someone's percent rises or they
# enter hitstun. Winner = the one who dealt it. How:
#   first_hit     loser was not attacking in the last --whiff frames
#   whiff_punish  loser had an attack state active within --whiff frames
#   trade         both took damage within 5 frames of each other
# Exchanges with no hit within --max frames end as "stall".
# Reported per set for the subject (bot / expert port 1) and the opponent.
#
#   mix run scripts/neutral_exchange.exs --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --set B1_human='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp' --expert expert \
#     --expert-limit 600 --out eval_runs/0829_neutral_exchange/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer, expert_limit: :integer,
             limit_files: :integer, lead: :integer, whiff: :integer, max: :integer, concurrency: :integer, out: :string]
  )

sets = Keyword.get_values(opts, :set) |> Enum.map(fn s -> [n, g] = String.split(s, "=", parts: 2); {n, g} end)
if sets == [], do: raise("--set NAME=GLOB required")
expert = opts[:expert] || "expert"
port = opts[:port] || 1
expert_port = opts[:expert_port] || 1
lead = opts[:lead] || 30
whiff = opts[:whiff] || 20
max_len = opts[:max] || 900
conc = opts[:concurrency] || 12

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(&(File.stat!(&1).size < 150_000))
  limit = if name == expert, do: opts[:expert_limit] || 600, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

Output.banner("Neutral exchanges (C1)")
Output.config(Enum.map(sets, fn {n, g} = s -> {n, "#{length(files_for.(s))} files (#{g})"} end) ++
  [{"Lead", "#{lead} f"}, {"Whiff window", "#{whiff} f"}, {"Max", "#{max_len} f"}])

hitstun = MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
lifecycle = MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
shield = MapSet.new([178, 179, 180])
attack = MapSet.new(Enum.to_list(44..69) ++ [50, 212, 214, 360, 361, 365, 366, 367])
dying = MapSet.new(0..10)

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
        act = fn i, who -> trunc(pl.(i, who).action || 0) end
        pct = fn i, who -> pl.(i, who).percent || 0.0 end

        neutral? = fn i ->
          Enum.all?([p, o], fn who ->
            a = act.(i, who)
            not MapSet.member?(hitstun, a) and not MapSet.member?(lifecycle, a) and
              not MapSet.member?(shield, a) and not MapSet.member?(dying, a)
          end) and not MapSet.member?(elem(my_sits, i), :offstage) and
            not MapSet.member?(elem(opp_sits, i), :offstage)
        end

        hit_on = fn i, who -> i > 0 and (pct.(i, who) > pct.(i - 1, who) or (MapSet.member?(hitstun, act.(i, who)) and not MapSet.member?(hitstun, act.(i - 1, who)))) end
        attacking_recently = fn i, who -> Enum.any?(max(i - whiff, 0)..i, fn k -> MapSet.member?(attack, act.(k, who)) end) end

        # walk frames: find lead-run of neutral, then scan to the first hit
        {exchanges, _} =
          Enum.reduce(0..(n - 1), {[], 0}, fn i, {acc, run} ->
            run = if neutral?.(i), do: run + 1, else: 0

            if run == lead do
              # exchange starts at i; scan forward
              j_end = min(i + max_len, n - 1)

              result =
                Enum.find_value((i + 1)..j_end, fn k ->
                  me = hit_on.(k, p)
                  them = hit_on.(k, o)

                  cond do
                    me and them -> {:trade, k - i}
                    them ->
                      near = Enum.any?((k + 1)..min(k + 5, n - 1), fn q -> hit_on.(q, p) end)
                      if near, do: {:trade, k - i}, else: {if(attacking_recently.(k, o), do: :win_whiff_punish, else: :win_first_hit), k - i}
                    me ->
                      near = Enum.any?((k + 1)..min(k + 5, n - 1), fn q -> hit_on.(q, o) end)
                      if near, do: {:trade, k - i}, else: {if(attacking_recently.(k, p), do: :loss_whiff_punished, else: :loss_first_hit), k - i}
                    true -> nil
                  end
                end) || {:stall, max_len}

              # restart the run so consecutive exchanges need a fresh lead
              {[result | acc], 0}
            else
              {acc, run}
            end
          end)

        %{ex: exchanges, files: 1, frames: n}
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
    Output.puts("Scanning #{name}: #{length(files)} files")

    r =
      files
      |> Task.async_stream(&scan.(&1, p), max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce(%{ex: [], files: 0, frames: 0}, fn {:ok, x}, acc ->
        if x, do: %{ex: x.ex ++ acc.ex, files: acc.files + 1, frames: acc.frames + x.frames}, else: acc
      end)

    {name, r}
  end)

names = Enum.map(sets, &elem(&1, 0))
f1 = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
f2 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end

col = fn n, fun -> fun.(agg[n].ex) end
share = fn ex, k -> if ex == [], do: "–", else: f1.(Enum.count(ex, &(elem(&1, 0) == k)) / length(ex)) end
decided = fn ex -> Enum.reject(ex, &(elem(&1, 0) in [:stall, :trade])) end

header = "| exchanges | " <> Enum.map_join(names, " | ", fn n -> "#{n} (n=#{length(agg[n].ex)})" end) <> " |"
sep = "|---|" <> String.duplicate("---:|", length(names))
rows = [
  "| exchanges / min | " <> Enum.map_join(names, " | ", fn n -> f2.(length(agg[n].ex) / max(agg[n].frames / 3600, 1)) end) <> " |",
  "| **subject win rate (decided)** | " <> Enum.map_join(names, " | ", fn n -> col.(n, fn ex -> d = decided.(ex); if d == [], do: "–", else: "**" <> f1.(Enum.count(d, &(elem(&1, 0) in [:win_first_hit, :win_whiff_punish])) / length(d)) <> "**" end) end) <> " |",
  "| win: first hit % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].ex, :win_first_hit) end) <> " |",
  "| win: whiff punish % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].ex, :win_whiff_punish) end) <> " |",
  "| loss: got first-hit % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].ex, :loss_first_hit) end) <> " |",
  "| loss: whiff punished % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].ex, :loss_whiff_punished) end) <> " |",
  "| trade % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].ex, :trade) end) <> " |",
  "| stall (no hit in #{max_len} f) % | " <> Enum.map_join(names, " | ", fn n -> share.(agg[n].ex, :stall) end) <> " |",
  "| median length (f) | " <> Enum.map_join(names, " | ", fn n -> col.(n, fn ex -> if ex == [], do: "–", else: ex |> Enum.map(&elem(&1, 1)) |> Enum.sort() |> Enum.at(div(length(ex), 2)) |> to_string() end) end) <> " |"
]

report = """
# Neutral-exchange outcomes (C1)

Sets: #{Enum.map_join(names, " · ", fn n -> "#{n}: #{agg[n].files} files" end)}. Exchange = #{lead} f of mutual
neutral, then the first hit within #{max_len} f. Subject = port #{port} (expert: port #{expert_port}).
"whiff punish" = the loser had an attack state active within #{whiff} f of being hit.
Expert sets are Fox dittos between two strong players, so the expert subject's win rate
is ~50% by construction — the bot's number is read against that.

#{Enum.join([header, sep | rows], "\n")}
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
