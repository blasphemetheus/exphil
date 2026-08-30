# C4 — stock-loss forensics, bot vs expert (EVAL_DIRECTIONS C4). Extends
# sd_scan: every death of the subject is classified by HOW it happened.
#
#   unforced_walkoff  no hit in --lookback f; last grounded action was walk/
#                     dash/run past the edge (ran off)
#   unforced_fall     no hit in --lookback f; was offstage/airborne and never
#                     made it back (mis-recovery with nobody hitting them)
#   edgeguarded       hit while offstage (or within 30 f of going off) in the
#                     lookback, then died
#   comboed           >= 2 hits taken in the lookback window ending in a hit
#                     that killed (hit chain)
#   neutral_kill      exactly one hit in the lookback and it killed from on stage
#   other             anything else (e.g. sacrifice / unusual)
# Also: mean % at death, and the killer's action family for hit deaths.
#
#   mix run scripts/death_classifier.exs --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --set B1_human='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp' --expert expert \
#     --expert-limit 800 --out eval_runs/0829_death_classifier/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer, expert_limit: :integer,
             limit_files: :integer, lookback: :integer, concurrency: :integer, out: :string]
  )

sets = Keyword.get_values(opts, :set) |> Enum.map(fn s -> [n, g] = String.split(s, "=", parts: 2); {n, g} end)
if sets == [], do: raise("--set NAME=GLOB required")
expert = opts[:expert] || "expert"
port = opts[:port] || 1
expert_port = opts[:expert_port] || 1
lookback = opts[:lookback] || 150
conc = opts[:concurrency] || 12

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(&(File.stat!(&1).size < 150_000))
  limit = if name == expert, do: opts[:expert_limit] || 800, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

Output.banner("Death classifier (C4)")
Output.config(Enum.map(sets, fn {n, g} = s -> {n, "#{length(files_for.(s))} files (#{g})"} end) ++ [{"Lookback", "#{lookback} f"}])

hitstun = MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
ground_move = MapSet.new([15, 16, 17, 18, 20, 21, 23])

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
        sits = Situations.label_states(states, p, as: :set) |> List.to_tuple()
        geo = Situations.geometry(elem(arr, 0).stage)
        pl = fn i -> elem(arr, i).players[p] end
        act = fn i -> trunc(pl.(i).action || 0) end
        off = fn i -> MapSet.member?(elem(sits, i), :offstage) end

        losses =
          Enum.filter(1..(n - 1), fn i ->
            a = pl.(i - 1).stock
            b = pl.(i).stock
            is_integer(a) and is_integer(b) and b < a
          end)

        Enum.map(losses, fn i ->
          lo = max(0, i - lookback)
          idx = Enum.to_list(lo..i)
          # hits taken: percent rises
          hits = Enum.filter(idx, fn k -> k > 0 and (pl.(k).percent || 0.0) > (pl.(k - 1).percent || 0.0) end)
          n_hits = length(hits)
          last_hit = List.last(hits)
          hit_offstage? = last_hit != nil and Enum.any?(max(last_hit - 30, lo)..last_hit, off)
          grounded = idx |> Enum.reverse() |> Enum.find(fn k -> pl.(k).on_ground end)
          walked? = grounded != nil and MapSet.member?(ground_move, act.(grounded)) and abs(pl.(grounded).x || 0.0) > geo.edge - 15
          was_off? = Enum.any?(idx, off)
          pct_at_death = idx |> Enum.reverse() |> Enum.find_value(fn k -> if act.(k) > 10, do: pl.(k).percent, else: nil end) || 0.0

          class =
            cond do
              n_hits == 0 and walked? -> :unforced_walkoff
              n_hits == 0 and was_off? -> :unforced_fall
              n_hits == 0 -> :other
              hit_offstage? -> :edgeguarded
              n_hits >= 2 -> :comboed
              true -> :neutral_kill
            end

          {class, pct_at_death}
        end)
        |> then(&%{deaths: &1, files: 1, frames: n})
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
      |> Enum.reduce(%{deaths: [], files: 0, frames: 0}, fn {:ok, x}, acc ->
        if x, do: %{deaths: x.deaths ++ acc.deaths, files: acc.files + 1, frames: acc.frames + x.frames}, else: acc
      end)

    {name, r}
  end)

names = Enum.map(sets, &elem(&1, 0))
f1 = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
classes = [:unforced_walkoff, :unforced_fall, :edgeguarded, :comboed, :neutral_kill, :other]

header = "| death class | " <> Enum.map_join(names, " | ", fn n -> "#{n} (deaths=#{length(agg[n].deaths)})" end) <> " |"
sep = "|---|" <> String.duplicate("---:|", length(names))
rows =
  ["| deaths / game | " <> Enum.map_join(names, " | ", fn n -> :erlang.float_to_binary(length(agg[n].deaths) / max(agg[n].files, 1), decimals: 2) end) <> " |"] ++
  Enum.map(classes, fn c -> "| #{c} % | " <> Enum.map_join(names, " | ", fn n -> d = agg[n].deaths; if d == [], do: "–", else: f1.(Enum.count(d, &(elem(&1, 0) == c)) / length(d)) end) <> " |" end) ++
  ["| **unforced (walkoff + fall) %** | " <> Enum.map_join(names, " | ", fn n -> d = agg[n].deaths; if d == [], do: "–", else: "**" <> f1.(Enum.count(d, &(elem(&1, 0) in [:unforced_walkoff, :unforced_fall])) / length(d)) <> "**" end) <> " |",
   "| mean % at death | " <> Enum.map_join(names, " | ", fn n -> d = agg[n].deaths; if d == [], do: "–", else: :erlang.float_to_binary(Enum.sum(Enum.map(d, &elem(&1, 1))) / length(d), decimals: 0) end) <> " |",
   "| mean % at death, hit deaths only | " <> Enum.map_join(names, " | ", fn n -> d = Enum.filter(agg[n].deaths, &(elem(&1, 0) in [:edgeguarded, :comboed, :neutral_kill])); if d == [], do: "–", else: :erlang.float_to_binary(Enum.sum(Enum.map(d, &elem(&1, 1))) / length(d), decimals: 0) end) <> " |"]

report = """
# Death classifier (C4)

Sets: #{Enum.map_join(names, " · ", fn n -> "#{n}: #{agg[n].files} files" end)}. Lookback #{lookback} f before each stock loss.
Unforced = no hit taken in the lookback. Edgeguarded = last hit taken while (or within 30 f of being)
offstage. Comboed = ≥2 hits in the lookback. Neutral kill = one hit, on stage.

#{Enum.join([header, sep | rows], "\n")}
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
