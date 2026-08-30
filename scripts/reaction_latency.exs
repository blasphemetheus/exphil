# C3 — reaction latency (EVAL_DIRECTIONS).
#
# Frames from a trigger to the subject's next action-state CHANGE, bot vs
# expert, same detector. Dithering shows up as a long/flat latency tail.
#
# Triggers:
#   opp_lands   opponent airborne -> grounded (actionable landing)
#   opp_ledge   opponent enters CliffCatch/CliffWait (252/253)
#   self_lands  subject airborne -> grounded
#
#   mix run scripts/reaction_latency.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' --expert expert \
#     --set B1='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp' \
#     --expert-limit 300 --out eval_runs/0830_reaction_latency/RESULTS.md
#
# Options: --set NAME=GLOB (repeatable) · --expert NAME · --port N (bot sets,
#   default 1) · --expert-port N | --expert-char N (auto-detect, default 2)
#   --expert-limit N (300) · --limit-files N · --max-latency N (45)
#   --concurrency N (8) · --out FILE.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer,
             expert_char: :integer, expert_limit: :integer, limit_files: :integer,
             max_latency: :integer, concurrency: :integer, out: :string]
  )

sets =
  Keyword.get_values(opts, :set)
  |> Enum.map(fn s ->
    [name, glob] = String.split(s, "=", parts: 2)
    {name, glob}
  end)

if sets == [], do: raise("at least one --set NAME=GLOB")
expert = opts[:expert] || "expert"
port = opts[:port] || 1
expert_char = opts[:expert_char] || 2
max_lat = opts[:max_latency] || 45
conc = opts[:concurrency] || 8

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(fn f -> File.stat!(f).size < 150_000 end)
  limit = if name == expert, do: opts[:expert_limit] || 300, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

resolve_port = fn name, path ->
  cond do
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

cliff = MapSet.new([252, 253])

scan_file = fn name, path ->
  try do
    with {:ok, p} <- resolve_port.(name, path),
         {:ok, replay} <- Peppi.parse(path, player_port: p) do
    opp_port = if p == 1, do: 2, else: 1

    frames =
      replay
      |> Peppi.to_training_frames(player_port: p, opponent_port: opp_port)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(fn f ->
        me = f.game_state.players[p]
        op = f.game_state.players[opp_port]

        %{
          a: trunc(me.action || 0), g: !!me.on_ground,
          oa: trunc(op.action || 0), og: !!op.on_ground
        }
      end)

    if length(frames) < 1800 do
      nil
    else
      arr = List.to_tuple(frames)
      n = tuple_size(arr)

      # latency: frames until subject action changes after trigger index
      latency = fn i ->
        base = elem(arr, i).a

        Enum.reduce_while((i + 1)..min(i + max_lat, n - 1), nil, fn j, _ ->
          if elem(arr, j).a != base, do: {:halt, j - i}, else: {:cont, nil}
        end) || max_lat
      end

      trigger? = fn kind, f, prev ->
        case kind do
          :opp_lands -> f.og and not prev.og
          :opp_ledge -> MapSet.member?(cliff, f.oa) and not MapSet.member?(cliff, prev.oa)
          :self_lands -> f.g and not prev.g
        end
      end

      triggers = fn kind ->
        for i <- 1..(n - 1),
            trigger?.(kind, elem(arr, i), elem(arr, i - 1)),
            do: latency.(i)
      end

      %{opp_lands: triggers.(:opp_lands), opp_ledge: triggers.(:opp_ledge),
        self_lands: triggers.(:self_lands)}
    end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

stats = fn lats ->
  n = length(lats)

  if n == 0 do
    "– (0)"
  else
    sorted = Enum.sort(lats)
    med = Enum.at(sorted, div(n, 2))
    p90 = Enum.at(sorted, min(trunc(n * 0.9), n - 1))
    mean = Enum.sum(lats) / n
    capped = Enum.count(lats, &(&1 >= max_lat)) / n
    "med #{med} · mean #{:erlang.float_to_binary(mean, decimals: 1)} · p90 #{p90} · ≥cap #{:erlang.float_to_binary(capped * 100, decimals: 0)}% (n=#{n})"
  end
end

rows =
  Enum.map(sets, fn {name, _} = s ->
    files = files_for.(s)
    Output.puts("Scanning #{name}: #{length(files)} files")

    agg =
      files
      |> Task.async_stream(&scan_file.(name, &1), max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce(%{opp_lands: [], opp_ledge: [], self_lands: []}, fn {:ok, r}, a ->
        if r do
          %{opp_lands: r.opp_lands ++ a.opp_lands, opp_ledge: r.opp_ledge ++ a.opp_ledge,
            self_lands: r.self_lands ++ a.self_lands}
        else
          a
        end
      end)

    "| #{name} | #{stats.(agg.opp_lands)} | #{stats.(agg.opp_ledge)} | #{stats.(agg.self_lands)} |"
  end)

report = """
# C3 — reaction latency (frames to next action-state change; cap #{max_lat})

| set | after opp lands | after opp grabs ledge | after self lands |
|---|---|---|---|
#{Enum.join(rows, "\n")}

Caveats: action-state change is a proxy for "reacted" (landing lag ends
count as changes; identical for every set, so comparisons stand). The cap
share is the dithering signal.
"""

Output.banner("C3 — reaction latency")
IO.puts(report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
