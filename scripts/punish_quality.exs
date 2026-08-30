# C2 — punish quality (EVAL_DIRECTIONS).
#
# Damage per opening, hits per opening, openings per minute, and the
# fraction of openings that end in a kill — bot vs expert, same detector.
#
# An OPENING starts when the opponent's percent rises after >= 30 frames
# without rising; it accumulates while further rises land within 45 frames
# of the last one (combo gap), and it converts to a KILL if the opponent
# loses a stock within 90 frames of the last hit.
#
#   mix run scripts/punish_quality.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' --expert expert \
#     --set B1='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp' \
#     --expert-limit 400 --out eval_runs/0830_punish_quality/RESULTS.md
#
# Options: --set NAME=GLOB (repeatable) · --expert NAME · --port N (bot sets,
#   default 1) · --expert-port N | --expert-char N (per-file auto-detect,
#   default char 2) · --expert-limit N (400) · --limit-files N ·
#   --concurrency N (8) · --out FILE.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer,
             expert_char: :integer, expert_limit: :integer, limit_files: :integer,
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
port = opts[:port] || 1
expert_char = opts[:expert_char] || 2
conc = opts[:concurrency] || 8

# combo gap / pre-opening quiet / kill window, in frames
gap = 45
pre_quiet = 30
kill_window = 90

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(fn f -> File.stat!(f).size < 150_000 end)
  limit = if name == expert, do: opts[:expert_limit] || 400, else: opts[:limit_files]
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

# openings from (opp_percent, opp_stock) streams
segment = fn opp ->
  # opp: [{percent, stock}]
  {segs, cur, _quiet, _prev} =
    opp
    |> Enum.with_index()
    |> Enum.reduce({[], nil, 9999, nil}, fn {{pct, stock}, i}, {segs, cur, quiet, prev} ->
      {prev_pct, prev_stock} = prev || {pct, stock}
      rose = pct > prev_pct + 0.01 and stock == prev_stock

      cond do
        rose and cur == nil and quiet >= pre_quiet ->
          {segs, %{start: i, last_hit: i, dmg: pct - prev_pct, hits: 1, kill: false}, 0, {pct, stock}}

        rose and cur != nil and i - cur.last_hit <= gap ->
          {segs, %{cur | last_hit: i, dmg: cur.dmg + (pct - prev_pct), hits: cur.hits + 1}, 0, {pct, stock}}

        rose ->
          # rose but previous segment expired -> close it, maybe open new
          segs = if cur, do: [cur | segs], else: segs
          cur = if quiet >= pre_quiet,
            do: %{start: i, last_hit: i, dmg: pct - prev_pct, hits: 1, kill: false}, else: nil
          {segs, cur, 0, {pct, stock}}

        stock < prev_stock and cur != nil and i - cur.last_hit <= kill_window ->
          {[%{cur | kill: true} | segs], nil, quiet + 1, {pct, stock}}

        cur != nil and i - cur.last_hit > gap ->
          {[cur | segs], nil, quiet + 1, {pct, stock}}

        true ->
          {segs, cur, quiet + 1, {pct, stock}}
      end
    end)

  segs = if cur, do: [cur | segs], else: segs
  Enum.reverse(segs)
end

scan_file = fn name, path ->
  try do
    with {:ok, p} <- resolve_port.(name, path),
         {:ok, replay} <- Peppi.parse(path, player_port: p) do
    opp_port = if p == 1, do: 2, else: 1

    frames =
      replay
      |> Peppi.to_training_frames(player_port: p, opponent_port: opp_port)
      |> Enum.reject(&(&1.game_state.frame < 0))

    if length(frames) < 1800 do
      nil
    else
      opp =
        Enum.map(frames, fn f ->
          op = f.game_state.players[opp_port]
          {(op.percent || 0) * 1.0, op.stock || 0}
        end)

      segs = segment.(opp)

      %{
        mins: length(frames) / 3600,
        openings: length(segs),
        dmg: Enum.sum(Enum.map(segs, & &1.dmg)),
        hits: Enum.sum(Enum.map(segs, & &1.hits)),
        kills: Enum.count(segs, & &1.kill)
      }
    end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

f = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end

rows =
  Enum.map(sets, fn {name, _} = s ->
    files = files_for.(s)
    Output.puts("Scanning #{name}: #{length(files)} files")

    agg =
      files
      |> Task.async_stream(&scan_file.(name, &1), max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce(%{mins: 0.0, openings: 0, dmg: 0.0, hits: 0, kills: 0, games: 0}, fn {:ok, r}, a ->
        if r do
          %{a | mins: a.mins + r.mins, openings: a.openings + r.openings, dmg: a.dmg + r.dmg,
            hits: a.hits + r.hits, kills: a.kills + r.kills, games: a.games + 1}
        else
          a
        end
      end)

    op = max(agg.openings, 1)

    "| #{name} | #{agg.games} | #{f.(agg.openings / max(agg.mins, 0.01))} | " <>
      "#{f.(agg.dmg / op)} | #{f.(agg.hits / op)} | #{f.(agg.kills / op * 100)} |"
  end)

report = """
# C2 — punish quality

Opening = opp percent rises after #{pre_quiet}f quiet; combo gap #{gap}f;
kill window #{kill_window}f. Same detector for every set.

| set | games | openings/min | damage/opening | hits/opening | kill conversion % |
|---|---:|---:|---:|---:|---:|
#{Enum.join(rows, "\n")}
"""

Output.banner("C2 — punish quality")
IO.puts(report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
