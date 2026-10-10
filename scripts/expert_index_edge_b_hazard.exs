# The EXPERT's B hazard at the Illusion state, straight from a :wide labeler
# index (2026-10-10 14:10; the counterpart of dagger_set_edge_b_hazard.exs,
# which measures the bot and its labels on the bot's own states). State S =
# airborne, within -30..+10 of the edge, facing OUT, B up at t-1, not in
# hitstun; cells by the previous stick x (outward / centred / toward the
# stage) and jumps left. Per cell: rows, P(B press), P(B press with the
# stick outward = an outward Illusion), P(jump press).
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/expert_index_edge_b_hazard.exs INDEX.bin
[path] = System.argv()
m = path |> File.read!() |> :erlang.binary_to_term()
:wide = m.window
dims = m.dims
nd = length(dims)
at = fn names, name -> Enum.find_index(names, &(&1 == name)) end
# dims order and scales from the labeler: value = stored / w * scale
scale = %{y: {40.0, 1.0}, dist: {30.0, 1.0}, jumps: {1.0, 3.0}, facing: {1.0, 1.0}, hitstun: {1.0, 2.0}, prev_sx: {1.0, 1.5}, prev_b: {1.0, 1.5}, grounded: {1.0, 3.0}}
unscale = fn row, name ->
  {s, w} = scale[name]
  Enum.at(row, at.(dims, name)) / w * s
end

rows = for <<v::float-32-native <- m.x>>, do: v
labs = for <<v::float-32-native <- m.labels>>, do: v
xs = Enum.chunk_every(rows, nd)
ls = Enum.chunk_every(labs, 12)
true = length(xs) == m.n and length(ls) == m.n

cells =
  Enum.zip(xs, ls)
  |> Enum.filter(fn {x, _} ->
    d = unscale.(x, :dist)
    unscale.(x, :grounded) < 0.5 and d > -30 and d < 10 and unscale.(x, :facing) < 0 and
      unscale.(x, :prev_b) < 0.5 and unscale.(x, :hitstun) < 0.5
  end)
  |> Enum.map(fn {x, [msx, _msy, _, _, _a, b, bx, by | _]} ->
    psx = unscale.(x, :prev_sx)
    stick = cond do
      abs(psx) < 0.33 -> :x0
      psx < 0 -> :outward
      true -> :toward_stage
    end
    %{stick: stick, jumps: min(round(unscale.(x, :jumps)), 1), b: b > 0.5, out: b > 0.5 and msx < 0.5 - 0.165, jump: bx > 0.5 or by > 0.5}
  end)

pct = fn n, d -> if d == 0, do: "-", else: :io_lib.format("~.4f", [n / d]) |> to_string() end
IO.puts("RESULT expert edge B hazard #{Path.basename(path)} (#{m.meta.games} games, #{m.n} rows): state rows #{length(cells)} (airborne, -30..+10 of the edge, facing out, B up at t-1, no hitstun)")
IO.puts("  cell (prev stick, jumps): rows | P(B press) | P(B & stick outward) | P(jump press)")
for stick <- [:outward, :x0, :toward_stage], j <- [1, 0] do
  c = Enum.filter(cells, &(&1.stick == stick and &1.jumps == j))
  n = length(c)
  IO.puts("  #{String.pad_trailing("#{stick} j#{j}", 18)} #{String.pad_leading("#{n}", 6)} | #{pct.(Enum.count(c, & &1.b), n)} | #{pct.(Enum.count(c, & &1.out), n)} | #{pct.(Enum.count(c, & &1.jump), n)}")
end
n = length(cells)
IO.puts("  all                #{String.pad_leading("#{n}", 6)} | #{pct.(Enum.count(cells, & &1.b), n)} | #{pct.(Enum.count(cells, & &1.out), n)} | #{pct.(Enum.count(cells, & &1.jump), n)}")
