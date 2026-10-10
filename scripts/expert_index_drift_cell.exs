# Label content of the drift cell in an expert recovery index (2026-10-10,
# the labeler v5 `:air` window gate): rows airborne over the stage with the
# double jump spent, by distance inside the edge — what the expert's input is
# there (stick toward the stage / toward the edge / down / up, B, jump) and
# whether the next state is still drifting out. This is the "what will the
# labels say" check before rolling a round on the window.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/expert_index_drift_cell.exs [data/silent_fall/expert_recovery_index_v5.bin]
path = List.first(System.argv()) || "data/silent_fall/expert_recovery_index_v5.bin"
m = File.read!(path) |> :erlang.binary_to_term()
dims = m.dims
nd = length(dims)
# x rows are scaled * weighted; undo both (dims from the labeler: {name, scale, weight})
spec = %{y: {40.0, 1.0}, dist: {30.0, 1.0}, vx: {2.0, 1.0}, vy: {2.0, 1.0}, jumps: {1.0, 3.0}, facing: {1.0, 1.0}, grounded: {1.0, 3.0},
         prev_sx: {1.0, 1.5}, prev_sy: {1.0, 1.5}, hitstun: {1.0, 2.0}}
idx = fn name -> Enum.find_index(dims, &(&1 == name)) || raise("no dim #{name} in #{inspect(dims)}") end
raw = fn row, name -> {s, w} = spec[name]; Enum.at(row, idx.(name)) * s / w end

xs = for <<v::float-32-native <- m.x>>, do: v
ls = for <<v::float-32-native <- m.labels>>, do: v
rows = Enum.chunk_every(xs, nd)
labels = Enum.chunk_every(ls, 12)
true = length(rows) == m.n and length(labels) == m.n
IO.puts("#{Path.basename(path)}: #{m.n} rows, window #{inspect(m[:window])}, dims #{nd}")

band = fn d -> cond do d > 0 -> "offstage"; d > -15 -> "-15..0"; d > -30 -> "-30..-15"; d > -50 -> "-50..-30"; d > -80 -> "-80..-50"; true -> "<-80" end end
cell =
  Enum.zip(rows, labels)
  |> Enum.filter(fn {r, _} -> raw.(r, :grounded) < 0.5 and raw.(r, :jumps) < 0.5 and raw.(r, :hitstun) < 0.5 and raw.(r, :dist) <= 0.0 end)
IO.puts("airborne over the stage, no jump, no hitstun: #{length(cell)} rows")
pct = fn a, b -> if b == 0, do: "-", else: :erlang.float_to_binary(100 * a / b, decimals: 1) <> " %" end
# label stick in the TOWARD-stage frame: msx -> (msx - 0.5) * 2, positive = toward the stage
for k <- ["-15..0", "-30..-15", "-50..-30", "-80..-50", "<-80"] do
  cs = Enum.filter(cell, fn {r, _} -> band.(raw.(r, :dist)) == k end)
  n = length(cs)
  sx = fn {_, l} -> (Enum.at(l, 0) - 0.5) * 2 end
  sy = fn {_, l} -> (Enum.at(l, 1) - 0.5) * 2 end
  inward = Enum.count(cs, &(sx.(&1) >= 0.6))
  outward = Enum.count(cs, &(sx.(&1) <= -0.6))
  down = Enum.count(cs, &(sy.(&1) <= -0.6))
  up = Enum.count(cs, &(sy.(&1) >= 0.6))
  b = Enum.count(cs, fn {_, l} -> Enum.at(l, 5) > 0.5 end)
  jump = Enum.count(cs, fn {_, l} -> Enum.at(l, 6) > 0.5 or Enum.at(l, 7) > 0.5 end)
  # what the player was already doing (prev input, toward frame) and moving
  prev_out = Enum.count(cs, fn {r, _} -> raw.(r, :prev_sx) <= -0.6 end)
  moving_out = Enum.count(cs, fn {r, _} -> raw.(r, :vx) < -0.3 end)
  facing_out = Enum.count(cs, fn {r, _} -> raw.(r, :facing) < 0 end)
  IO.puts("  #{String.pad_trailing(k, 9)} n=#{String.pad_leading(to_string(n), 6)}  label stick: toward stage #{pct.(inward, n)}  toward edge #{pct.(outward, n)}  down #{pct.(down, n)}  up #{pct.(up, n)}  B #{pct.(b, n)}  jump #{pct.(jump, n)}  | state: prev stick outward #{pct.(prev_out, n)}  moving out #{pct.(moving_out, n)}  facing out #{pct.(facing_out, n)}")
  # the cell that makes the bot's carried trips: moving out with the stick held outward — what does the expert's label do THERE
  sub = Enum.filter(cs, fn {r, _} -> raw.(r, :prev_sx) <= -0.6 and raw.(r, :vx) < -0.3 end)
  ns = length(sub)
  if ns > 0 do
    IO.puts("            of which prev stick outward AND moving out: n=#{ns}  label keeps outward #{pct.(Enum.count(sub, &(sx.(&1) <= -0.6)), ns)}  turns to stage #{pct.(Enum.count(sub, &(sx.(&1) >= 0.6)), ns)}  releases #{pct.(Enum.count(sub, &(abs(sx.(&1)) < 0.33)), ns)}  down #{pct.(Enum.count(sub, &(sy.(&1) <= -0.6)), ns)}  B #{pct.(Enum.count(sub, fn {_, l} -> Enum.at(l, 5) > 0.5 end), ns)}")
  end
end

# the gate line for the queue: pooled over -80..-15, the bot's carried-trip sub-cell (prev stick outward, moving
# out) — share of rows where the expert's label is NOT "keep the stick outward" (turn / release / down / up)
pool = Enum.filter(cell, fn {r, _} -> d = raw.(r, :dist); d > -80 and d <= -15 and raw.(r, :prev_sx) <= -0.6 and raw.(r, :vx) < -0.3 end)
np = length(pool)
not_out = Enum.count(pool, fn {_, l} -> (Enum.at(l, 0) - 0.5) * 2 > -0.6 end)
IO.puts("RESULT drift cell -80..-15, prev stick outward & moving out: n=#{np}  label not outward #{if np == 0, do: 0.0, else: Float.round(100 * not_out / np, 1)} %")
