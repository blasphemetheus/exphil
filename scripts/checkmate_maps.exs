# Checkmate maps for Fox: for each stage and each combination of recovery
# resources (double jump, up-B, side-B, air dodge, wall jump), which grid
# cells are checkmate from rest (zero self and knockback velocity).
# Uses ExPhil.Melee.Checkmate with a coarser plan grid; mirrored at x = 0.
# A cell safe with fewer resources is safe with more, so each combo only
# searches the cells no subset already cleared.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/checkmate_maps.exs --out eval_runs/0923_checkmate/maps.json
alias ExPhil.Melee.Checkmate
{opts, _, _} = OptionParser.parse(System.argv(), strict: [out: :string, step: :float, stages: :string])
out = opts[:out] || "eval_runs/0923_checkmate/maps.json"
step = opts[:step] || 4.0
stages = (opts[:stages] || "31,32,3,2,28,8") |> String.split(",") |> Enum.map(&String.to_integer/1)
plan_opts = [delays: [0, 8, 20, 40], angles: Enum.map(0..11, &(&1 * 30.0)), dodge_angles: Enum.map(0..7, &(&1 * 45.0))]
resources = [:jump, :up_b, :side_b, :air_dodge, :wall_jump]
combos = for bits <- 0..31, do: Enum.with_index(resources) |> Enum.filter(fn {_, i} -> Bitwise.band(bits, Bitwise.bsl(1, i)) != 0 end) |> Enum.map(&elem(&1, 0))
combos = Enum.sort_by(combos, &length/1)
key = fn combo -> Enum.map_join(resources, "", &if(&1 in combo, do: "1", else: "0")) end
# cells inside the stage's solid collision shell are skipped and drawn as stage

result =
  for stage <- stages, into: %{} do
    geo = ExPhil.Situations.geometry(stage)
    {_bl, br, bt, bb} = geo.blast
    xs = for i <- 0..trunc(br / step), do: i * step + step / 2
    ys = for j <- 0..trunc((min(bt, 160) - bb) / step), do: bb + j * step + step / 2
    cells = for {y, j} <- Enum.with_index(ys), {x, i} <- Enum.with_index(xs), do: {i, j, x, y}
    solid? = fn x, y -> Checkmate.solid?(stage, x, y) end
    cg = Checkmate.geometry(stage)
    t0 = System.monotonic_time(:millisecond)

    {maps, _} =
      Enum.reduce(combos, {%{}, %{}}, fn combo, {maps, alive_by} ->
        subsets = for r <- combo, do: key.(combo -- [r])
        known_alive = subsets |> Enum.map(&Map.get(alive_by, &1, MapSet.new())) |> Enum.reduce(MapSet.new(), &MapSet.union/2)
        todo = Enum.reject(cells, fn {i, j, x, y} -> solid?.(x, y) or MapSet.member?(known_alive, {i, j}) end)
        state = fn x, y -> %{stage: stage, x: x, y: y, jumps_left: if(:jump in combo, do: 1, else: 0), up_b: :up_b in combo, side_b: :side_b in combo, air_dodge: :air_dodge in combo, wall_jump: :wall_jump in combo} end
        dead =
          todo
          |> Task.async_stream(fn {i, j, x, y} -> {{i, j}, Checkmate.checkmate?(state.(x, y), plan_opts)} end, max_concurrency: System.schedulers_online(), ordered: false, timeout: :infinity)
          |> Enum.flat_map(fn {:ok, {ij, d}} -> if d, do: [ij], else: [] end)
          |> MapSet.new()
        alive = MapSet.union(known_alive, cells |> Enum.reject(fn {i, j, x, y} -> solid?.(x, y) end) |> Enum.map(fn {i, j, _, _} -> {i, j} end) |> MapSet.new() |> MapSet.difference(dead))
        cell_char = fn i, j, x, y -> cond do
          solid?.(x, y) -> "2"
          MapSet.member?(dead, {i, j}) -> "1"
          true -> "0"
        end end
        bits = for {i, j, x, y} <- cells, into: "", do: cell_char.(i, j, x, y)
        IO.puts("stage #{stage} #{key.(combo)}: searched #{length(todo)}, dead #{MapSet.size(dead)}")
        {Map.put(maps, key.(combo), bits), Map.put(alive_by, key.(combo), alive)}
      end)

    IO.puts("stage #{stage} done in #{div(System.monotonic_time(:millisecond) - t0, 1000)} s")
    {stage, %{edge: geo.edge, blast: Tuple.to_list(geo.blast), platforms: Enum.map(geo.platforms, &Tuple.to_list/1), loops: cg.loops, ledges: Enum.map(cg.ledges, fn {x, y, _} -> [x, y] end), step: step, nx: length(xs), ny: length(ys), x0: 0.0, y0: bb, maps: maps}}
  end

File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(%{resources: resources, combos: Enum.map(combos, key), stages: result}))
IO.puts("wrote #{out}")
