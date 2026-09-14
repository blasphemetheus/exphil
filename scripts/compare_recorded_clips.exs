# Semantic equality of two recorded-clip envelopes (.frames): frame lists and
# validation JSON. Byte equality of term_to_binary output is NOT stable across
# exports (map/compression ordering), so compare with this, never with cmp.
#   mix run --no-start scripts/compare_recorded_clips.exs A.frames B.frames

[a, b] = System.argv()
load = fn p -> p |> File.read!() |> :erlang.binary_to_term() end
ea = load.(a); eb = load.(b)
IO.puts("keys equal: #{Map.keys(ea) == Map.keys(eb)}; validation json equal: #{ea.teacher_validation_json == eb.teacher_validation_json}")
[la] = ea.frame_lists; [lb] = eb.frame_lists
IO.puts("frames: #{length(la)} vs #{length(lb)}; lists equal: #{la == lb}")
if la != lb do
  Enum.zip(la, lb) |> Enum.with_index() |> Enum.reject(fn {{x, y}, _} -> x == y end) |> Enum.take(2)
  |> Enum.each(fn {{x, y}, i} ->
    IO.puts("frame #{i} differs; top keys differing: #{inspect(Enum.filter(Map.keys(x), &(x[&1] != y[&1])))}")
    for k <- Map.keys(x), x[k] != y[k], k == :game_state do
      gx = x.game_state; gy = y.game_state
      IO.puts("  game_state keys differing: #{inspect(Enum.filter(Map.keys(gx), &(Map.get(gx, &1) != Map.get(gy, &1))))}")
      for p <- [1, 2], gx.players[p] != gy.players[p] do
        px = gx.players[p]; py = gy.players[p]
        IO.puts("  player #{p}: #{inspect(Enum.map(Enum.filter(Map.keys(px), &(Map.get(px, &1) != Map.get(py, &1))), &{&1, Map.get(px, &1), Map.get(py, &1)}))}")
      end
    end
  end)
end
if ea.teacher_validation_json != eb.teacher_validation_json, do: IO.puts("A json: #{String.slice(ea.teacher_validation_json, 0, 200)}\nB json: #{String.slice(eb.teacher_validation_json, 0, 200)}")
