# Hunt for non-finite values in the raw fields of a replay corpus — a single
# corrupt .slp (garbage floats in position/speed/percent) NaNs the embedding
# batch it lands in and kills training within a few batches.
#
#   mix run scratchpad_scan_nonfinite.exs <replays_dir> <max_files>

[dir, max_files] = System.argv()
max_files = String.to_integer(max_files)

files =
  dir
  |> Path.expand()
  |> Path.join("*.slp")
  |> Path.wildcard()
  |> Enum.sort()
  |> Enum.take(max_files)

IO.puts("scanning #{length(files)} files for non-finite player fields...")

bad? = fn v -> is_float(v) and (v != v or abs(v) > 1.0e10) end

check_player = fn p ->
  [p.x, p.y, p.percent, p.speed_air_x_self, p.speed_ground_x_self,
   p.speed_y_self, p.speed_x_attack, p.speed_y_attack, p.shield_strength]
  |> Enum.filter(&bad?.(&1))
end

for {path, idx} <- Enum.with_index(files) do
  if rem(idx, 50) == 0, do: IO.puts("  #{idx}/#{length(files)}")

  case ExPhil.Data.Peppi.parse(path) do
    {:ok, replay} ->
      frames =
        try do
          ExPhil.Data.Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)
        rescue
          e ->
            IO.puts("PARSE-FRAMES FAIL: #{Path.basename(path)} — #{Exception.message(e)}")
            []
        end

      bad_frames =
        frames
        |> Enum.with_index()
        |> Enum.filter(fn {f, _i} ->
          Enum.any?(Map.values(f.game_state.players || %{}), fn p ->
            p && check_player.(p) != []
          end)
        end)
        |> Enum.take(3)

      unless bad_frames == [] do
        {f, i} = hd(bad_frames)
        sample =
          f.game_state.players
          |> Enum.flat_map(fn {port, p} -> if p, do: [{port, check_player.(p)}], else: [] end)
          |> Enum.reject(fn {_, bads} -> bads == [] end)

        IO.puts("NONFINITE: #{Path.basename(path)} frame_idx=#{i} (#{length(bad_frames)}+ frames) #{inspect(sample)}")
      end

    {:error, reason} ->
      IO.puts("PARSE FAIL: #{Path.basename(path)} — #{inspect(reason)}")
  end
end

IO.puts("scan done")
