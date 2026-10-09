# Velocity conventions, sim vs replay: quantiles of speed_y_self /
# speed_air_x_self / speed_y_attack on the bot's offstage airborne states in
# a DAgger set (sim) vs the expert index's vy/vx columns (Peppi). Mix-free.
#   elixir -pa _build/dev/lib/*/ebin scripts/dagger_set_speed_check.exs SET [INDEX]
for app <- [:nx], do: Application.ensure_all_started(app)
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler, as: L

[path | rest] = System.argv()
set = path |> File.read!() |> :erlang.binary_to_term()
frames = set.frame_lists |> List.flatten() |> Enum.reject(&(&1[:input_only] == true))
q = fn l -> s = Enum.sort(l); [0.05, 0.25, 0.5, 0.75, 0.95] |> Enum.map(fn x -> "q#{trunc(x * 100)} #{Float.round(Enum.at(s, min(length(s) - 1, trunc(x * length(s)))) * 1.0, 2)}" end) |> Enum.join("  ") end
ps = Enum.map(frames, & &1.game_state.players[1])
IO.puts("RESULT sim (set) n=#{length(ps)}  speed_y_self: #{q.(Enum.map(ps, &(&1.speed_y_self || 0.0)))}")
IO.puts("RESULT sim (set)  speed_air_x_self: #{q.(Enum.map(ps, &(&1.speed_air_x_self || 0.0)))}")
IO.puts("RESULT sim (set)  speed_y_attack: #{q.(Enum.map(ps, &(&1.speed_y_attack || 0.0)))}   speed_x_attack: #{q.(Enum.map(ps, &(&1.speed_x_attack || 0.0)))}")
IO.puts("RESULT sim (set)  y: #{q.(Enum.map(ps, &(&1.y || 0.0)))}   hitstun>0 share #{Float.round(Enum.count(ps, &((&1.hitstun_frames_left || 0) > 0)) / length(ps), 3)}")

index_path = List.first(rest) || "data/silent_fall/expert_recovery_index.bin"
index = L.load(index_path)
names = L.dim_names()
cols = index.x |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_list() |> Enum.zip_with(& &1) |> Enum.zip(names) |> Map.new(fn {c, n} -> {n, c} end)
# undo scale/weight: vy = col * 2.0 / 1.0, vx = col * 2.0 / 1.0 (toward frame)
IO.puts("RESULT expert index n=#{index.n}  vy (speed_y_self): #{q.(Enum.map(cols[:vy], &(&1 * 2.0)))}")
IO.puts("RESULT expert index  vx (speed_air_x_self*toward): #{q.(Enum.map(cols[:vx], &(&1 * 2.0)))}")
IO.puts("RESULT expert index  y: #{q.(Enum.map(cols[:y], &(&1 * 40.0)))}   hitstun share #{Float.round(Enum.count(cols[:hitstun], &(&1 > 0)) / index.n, 3)}")
