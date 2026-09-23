# Measure Fox's recovery moves in the sim: displacement and duration of double
# jump, Illusion, Fire Fox (up / sideways / diagonal), air dodge, drift and
# free fall, all from the same airborne start. Feeds ExPhil.Melee.Checkmate.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/measure_recovery_moves.exs
alias ExPhil.Sim.{Drill, Env}
n = Drill.neutral()
right = fn c -> %{c | main_stick: %{x: 1.0, y: 0.5}} end
{:ok, sim} = Env.start(:nif, stage: "final_destination", players: [%{character: "fox", costume: 0}, %{character: "fox", costume: 1}], batch_size: 1, seed: 7)
{:ok, _} = Env.reset(sim)
gs0 = Enum.reduce_while(1..400, nil, fn _, _ -> {:ok, [gs], _} = Env.step(sim, [[n, n]]); p = gs.players[1]; if p.action == 14 and p.on_ground, do: {:halt, gs}, else: {:cont, gs} end)
{:ok, blob} = Env.save(sim, 0)
IO.puts("start p1 x #{Float.round(gs0.players[1].x, 1)} y #{Float.round(gs0.players[1].y, 1)} action #{gs0.players[1].action}")

# Full hop then wait for the apex, save that airborne state as the common start.
{:ok, _} = Env.restore(sim, 0, blob)
apex =
  Enum.reduce_while(1..60, nil, fn t, prev ->
    c = if t == 1, do: %{n | button_y: true}, else: n
    {:ok, [gs], _} = Env.step(sim, [[c, n]])
    p = gs.players[1]
    if prev != nil and not p.on_ground and p.y < prev.y, do: {:halt, prev}, else: {:cont, p}
  end)
{:ok, air} = Env.save(sim, 0)
IO.puts("airborne start x #{Float.round(apex.x, 2)} y #{Float.round(apex.y, 2)} jumps #{apex.jumps_left}")

run = fn name, inputs ->
  {:ok, _} = Env.restore(sim, 0, air)
  {traj, _} =
    Enum.reduce(inputs, {[], :run}, fn c, {acc, _} ->
      {:ok, [gs], _} = Env.step(sim, [[c, n]])
      p = gs.players[1]
      {[{p.x, p.y, p.action, p.on_ground} | acc], :run}
    end)
  traj = Enum.reverse(traj)
  {x0, y0} = {apex.x, apex.y}
  land = Enum.find_index(traj, fn {_, _, _, g} -> g end)
  {mx, my, _, _} = Enum.max_by(traj, fn {_, y, _, _} -> y end)
  {lx, ly, la, _} = List.last(traj)
  actions = traj |> Enum.map(&elem(&1, 2)) |> Enum.dedup() |> Enum.take(8)
  IO.puts(String.pad_trailing(name, 26) <> "apex dx #{Float.round(mx - x0, 1)} dy #{Float.round(my - y0, 1)}  end dx #{Float.round(lx - x0, 1)} dy #{Float.round(ly - y0, 1)} (f#{length(traj)}, action #{la})" <> if(land, do: "  lands f#{land + 1}", else: "") <> "  actions #{inspect(actions)}")
  traj
end

hold = fn c, k -> List.duplicate(c, k) end
fall = run.("free fall", hold.(n, 90))
run.("drift right", hold.(right.(n), 90))
run.("double jump", [%{n | button_y: true}] ++ hold.(n, 89))
run.("double jump + drift", [%{right.(n) | button_y: true}] ++ hold.(right.(n), 89))
run.("illusion right", [%{right.(n) | button_b: true}] ++ hold.(n, 89))
run.("illusion + drift", [%{right.(n) | button_b: true}] ++ hold.(right.(n), 89))
run.("firefox up", [%{n | button_b: true, main_stick: %{x: 0.5, y: 1.0}}] ++ hold.(%{n | main_stick: %{x: 0.5, y: 1.0}}, 60) ++ hold.(n, 40))
run.("firefox right", [%{n | button_b: true, main_stick: %{x: 0.5, y: 1.0}}] ++ hold.(%{n | main_stick: %{x: 1.0, y: 0.5}}, 60) ++ hold.(n, 40))
run.("firefox up-right", [%{n | button_b: true, main_stick: %{x: 0.5, y: 1.0}}] ++ hold.(%{n | main_stick: %{x: 0.85, y: 0.85}}, 60) ++ hold.(n, 40))
run.("air dodge right", [%{n | button_l: true, l_shoulder: 1.0, main_stick: %{x: 1.0, y: 0.5}}] ++ hold.(n, 89))
run.("air dodge up-right", [%{n | button_l: true, l_shoulder: 1.0, main_stick: %{x: 0.85, y: 0.85}}] ++ hold.(n, 89))
# per-frame fall profile for gravity / terminal velocity
vys = fall |> Enum.map(&elem(&1, 1)) |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> Float.round(b - a, 3) end)
IO.puts("fall vy per frame: #{inspect(Enum.take(vys, 16))} ... terminal #{Enum.min(vys)}")
Env.stop(sim)

# Illusion shortening: a second B press during the dash cuts it; measure each press frame.
{:ok, sim} = Env.start(:nif, stage: "final_destination", players: [%{character: "fox", costume: 0}, %{character: "fox", costume: 1}], batch_size: 1, seed: 7)
{:ok, _} = Env.reset(sim)
Enum.reduce_while(1..400, nil, fn _, _ -> {:ok, [gs], _} = Env.step(sim, [[n, n]]); p = gs.players[1]; if p.action == 14 and p.on_ground, do: {:halt, gs}, else: {:cont, gs} end)
apex2 = Enum.reduce_while(1..60, nil, fn t, prev -> c = if t == 1, do: %{n | button_y: true}, else: n; {:ok, [gs], _} = Env.step(sim, [[c, n]]); p = gs.players[1]; if prev != nil and not p.on_ground and p.y < prev.y, do: {:halt, prev}, else: {:cont, p} end)
{:ok, air2} = Env.save(sim, 0)
for press <- [2, 3, 4, 5, 6, 8, 10] do
  {:ok, _} = Env.restore(sim, 0, air2)
  inputs = [%{right.(n) | button_b: true}] ++ List.duplicate(n, press - 2) ++ [%{n | button_b: true}] ++ List.duplicate(n, 60)
  {xs, acts} = Enum.reduce(inputs, {[], []}, fn c, {xs, as} -> {:ok, [gs], _} = Env.step(sim, [[c, n]]); p = gs.players[1]; {[p.x | xs], [p.action | as]} end)
  IO.puts("illusion, B again on frame #{press}: dx #{Float.round(Enum.max(xs) - apex2.x, 1)}  actions #{inspect(acts |> Enum.reverse() |> Enum.dedup() |> Enum.take(6))}")
end
Env.stop(sim)
