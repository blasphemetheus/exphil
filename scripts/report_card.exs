# Tier-1 report card (task #27): mechanics-health gates for a policy replay.
# Composes the pathology detectors proven on 2026-07-15 (jump discipline,
# shield discipline, idle deadlock, press durations) into one PASS/FAIL
# readout per replay. Gates calibrated from the user's fixtures with
# headroom; see docs/planning/INTERP_ROADMAP.md pathology #4 entries.
#
#   mix run scripts/report_card.exs <replay.slp> [more.slp ...]
#
# Port 1 is assumed to be the policy.

alias ExPhil.Interp.ReplayStats

jumpsquat = 24
rising = MapSet.new([25, 26])
dj_states = MapSet.new([27, 28])
airborne_states = MapSet.new(25..34)
shield_states = MapSet.new(178..182)
# Post-shield options (approximate ID sets; good enough for trend gates)
oos_jump = MapSet.new(24..28)
oos_escape = MapSet.new(233..235)
oos_grab = MapSet.new([212, 213])

pct = fn xs, p ->
  case Enum.sort(xs) do
    [] -> nil
    s -> Enum.at(s, min(length(s) - 1, floor(p * length(s))))
  end
end

gate = fn name, value, ok?, target ->
  status = if ok?, do: "PASS", else: "FAIL"
  vs = if is_float(value), do: Float.round(value, 2), else: value
  IO.puts("  [#{status}] #{String.pad_trailing(name, 26)} #{inspect(vs)}  (target #{target})")
  ok?
end

for path <- System.argv() do
  data = ReplayStats.load(path)
  actions = data.p1.actions
  controllers = data.p1.controllers
  n = data.n

  # -- jump discipline
  grounded = Enum.count(actions, &(&1 not in 24..34))
  squats = actions |> Enum.chunk_every(2, 1, :discard)
           |> Enum.count(fn [a, b] -> a != jumpsquat and b == jumpsquat end)
  jump_rate = squats * 100 / max(grounded, 1)

  stints =
    actions
    |> Enum.chunk_by(&MapSet.member?(airborne_states, &1))
    |> Enum.filter(fn chunk -> MapSet.member?(rising, hd(chunk)) end)

  dj_idx = Enum.map(stints, fn c -> Enum.find_index(c, &MapSet.member?(dj_states, &1)) end)
  dj_pct = Enum.count(dj_idx, & &1) * 100 / max(length(stints), 1)
  dj_p50 = pct.(Enum.filter(dj_idx, & &1), 0.5)

  # -- shield discipline + OOS follow-up
  shield = ReplayStats.shield_stats(actions)

  oos =
    actions
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.with_index()
    |> Enum.filter(fn {[a, b], _} ->
      MapSet.member?(shield_states, a) and not MapSet.member?(shield_states, b)
    end)
    |> Enum.map(fn {_, i} ->
      window = Enum.slice(actions, i + 1, 20)

      cond do
        Enum.any?(window, &MapSet.member?(oos_jump, &1)) -> :jump
        Enum.any?(window, &MapSet.member?(oos_escape, &1)) -> :escape
        Enum.any?(window, &MapSet.member?(oos_grab, &1)) -> :grab
        true -> :idle
      end
    end)
    |> Enum.frequencies()

  oos_total = oos |> Map.values() |> Enum.sum()
  oos_idle_pct = Map.get(oos, :idle, 0) * 100 / max(oos_total, 1)

  # -- idle deadlock
  idle_runs = ReplayStats.run_lengths(actions, [14])
  max_idle = Enum.max(idle_runs, fn -> 0 end)

  # -- press durations (short-hop capability)
  press_runs = fn key ->
    controllers
    |> Enum.map(fn c -> if c && Map.get(c, key), do: 1, else: 0 end)
    |> Enum.chunk_by(& &1)
    |> Enum.filter(&(hd(&1) == 1))
    |> Enum.map(&length/1)
  end

  xy = press_runs.(:button_x) ++ press_runs.(:button_y)
  xy_p50 = pct.(xy, 0.5)

  IO.puts("\n== REPORT CARD: #{Path.basename(path)} (#{n} frames)")

  results = [
    gate.("jumps/100 grounded", jump_rate, jump_rate <= 2.5, "<=2.5; user 1.0-1.5"),
    gate.("DJ per air stint %", dj_pct, dj_pct <= 35.0, "<=35; user 13-21"),
    gate.("liftoff->DJ p50 (f)", dj_p50, dj_p50 == nil or dj_p50 >= 8, ">=8; user 12-16"),
    gate.("shield occupancy %", shield.shield_frac * 100, shield.shield_frac <= 0.04, "<=4"),
    gate.("shield run p95 (f)", shield.p95 || 0, (shield.p95 || 0) <= 180, "<=180"),
    gate.("OOS idle %", oos_idle_pct, oos_idle_pct <= 50.0, "<=50"),
    gate.("max idle streak (f)", max_idle, max_idle <= 300, "<=300 (5s; deadlock detector)"),
    gate.("X/Y press p50 (f)", xy_p50, xy_p50 == nil or xy_p50 <= 6, "<=6 (SH-capable)")
  ]

  passed = Enum.count(results, & &1)
  IO.puts("  SCORE: #{passed}/#{length(results)} gates" <>
            if(passed == length(results), do: "  ✓ ALL PASS", else: ""))
end
