# Jump behavior stats for pathology #4 (jump -> double-jump metronome loop).
# Experiment A of the pre-registered ladder: is jump the modal action in the
# data (label prior), and how does the policy's closed-loop behavior compare?
#
#   mix run scripts/jump_stats.exs <replay.slp> [more.slp ...]
#
# Per replay, for port 1 (bot in probes; recorder in fixtures):
#   - jump initiations per 100 grounded frames (jumpsquat entries)
#   - %% of airborne stints (from a ground jump) containing a double jump,
#     and median frames from liftoff to DJ
#   - jump-button (X/Y) press-duration histogram — SH needs a <=4f release
#     inside Mewtwo's 5f jumpsquat; p50 tells us if SH is even in the data

alias ExPhil.Interp.ReplayStats

jumpsquat = 24
rising = MapSet.new([25, 26])
dj_states = MapSet.new([27, 28])
airborne_states = MapSet.new(25..34)

press_runs = fn controllers, key ->
  controllers
  |> Enum.map(fn c -> if c && Map.get(c, key), do: 1, else: 0 end)
  |> Enum.chunk_by(& &1)
  |> Enum.filter(&(hd(&1) == 1))
  |> Enum.map(&length/1)
end

pct = fn xs, p ->
  case Enum.sort(xs) do
    [] -> nil
    s -> Enum.at(s, min(length(s) - 1, floor(p * length(s))))
  end
end

for path <- System.argv() do
  data = ReplayStats.load(path)
  actions = data.p1.actions
  controllers = data.p1.controllers

  grounded = Enum.count(actions, &(&1 not in 24..34))
  squats = actions |> Enum.chunk_every(2, 1, :discard)
           |> Enum.count(fn [a, b] -> a != jumpsquat and b == jumpsquat end)

  # walk each airborne stint that starts with a ground jump (25/26)
  stints =
    actions
    |> Enum.chunk_by(&MapSet.member?(airborne_states, &1))
    |> Enum.filter(fn chunk -> MapSet.member?(rising, hd(chunk)) end)

  dj_stints =
    Enum.map(stints, fn chunk ->
      Enum.find_index(chunk, &MapSet.member?(dj_states, &1))
    end)

  n_stints = length(stints)
  n_dj = Enum.count(dj_stints, & &1)
  dj_delays = Enum.filter(dj_stints, & &1)

  xy = press_runs.(controllers, :button_y) ++ press_runs.(controllers, :button_x)

  IO.puts("\n== #{Path.basename(path)} (#{data.n} frames, port 1)")
  IO.puts("  jumpsquat entries: #{squats}  (#{Float.round(squats * 100 / max(grounded, 1), 2)} per 100 grounded frames)")
  IO.puts("  air stints from ground jump: #{n_stints}; with DJ: #{n_dj} (#{Float.round(n_dj * 100 / max(n_stints, 1), 1)}%)")
  IO.puts("  frames liftoff->DJ: p50=#{pct.(dj_delays, 0.5)} p90=#{pct.(dj_delays, 0.9)}")
  IO.puts("  X/Y press durations: n=#{length(xy)} p50=#{pct.(xy, 0.5)}f p90=#{pct.(xy, 0.9)}f  (<=4f = short-hop capable)")
end
