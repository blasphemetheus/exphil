# One-line multishine scoreboard for a replay: how many jump-cancels,
# shines-from-jumpsquat, and empty hops did the (port 1) player perform?
#
#   mix run scripts/trace_multishine.exs path/to/Game.slp [more.slp ...]
#
# Used standalone and by scripts/dagger_loop.sh to track live progress
# across DAgger iterations.

paths = System.argv()

if paths == [] do
  IO.puts(:stderr, "usage: mix run scripts/trace_multishine.exs <replay.slp> [...]")
  System.halt(1)
end

for path <- paths do
  {:ok, replay} = ExPhil.Data.Peppi.parse(path)

  rows =
    replay
    |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> Enum.map(fn f ->
      p = f.game_state.players[1]
      {trunc(p.action), p.on_ground}
    end)

  pairs = Enum.chunk_every(rows, 2, 1, :discard)

  # Reflector family 360..368, jumpsquat 24, aerial jump 25
  jc = Enum.count(pairs, fn [{a, _}, {b, _}] -> a in 360..368 and b == 24 end)
  shine = Enum.count(pairs, fn [{a, _}, {b, _}] -> a == 24 and b in [360, 365] end)
  empty_hop = Enum.count(pairs, fn [{a, _}, {b, _}] -> a == 24 and b == 25 end)
  frames = length(rows)
  rate = if frames > 0, do: Float.round(jc * 1000 / frames, 1), else: 0.0

  IO.puts(
    "#{Path.basename(path)}: #{frames}f | JC=#{jc} shine=#{shine} empty_hop=#{empty_hop} | #{rate} JC/1k frames"
  )
end
