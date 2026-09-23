# TAS a recovery from a replay position (ExPhil.Sim.RecoveryProbe).
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/recovery_probe.exs PATH.slp --frame 12338 --port 1
alias ExPhil.Sim.{Env, RecoveryProbe, Seed}
{opts, [path], _} = OptionParser.parse(System.argv(), strict: [frame: :integer, port: :integer, horizon: :integer])
frame = opts[:frame] || raise("--frame")
port = opts[:port] || 1
{:ok, seed} = Seed.from_replay(path, frame: frame, warm: 0)
if seed.divergence, do: IO.puts("WARNING: replay diverged before f#{frame}: #{inspect(seed.divergence)}")
Env.stop(seed.sim)
save = List.last(seed.saves)
p0 = save.state.players[port]
IO.puts("f#{frame} port #{port}: x #{Float.round(p0.x, 1)} y #{Float.round(p0.y, 1)} action #{p0.action} jumps #{p0.jumps_left} hitstun #{p0.hitstun_frames_left} stock #{p0.stock}")
batch = 128
{:ok, sim} = Env.start(:nif, stage: seed.stage, players: seed.players, batch_size: batch, seed: 7, ucf_cardinals: 1)
{:ok, id} = Env.upload(sim, save.blob)
results = RecoveryProbe.run(sim, {:id, id}, port, p0, batch: batch, horizon: opts[:horizon] || 240)
made = Enum.filter(results, fn {_, o} -> RecoveryProbe.made_it?(o) end)
IO.puts("#{length(results)} plans\nmade it back: #{length(made)} of #{length(results)}")
for {name, o} <- Enum.take(made, 25), do: IO.puts("  #{name}: #{inspect(o)}")
IO.puts("outcomes: #{inspect(Enum.frequencies_by(results, fn {_, o} -> if o, do: elem(o, 0), else: :still_alive end))}")
Env.stop(sim)
