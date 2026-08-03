# CycleSim gate diagnostic (task #5): WHY does the champion never
# aerial-shine in simulation?
#
# Compares, at matched {action, af} airborne states:
#   sim:  B/X logits + the frame embedding CycleSim synthesized
#   live: B/X logits + the embedding of the real replay frame
# and prints the top embedding dims by |sim - live| — the input signal
# the synthetic frames get wrong is what suppresses the aerial B.
#
#   mix run scripts/cyclesim_diag.exs [--policy checkpoints/ms_open_z.bin]
#     [--replay eval_runs/0728_open_z_idle/r1.slp] [--max-frames 300]

alias ExPhil.Interp.{Activations, CycleSim}
alias ExPhil.Training.{Data, Output}
alias ExPhil.Data.Peppi

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replay: :string, max_frames: :integer]
  )

policy = opts[:policy] || "checkpoints/ms_open_z.bin"
replay = opts[:replay] || "eval_runs/0728_open_z_idle/r1.slp"
max_frames = opts[:max_frames] || 300

Output.banner("CycleSim diagnostic: sim vs live at matched states")

graph = Path.wildcard("eval_runs/0728_open_z_idle*/r*.slp")
{entry, table} = CycleSim.from_fixture("test/fixtures/replays/fox_multishine_closed.slp", graph_replays: graph)
loaded = Activations.load_heads(policy)

sim = CycleSim.rollout(loaded.predict_fn, loaded.params, entry, table, max_frames: max_frames, trace: true)
Output.puts("sim: #{sim.frames} frames, chains=#{inspect(Enum.take(sim.chains, 5))}, soft=#{sim.soft}")

# Live side: embeddings + logits per frame of the real replay
{:ok, parsed} = Peppi.parse(replay)

frames =
  parsed
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))

ds = frames |> Data.from_frames() |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)
emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
{total, _} = Nx.shape(emb)
window = loaded.window
frames_arr = List.to_tuple(frames)

live_logits = fn t ->
  win = Nx.slice_along_axis(emb, t - window + 1, window, axis: 0)
  out = loaded.predict_fn.(loaded.params, Nx.new_axis(win, 0))
  buttons = out |> elem(0) |> Nx.squeeze() |> Nx.to_flat_list()
  {Enum.at(buttons, 1), Enum.at(buttons, 2)}
end

# Matched states: the airborne stretch (whatever action the sim dwells in
# longest while airborne) at afs 1..25
airborne_action =
  sim.trace
  |> Enum.frequencies_by(& &1.action)
  |> Enum.max_by(fn {a, n} -> if Map.get(table.grounded, a, true), do: -1, else: n end)
  |> elem(0)

Output.puts("airborne dwell action: #{airborne_action}")

sim_by_af = sim.trace |> Enum.filter(&(&1.action == airborne_action)) |> Map.new(&{&1.af, &1})

live_rows =
  for t <- (window - 1)..(total - 1),
      p = elem(frames_arr, t).game_state.players[1],
      p.action == airborne_action,
      do: {p.action_frame, t}

live_by_af = Map.new(Enum.reverse(live_rows))

for af <- Enum.sort(Map.keys(sim_by_af)), Map.has_key?(live_by_af, af), rem(af, 3) == 1 do
  s = sim_by_af[af]
  t = live_by_af[af]
  {lb, lx} = live_logits.(t)

  live_emb = emb[t]
  diff = Nx.abs(Nx.subtract(s.emb, live_emb))
  top = diff |> Nx.argsort(direction: :desc) |> Nx.slice_along_axis(0, 8, axis: 0) |> Nx.to_flat_list()
  top_str = Enum.map_join(top, " ", fn d ->
    "#{d}:#{Float.round(Nx.to_number(diff[d]), 2)}"
  end)

  Output.puts(
    "af#{String.pad_leading(to_string(af), 3)}  B sim=#{Float.round(s.b_logit, 2)} live=#{Float.round(lb, 2)}  " <>
      "X sim=#{Float.round(s.x_logit, 2)} live=#{Float.round(lx, 2)}  top-dims #{top_str}"
  )
end
