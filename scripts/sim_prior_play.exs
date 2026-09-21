# SIM_INTEGRATION.md step 4 (R1): the imitation prior plays INSIDE the sim.
#
# Two Agents (the policy vs a frozen copy of itself, or vs an idle dummy)
# drive a Fox ditto on FD through ExPhil.Bridge.SimPort with the same
# closed-loop contract as the Dolphin sync runner (reaction 0, live AF,
# stateful step, T=1.0): the controller decided on frame F is applied on
# the step that produces F+1 (latency 1, like Dolphin `--frame-delay 0`).
# Every game is fingerprinted with the human pipeline and written as one
# JSON row per port, in the same shape as scripts/style_fingerprint.exs,
# so the D3/S6 comparison scripts read it unchanged.
#
#   devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest \
#     mix run scripts/sim_prior_play.exs --policy checkpoints/.../model_best_policy.bin \
#       --games 10 --frames 1800 --out eval_runs/0921_sim_r1/anon [--opponent self|idle] [--seed 7]
#       [--style-tag C2 --player-registry .../model_best_players.json]

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.{ControllerState, SimPort}
alias ExPhil.Interp.StyleFingerprint
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, games: :integer, frames: :integer, out: :string, opponent: :string, seed: :integer,
             style_tag: :string, player_registry: :string, temperature: :float, af: :string]
  )

policy = opts[:policy] || raise("--policy required")
games = opts[:games] || 10
post_frames = opts[:frames] || 1800
out = opts[:out] || raise("--out DIR required")
opponent = String.to_atom(opts[:opponent] || "self")
seed0 = opts[:seed] || 7
temperature = opts[:temperature] || 1.0
File.mkdir_p!(out)

Output.banner("Prior in the sim (R1)")
Output.config([{"Policy", policy}, {"Games", games}, {"Post-zero frames", post_frames}, {"Opponent", opponent}, {"Style tag", opts[:style_tag] || "anonymous"}, {"Out", out}])

style_opts =
  case {opts[:style_tag], opts[:player_registry]} do
    {nil, _} -> []
    {tag, reg} when is_binary(reg) ->
      {:ok, registry} = ExPhil.Training.PlayerRegistry.from_json(reg)
      id = ExPhil.Training.PlayerRegistry.get_id(registry, tag) || raise("style tag #{tag} not in registry")
      Output.puts("Style: #{tag} (id #{id})")
      [style_id: id]
    _ -> raise("--style-tag needs --player-registry")
  end

agent_opts = fn ->
  [policy_path: policy, deterministic: false, temperature: temperature, af_convention: String.to_atom(opts[:af] || "parsed"),
   frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true] ++ style_opts
end

Output.step(1, 3, "Loading agents")
{:ok, agent_a} = Agent.start_link(agent_opts.())
{:ok, _} = Agent.warmup(agent_a)
agent_b = if opponent == :self, do: (fn -> {:ok, b} = Agent.start_link(Keyword.delete(agent_opts.(), :style_id)); {:ok, _} = Agent.warmup(b); b end).(), else: nil

neutral = %ControllerState{
  main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, l_shoulder: 0.0, r_shoulder: 0.0,
  button_a: false, button_b: false, button_x: false, button_y: false, button_z: false,
  button_l: false, button_r: false, button_d_up: false
}

# Dolphin probe geometry: bot on port 1 (costume 1), opponent port 2 (costume 0).
Output.step(2, 3, "Starting sim")
{:ok, sim} = SimPort.start_link(stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], length: 256, seed: seed0)

io = File.open!(Path.join(out, "sim_fingerprints.jsonl"), [:write, :utf8])
summary_io = File.open!(Path.join(out, "games.jsonl"), [:write, :utf8])

Output.step(3, 3, "Playing #{games} games")

for g <- 1..games do
  if g > 1 do
    {:ok, _} = SimPort.reinit(sim, %{stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], length: 256, seed: seed0 + g - 1})
    Agent.reset_buffer(agent_a)
    if agent_b, do: Agent.reset_buffer(agent_b)
  end

  {:ok, [gs]} = SimPort.frames(sim)
  t0 = System.monotonic_time(:millisecond)
  last_frame = post_frames - 123

  {states, ctrls, errors, final} =
    Enum.reduce_while(Stream.iterate(gs, & &1), {[], [], 0, gs}, fn _, {st, ct, errs, gs} ->
      c1 =
        case Agent.get_controller(agent_a, gs, player_port: 1) do
          {:ok, c} -> c
          {:error, _} -> nil
        end

      c2 =
        case agent_b && Agent.get_controller(agent_b, %{gs | own_port: 2}, player_port: 2) do
          {:ok, c} -> c
          _ -> neutral
        end

      errs = if c1 == nil, do: errs + 1, else: errs
      row = [c1 || neutral, c2]

      case SimPort.step(sim, [row]) do
        {:ok, [next], [term]} ->
          done = term["done"] == 1 or next.frame >= last_frame
          acc = {[gs | st], [{c1 || neutral, c2} | ct], errs, next}
          if done, do: {:halt, acc}, else: {:cont, acc}

        {:error, reason} ->
          Output.error("game #{g}: #{inspect(reason)}")
          {:halt, {st, ct, errs + 1, gs}}
      end
    end)

  states = Enum.reverse(states)
  ctrls = Enum.reverse(ctrls)
  ms = System.monotonic_time(:millisecond) - t0

  in_game = Enum.zip(states, ctrls) |> Enum.reject(fn {s, _} -> s.frame < 0 end)
  {ig_states, ig_pairs} = Enum.unzip(in_game)
  ig_ctrls = Enum.map(ig_pairs, &elem(&1, 0))
  ig_ctrls2 = Enum.map(ig_pairs, &elem(&1, 1))

  p1 = final.players[1]
  p2 = final.players[2]
  actions = ig_states |> Enum.map(& &1.players[1].action) |> Enum.frequencies() |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(6)
  Output.puts("game #{g}: #{length(states)} frames in #{ms} ms (#{Float.round(length(states) / max(ms, 1) * 1000, 0)} fps), agent errors #{errors}; p1 stocks #{p1.stock} #{Float.round(p1.percent, 0)}% | p2 stocks #{p2.stock} #{Float.round(p2.percent, 0)}%; top actions #{inspect(actions)}")

  IO.write(summary_io, Jason.encode!(%{game: g, frames: length(states), ms: ms, errors: errors, p1_stocks: p1.stock, p1_percent: p1.percent, p2_stocks: p2.stock, p2_percent: p2.percent, top_actions: Enum.map(actions, &Tuple.to_list/1)}) <> "\n")

  # Per-frame trace (frame, p1/p2 pos+action, p1 inputs) for debugging the loop.
  trace_io = File.open!(Path.join(out, "trace_game#{g}.tsv"), [:write, :utf8])
  IO.write(trace_io, "frame\tp1_x\tp1_y\tp1_action\tp1_af\tp1_stock\tp1_pct\tp2_x\tp2_y\tp2_action\tp2_stock\tmx\tmy\tcx\tcy\tA\tB\tX\tY\tZ\tL\tR\tsh\n")
  Enum.zip(states, ctrls)
  |> Enum.each(fn {s, {c, _}} ->
    a = s.players[1]; b = s.players[2]
    IO.write(trace_io, Enum.join([s.frame, Float.round(a.x, 2), Float.round(a.y, 2), a.action, a.action_frame, a.stock, round(a.percent), Float.round(b.x, 2), Float.round(b.y, 2), b.action, b.stock,
      Float.round(c.main_stick.x, 2), Float.round(c.main_stick.y, 2), Float.round(c.c_stick.x, 2), Float.round(c.c_stick.y, 2),
      (if c.button_a, do: 1, else: 0), (if c.button_b, do: 1, else: 0), (if c.button_x, do: 1, else: 0), (if c.button_y, do: 1, else: 0), (if c.button_z, do: 1, else: 0), (if c.button_l, do: 1, else: 0), (if c.button_r, do: 1, else: 0), Float.round(max(c.l_shoulder || 0.0, c.r_shoulder || 0.0), 2)], "\t") <> "\n")
  end)
  File.close(trace_io)

  if length(ig_states) >= 600 do
    for {port, ctl} <- [{1, ig_ctrls}, {2, ig_ctrls2}] do
      fp = StyleFingerprint.fingerprint(ig_states, port, ctl)
      row = %{path: "#{out}/game#{g}", tag: nil, port: port, character: "Fox", ditto: true, candidates: [], started_at: DateTime.utc_now() |> DateTime.to_iso8601(), costume: if(port == 1, do: 1, else: 0), player_type: "sim_agent", opponent_character: "Fox", stage: 32, features: fp |> Map.new(fn {k, v} -> {k, (if is_tuple(v), do: Tuple.to_list(v), else: v)} end)}
      IO.write(io, Jason.encode!(row) <> "\n")
    end
  end
end

File.close(io)
File.close(summary_io)
SimPort.stop(sim)
Output.success("wrote #{out}/sim_fingerprints.jsonl and games.jsonl")
