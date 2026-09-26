# Two distinct exported policies, balanced ports, FD development matches.
# mix run scripts/sim_policy_match.exs --a POLICY --b POLICY --games 4 --out DIR
alias ExPhil.Agents.Agent
alias ExPhil.Bridge.SimPort
alias ExPhil.Training.{Checkpoint, Output}
{opts, _, bad} = OptionParser.parse(System.argv(), strict: [a: :string, b: :string,
  games: :integer, frames: :integer, out: :string, seed: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")
out = Keyword.fetch!(opts, :out)
File.mkdir_p!(out)
Output.banner("Two-policy Fox match (FD development evaluation)")
agents = for key <- [:a, :b], into: %{} do
  path = Keyword.fetch!(opts, key)
  {:ok, export} = Checkpoint.load_policy(path)
  contract = ExPhil.Networks.Policy.ExecutionContract.load(export.config)
  {:ok, agent} = Agent.start_link(policy_path: path, deterministic: false,
    temperature: 1.0, af_convention: :parsed, frame_delay: 0, reaction_delay: 0,
    harness: :sync_runner, stateful_step: contract.recurrent_state == :carried_zero)
  {:ok, _} = Agent.warmup(agent)
  {key, agent}
end
players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]
seed = opts[:seed] || 905
{:ok, sim} = SimPort.start_link(stage: "final_destination", players: players, length: 256, seed: seed)
games = opts[:games] || 4
if rem(games, 2) != 0, do: raise("use an even game count for balanced ports")
rows = for game <- 1..games do
  # Same seed within each port-swapped pair.
  {:ok, _} = SimPort.reinit(sim, %{stage: "final_destination", players: players,
    length: 256, seed: seed + div(game - 1, 2)})
  Enum.each(agents, fn {_, a} -> Agent.reset_buffer(a) end)
  a_port = if rem(game, 2) == 1, do: 1, else: 2
  b_port = 3 - a_port
  by_port = %{a_port => agents.a, b_port => agents.b}
  {:ok, [initial]} = SimPort.frames(sim)
  final = Enum.reduce_while(1..(opts[:frames] || 18000), initial, fn _, gs ->
    controllers = for port <- [1, 2] do
      {:ok, controller} = Agent.get_controller(by_port[port], %{gs | own_port: port}, player_port: port)
      controller
    end
    {:ok, [next], [term]} = SimPort.step(sim, [controllers])
    if term["done"] == 1, do: {:halt, next}, else: {:cont, next}
  end)
  a = final.players[a_port]
  b = final.players[b_port]
  row = %{game: game, a_port: a_port, seed: seed + div(game - 1, 2),
    a_policy: opts[:a], b_policy: opts[:b], frame: final.frame,
    a_stock: a.stock, b_stock: b.stock, a_percent: a.percent, b_percent: b.percent,
    outcome: cond do
      a.stock == 0 and b.stock > 0 -> "b_win"
      b.stock == 0 and a.stock > 0 -> "a_win"
      true -> "unfinished_or_draw"
    end}
  File.write!(Path.join(out, "games.jsonl"), Jason.encode!(row) <> "\n", [:append])
  Output.puts(inspect(row))
  row
end
SimPort.stop(sim)
Enum.each(agents, fn {_, a} -> GenServer.stop(a) end)
File.write!(Path.join(out, "summary.json"), Jason.encode!(rows, pretty: true))
Output.success("Match records: #{out}")
