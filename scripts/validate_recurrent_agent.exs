# Offline Agent-level check with a real window-trained checkpoint and replay.
# mix run scripts/validate_recurrent_agent.exs --policy PATH --replay PATH \
#   --character mewtwo --out /tmp/agent-replay.json
alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replay: :string, character: :string, out: :string]
  )

policy = Keyword.fetch!(opts, :policy)
replay_path = Keyword.fetch!(opts, :replay)
{:ok, export} = ExPhil.Training.load_policy(policy)

if export.config[:bptt],
  do: raise("this probe compares window-trained policies; BPTT must not resync")

history = export.config.window_size + 4
{:ok, metadata} = Peppi.metadata(replay_path)
character = Keyword.get(opts, :character, "mewtwo")

own =
  Enum.find(metadata.players, &(String.downcase(&1.character_name || "") == character)) ||
    raise("replay has no #{character}")

opp = Enum.find(metadata.players, &(&1.port != own.port))
{:ok, replay} = Peppi.parse(replay_path, player_port: own.port)

frames =
  replay
  |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.take(history + 32)

unless length(frames) == history + 32, do: raise("replay too short for rollback/resync check")

{:ok, agent} =
  Agent.start_link(policy: export, deterministic: true, stateful_step: true, stateful_resync: 30)

{:ok, windowed} = Agent.start_link(policy: export, deterministic: true)
true = Agent.get_config(agent).stateful_step_active
{:ok, initial} = Agent.snapshot_state(agent)
{:ok, warmup_ms} = Agent.warmup(agent)
{:ok, ^initial} = Agent.snapshot_state(agent)

signature = fn action ->
  action
  |> Map.take([:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder])
  |> Map.new(fn {k, v} -> {k, Nx.to_binary(v)} end)
end

play = fn target, sequence ->
  Enum.map(sequence, fn f ->
    {:ok, action} = Agent.get_action(target, f.game_state)
    signature.(action)
  end)
end

true = play.(agent, Enum.take(frames, 1)) == play.(windowed, Enum.take(frames, 1))
play.(agent, Enum.slice(frames, 1, history - 1))
{:ok, saved} = Agent.snapshot_state(agent)
suffix = Enum.slice(frames, history, 32)
first = play.(agent, suffix)
{:ok, final} = Agent.snapshot_state(agent)
:ok = Agent.restore_state(agent, saved)
true = play.(agent, suffix) == first
{:ok, ^final} = Agent.snapshot_state(agent)
GenServer.stop(agent)
GenServer.stop(windowed)

result = %{
  policy: policy,
  replay: replay_path,
  frames: length(frames),
  client: EXLA.Client.default_name(),
  precision: Nx.Defn.default_options()[:precision],
  cold_action_matches_window: true,
  warmup_preserves_state: true,
  rollback_exact: true,
  warmup_ms: warmup_ms,
  stateful_resync: 30
}

out = Keyword.fetch!(opts, :out)
File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(result, pretty: true))
IO.inspect(result)
