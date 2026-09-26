# Reload exported policy and time the actual Agent API on consecutive Fox frames.
alias ExPhil.{Agents.Agent, Data.Peppi, Training.Output}
[policy, out] = System.argv()
Output.banner("Full Fox Agent latency")
path = File.stream!("replays/erickfm_ranked/v2_filtered/manifest.jsonl")
  |> Stream.map(&Jason.decode!/1) |> Enum.find(&(&1["verdict"] == "keep")) |> Map.fetch!("path")
{:ok, meta} = Peppi.metadata(path)
own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
opponent = Enum.find(meta.players, &(&1.port != own.port))
{:ok, replay} = Peppi.parse(path)
frames = Peppi.to_training_frames(replay, player_port: own.port, opponent_port: opponent.port)
  |> Enum.reject(&(&1.game_state.frame < 0)) |> Enum.take(500)
{:ok, agent} = Agent.start_link(policy_path: policy, deterministic: false,
  temperature: 1.0, af_convention: :parsed, frame_delay: 0,
  harness: :sync_runner, reaction_delay: 0, stateful_step: false)
{:ok, _} = Agent.warmup(agent)
times = Enum.map(frames, fn frame ->
  {us, {:ok, _controller}} = :timer.tc(fn ->
    Agent.get_controller(agent, frame.game_state, player_port: own.port)
  end)
  us / 1000
end) |> Enum.drop(100) |> Enum.sort()
row = %{policy: policy, samples: length(times), warmup_frames: 100,
  median_ms: Enum.at(times, div(length(times), 2)),
  p95_ms: Enum.at(times, ceil(length(times) * 0.95) - 1), max_ms: List.last(times),
  over_16_67_ms: Enum.count(times, &(&1 > 1000 / 60)),
  note: "Consecutive recorded states, full stochastic Agent API; no Dolphin transport or rendering."}
File.write!(out, Jason.encode!(row, pretty: true))
Output.success(inspect(row))

# Isolate live embedding and test CPU assembly + one device transfer.
cfg = ExPhil.Embeddings.config_for_source([stage_internals: true], Peppi.provides())
backend = Nx.default_backend()
embedding = Enum.map(Enum.take(frames, 100), fn frame ->
  {gpu_us, gpu} = :timer.tc(fn ->
    t = ExPhil.Embeddings.Game.embed(frame.game_state, nil, own.port, config: cfg)
    {t, Nx.to_flat_list(t)}
  end)
  {cpu_us, cpu} = :timer.tc(fn ->
    t = Nx.with_default_backend(Nx.BinaryBackend, fn ->
      ExPhil.Embeddings.Game.embed(frame.game_state, nil, own.port, config: cfg)
    end) |> Nx.backend_transfer(backend)
    {t, Nx.to_flat_list(t)}
  end)
  difference = Enum.zip(elem(gpu, 1), elem(cpu, 1))
    |> Enum.map(fn {a, b} -> abs(a-b) end) |> Enum.max()
  %{gpu_ms: gpu_us/1000, cpu_transfer_ms: cpu_us/1000, max_abs_diff: difference}
end)
File.write!(out <> ".embedding.json", Jason.encode!(embedding, pretty: true))
