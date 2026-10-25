# Full Agent resync/reset timing with a trained autoregressive checkpoint.
# mix run scripts/bench_batched_resync.exs --policy PATH --replay PATH --out PATH
# --fixtures DIR writes numerical fixtures; --reference DIR checks a prior run.
alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      replay: :string,
      character: :string,
      out: :string,
      batch: :integer,
      iterations: :integer,
      fixtures: :string,
      reference: :string,
      profile: :boolean
    ]
  )

{:ok, export} = ExPhil.Training.load_policy(Keyword.fetch!(opts, :policy))
if export.config[:bptt], do: raise("benchmark requires a window-trained policy")
n = opts[:batch] || 8
iterations = opts[:iterations] || 30
{:ok, metadata} = Peppi.metadata(Keyword.fetch!(opts, :replay))
character = opts[:character] || "fox"
own = Enum.find(metadata.players, &(String.downcase(&1.character_name || "") == character))
opp = Enum.find(metadata.players, &(&1.port != own.port))
{:ok, replay} = Peppi.parse(opts[:replay], player_port: own.port)

frames =
  replay
  |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.take(export.config.window_size + 2)

{:ok, agent} =
  Agent.start_link(policy: export, deterministic: true, stateful_step: true, stateful_resync: 30)

{:ok, _} = Agent.warmup(agent)
{setup_us, :ok} = :timer.tc(fn -> Agent.batch_init(agent, n) end)

for frame <- Enum.take(frames, export.config.window_size) do
  {:ok, _} = Agent.batch_get_controllers(agent, List.duplicate(frame.game_state, n))
end

base = :sys.get_state(agent)
input = List.duplicate(List.last(frames).game_state, n)
scenarios = for count <- Enum.uniq([0, 1, min(3, n), n]), do: {:resync, count}
scenarios = scenarios ++ for count <- Enum.uniq([1, min(3, n)]), do: {:cold_reset, count}

results =
  for {kind, count} <- scenarios do
    reset = fn ->
      counters = for row <- 0..(n - 1), do: if(kind == :resync and row < count, do: 30, else: 0)

      :sys.replace_state(agent, fn _ ->
        %{base | batch: %{base.batch | resync_steps: counters}}
      end)

      if kind == :cold_reset, do: Agent.batch_reset_rows(agent, Enum.to_list(0..(count - 1)))
    end

    reset.()

    {:ok, controllers, features} =
      Agent.batch_get_controllers(agent, input, return_features: true)

    state = Nx.backend_copy(:sys.get_state(agent).batch.trunk_state)
    sample = %{features: features, state: state}
    filename = "#{kind}_#{count}.nx"

    if opts[:fixtures] do
      File.mkdir_p!(opts[:fixtures])
      File.write!(Path.join(opts[:fixtures], filename), Nx.serialize(sample))

      File.write!(
        Path.join(opts[:fixtures], filename <> ".controllers"),
        :erlang.term_to_binary(controllers)
      )
    end

    if opts[:reference] do
      reference = Path.join(opts[:reference], filename) |> File.read!() |> Nx.deserialize()
      Nx.Testing.assert_all_close(features, reference.features, atol: 1.0e-4, rtol: 1.0e-4)

      for {key, value} <- state,
          do: Nx.Testing.assert_all_close(value, reference.state[key], atol: 1.0e-4, rtol: 1.0e-4)

      expected =
        Path.join(opts[:reference], filename <> ".controllers")
        |> File.read!()
        |> :erlang.binary_to_term([:safe])

      if controllers != expected,
        do: raise("controller mismatch: #{filename}: #{inspect({controllers, expected})}")
    end

    # Reset outside the timer so every sample starts from identical warmed state.
    times =
      for i <- 1..(iterations + 4), reduce: [] do
        acc ->
          reset.()
          {us, {:ok, _}} = :timer.tc(fn -> Agent.batch_get_controllers(agent, input) end)
          if i <= 4, do: acc, else: [us | acc]
      end
      |> Enum.sort()

    %{
      kind: kind,
      due: count,
      median_us: Enum.at(times, div(iterations, 2)),
      p95_us: Enum.at(times, min(iterations - 1, floor(iterations * 0.95)))
    }
  end

stage_profile =
  if opts[:profile] do
    functions = [
      {Agent, :batch_embed, 3},
      {Agent, :batch_trunk_step, 4},
      {Agent, :batch_resync, 6},
      {Agent, :action_to_controller, 2},
      {ExPhil.Networks.Policy.Sampling, :sample_autoregressive_from_features, 3}
    ]

    for mfa <- functions, do: :erlang.trace_pattern(mfa, true, [:local, :call_time])
    :erlang.trace(agent, true, [:call])

    try do
      for _ <- 1..30 do
        :sys.replace_state(agent, fn _ ->
          %{base | batch: %{base.batch | resync_steps: List.duplicate(0, n)}}
        end)

        {:ok, _} = Agent.batch_get_controllers(agent, input)
      end

      Map.new(functions, fn {mod, fun, arity} = mfa ->
        {:call_time, entries} = :erlang.trace_info(mfa, :call_time)
        entries = if is_list(entries), do: entries, else: []

        {calls, us} =
          Enum.reduce(entries, {0, 0}, fn {pid, count, sec, micros}, {calls, time} ->
            if pid == agent,
              do: {calls + count, time + sec * 1_000_000 + micros},
              else: {calls, time}
          end)

        {"#{inspect(mod)}.#{fun}/#{arity}", %{calls: calls, total_us: us, us_per_frame: us / 30}}
      end)
    after
      :erlang.trace(agent, false, [:call])
      for mfa <- functions, do: :erlang.trace_pattern(mfa, false, [:local, :call_time])
    end
  end

GenServer.stop(agent)

report = %{
  policy: opts[:policy],
  replay: opts[:replay],
  batch: n,
  iterations: iterations,
  client: EXLA.Client.default_name(),
  precision: Nx.Defn.default_options()[:precision],
  batch_setup_us: setup_us,
  reference_validated: opts[:reference] != nil,
  results: results,
  stage_profile: stage_profile,
  stage_profile_note:
    "Optional Erlang call-time attribution on 30 ordinary frames, separate from timings. Traced callees are accounted separately. Not synchronized GPU kernel time; tracing overhead applies.",
  note:
    "Full Agent call includes embedding, recurrence, sampling and output transfer. Compilation, fixture reset and row-reset API calls excluded. Deterministic heads. Independent-head exports are not supported by this batched API."
}

File.mkdir_p!(Path.dirname(Keyword.fetch!(opts, :out)))
File.write!(opts[:out], Jason.encode!(report, pretty: true))
IO.inspect(report)
