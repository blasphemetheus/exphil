# Full deployed inference paths, including legacy independent heads and BPTT.
alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      replay: :string,
      character: :string,
      batch: :integer,
      iterations: :integer,
      out: :string
    ]
  )

{:ok, export} = ExPhil.Training.load_policy(Keyword.fetch!(opts, :policy))
n = opts[:batch] || 8
iterations = opts[:iterations] || 50
if n < 1 or iterations < 1, do: raise("batch and iterations must be positive")
bptt = export.config[:bptt] == true
{:ok, metadata} = Peppi.metadata(Keyword.fetch!(opts, :replay))

own =
  Enum.find(metadata.players, &(String.downcase(&1.character_name || "") == opts[:character])) ||
    raise("character missing in replay")

opp = Enum.find(metadata.players, &(&1.port != own.port))
{:ok, replay} = Peppi.parse(opts[:replay], player_port: own.port)
history = 2 * export.config.window_size + 10

frames =
  replay
  |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.dedup_by(& &1.game_state.frame)
  |> Enum.take(history + 3)

true = length(frames) == history + 3

agent_opts =
  [
    policy: export,
    deterministic: true,
    stateful_step: true,
    harness: :sync_runner,
    frame_delay: 0,
    reaction_delay: 0,
    af_convention: :parsed
  ] ++ if(bptt, do: [], else: [stateful_resync: 30])

Output.banner("Trained recurrent full-policy profile")
Output.config([{"Policy", opts[:policy]}, {"Rows", n}, {"BPTT", bptt}])

{setup_us, {driver, agents}} =
  :timer.tc(fn ->
    {:ok, first} = Agent.start_link(agent_opts)
    {:ok, initial} = Agent.snapshot_state(first)
    {:ok, _} = Agent.warmup(first)
    {:ok, ^initial} = Agent.snapshot_state(first)
    batch_result = if n > 1, do: Agent.batch_init(first, n), else: :single_requested

    unless batch_result in [:ok, :single_requested, {:error, :batch_needs_autoregressive_head}],
      do: raise("batch init failed: #{inspect(batch_result)}")

    if batch_result == :ok do
      {:batch, [first]}
    else
      others =
        for _ <- List.duplicate(nil, n - 1) do
          {:ok, agent} = Agent.start_link(agent_opts)
          {:ok, _} = Agent.warmup(agent)
          agent
        end

      {:single, [first | others]}
    end
  end)

call = fn frame ->
  case driver do
    :batch ->
      {:ok, cs} = Agent.batch_get_controllers(hd(agents), List.duplicate(frame.game_state, n))
      cs

    :single ->
      Enum.map(agents, fn a ->
        {:ok, c} = Agent.get_controller(a, frame.game_state)
        c
      end)
  end
end

for frame <- Enum.take(frames, history), do: call.(frame)
bases = Enum.map(agents, &:sys.get_state/1)

Output.puts(
  "Primed state: " <>
    inspect(
      Map.take(hd(bases), [
        :window_size,
        :action_repeat,
        :stateful_resync,
        :steps_since_resync,
        :trunk_cold
      ])
    ) <> " history=#{length(hd(bases).step_frame_buffer)}"
)

if driver == :single and not bptt,
  do: true = Enum.all?(bases, &(length(&1.step_frame_buffer) >= &1.window_size - 1))

next = Enum.at(frames, history)
suffix = Enum.drop(frames, history)

restore = fn ->
  Enum.zip(agents, bases)
  |> Enum.each(fn {a, base} -> :sys.replace_state(a, fn _ -> base end) end)
end

# Exercise the public rollback API on single inference; batch rollback is checked
# separately via an internal state fixture, not advertised as a public snapshot.
rollback =
  if driver == :single do
    saved =
      Enum.map(agents, fn a ->
        {:ok, blob} = Agent.snapshot_state(a)
        blob
      end)

    first = Enum.map(suffix, call)
    final = Enum.map(agents, &Agent.snapshot_state/1)
    Enum.zip(agents, saved) |> Enum.each(fn {a, blob} -> :ok = Agent.restore_state(a, blob) end)
    true = first == Enum.map(suffix, call)
    true = final == Enum.map(agents, &Agent.snapshot_state/1)
    :public_snapshot_exact
  else
    first = Enum.map(suffix, call)
    final = :sys.get_state(hd(agents)).batch.trunk_state |> Edifice.Stateful.serialize()
    restore.()
    true = first == Enum.map(suffix, call)
    true = final == :sys.get_state(hd(agents)).batch.trunk_state |> Edifice.Stateful.serialize()
    :internal_batch_fixture_exact
  end

scenarios =
  [{:ordinary, 0}] ++
    for(count <- Enum.uniq([1, min(3, n), n]), do: {:cold_reset, count}) ++
    if(bptt, do: [], else: for(count <- Enum.uniq([1, min(3, n), n]), do: {:resync, count}))

results =
  for {kind, count} <- scenarios do
    reset = fn ->
      restore.()

      case driver do
        :batch ->
          counters =
            for row <- 0..(n - 1), do: if(kind == :resync and row < count, do: 30, else: 0)

          :sys.replace_state(hd(agents), fn state ->
            %{state | batch: %{state.batch | resync_steps: counters}}
          end)

          if kind == :cold_reset,
            do: Agent.batch_reset_rows(hd(agents), Enum.to_list(0..(count - 1)))

        :single ->
          Enum.with_index(agents)
          |> Enum.each(fn {a, row} ->
            :sys.replace_state(a, fn state ->
              %{state | steps_since_resync: if(kind == :resync and row < count, do: 30, else: 0)}
            end)

            if kind == :cold_reset and row < count, do: Agent.reset_buffer(a)
          end)
      end
    end

    reset.()
    call.(next)

    if kind == :resync do
      actual =
        if driver == :batch,
          do: :sys.get_state(hd(agents)).batch.resync_steps,
          else: Enum.map(agents, &:sys.get_state(&1).steps_since_resync)

      true = Enum.take(actual, count) == List.duplicate(0, count)
    end

    times =
      for i <- 1..(iterations + 5), reduce: [] do
        acc ->
          reset.()
          {us, _} = :timer.tc(fn -> call.(next) end)
          if i > 5, do: [us | acc], else: acc
      end
      |> Enum.sort()

    result = %{
      kind: kind,
      rows: count,
      samples: iterations,
      median_us: Enum.at(times, div(iterations, 2)),
      p95_us: Enum.at(times, ceil(iterations * 0.95) - 1)
    }

    Output.puts(inspect(result))
    result
  end

for a <- agents, do: GenServer.stop(a)

results =
  if bptt do
    results
  else
    window_opts =
      agent_opts |> Keyword.delete(:stateful_resync) |> Keyword.put(:stateful_step, false)

    window_agents =
      for _ <- 1..if(driver == :batch, do: 1, else: n) do
        {:ok, a} = Agent.start_link(window_opts)
        {:ok, _} = Agent.warmup(a)
        if driver == :batch, do: :ok = Agent.batch_init(a, n)
        a
      end

    window_call = fn frame ->
      if driver == :batch do
        {:ok, cs} =
          Agent.batch_get_controllers(hd(window_agents), List.duplicate(frame.game_state, n))

        cs
      else
        Enum.map(window_agents, fn a ->
          {:ok, c} = Agent.get_controller(a, frame.game_state)
          c
        end)
      end
    end

    for frame <- Enum.take(frames, history), do: window_call.(frame)
    saved = Enum.map(window_agents, &:sys.get_state/1)

    times =
      for i <- 1..(iterations + 5), reduce: [] do
        acc ->
          Enum.zip(window_agents, saved)
          |> Enum.each(fn {a, state} -> :sys.replace_state(a, fn _ -> state end) end)

          {us, _} = :timer.tc(fn -> window_call.(next) end)
          if i > 5, do: [us | acc], else: acc
      end
      |> Enum.sort()

    for a <- window_agents, do: GenServer.stop(a)

    result = %{
      kind: :windowed,
      rows: n,
      samples: iterations,
      median_us: Enum.at(times, div(iterations, 2)),
      p95_us: Enum.at(times, ceil(iterations * 0.95) - 1)
    }

    Output.puts(inspect(result))
    results ++ [result]
  end

revision = fn path ->
  {sha, 0} = System.cmd("git", ["-C", path, "rev-parse", "HEAD"])
  String.trim(sha)
end

report = %{
  policy: opts[:policy],
  replay: opts[:replay],
  character: opts[:character],
  backbone: export.config.backbone,
  head: export.config[:head],
  bptt: bptt,
  rows: n,
  driver: driver,
  setup_us: setup_us,
  rollback: rollback,
  warmup_preserves_state: true,
  client: EXLA.Client.default_name(),
  precision: Nx.Defn.default_options()[:precision],
  exphil_revision: revision.("."),
  edifice_revision: revision.("../edifice"),
  results: results,
  note:
    "Full Agent controllers including embedding, recurrence, heads and transfer. Restore/reset calls and compilation excluded. Deterministic sampling. BPTT never resynchronized. Independent heads use sequential per-row Agents."
}

File.mkdir_p!(Path.dirname(Keyword.fetch!(opts, :out)))
File.write!(opts[:out], Jason.encode!(report, pretty: true))
