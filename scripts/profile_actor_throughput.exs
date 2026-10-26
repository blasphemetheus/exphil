# Run through scripts/run_recurrent_validation.exs; never rebuild NIFs during load.
Code.require_file("lib/resync_spread.exs", __DIR__)

defmodule ActorThroughputProfile do
  alias ExPhil.Agents.Agent, as: PolicyAgent
  alias ExPhil.Data.Peppi

  def now, do: System.monotonic_time(:microsecond)

  def stats(values) do
    sorted = Enum.sort(values)
    n = length(sorted)

    %{
      samples: n,
      median_us: Enum.at(sorted, div(n, 2)),
      p95_us: Enum.at(sorted, ceil(n * 0.95) - 1),
      p99_us: Enum.at(sorted, ceil(n * 0.99) - 1),
      max_us: List.last(sorted)
    }
  end

  def wait_until(target) do
    remaining = target - now()
    if remaining > 0, do: Process.sleep(ceil(remaining / 1000))
  end

  def work(call, rate, start, duration, tick, samples) do
    target = if rate == :paced, do: start + div(tick * 1_000_000, 60), else: now()

    done =
      if rate == :paced,
        do: tick >= round(duration * 60),
        else: now() >= start + duration * 1_000_000

    if done do
      Enum.reverse(samples)
    else
      if rate == :paced, do: wait_until(target)
      began = now()
      call.(tick)
      ended = now()
      sample = [began - start, ended - began, ended - target]
      work(call, rate, start, duration, tick + 1, [sample | samples])
    end
  end

  def load_loop(fun, x, started, count) do
    receive do
      :stop -> %{iterations: count, elapsed_us: now() - started}
    after
      0 ->
        value = fun.(x) |> Nx.to_number()
        unless is_number(value) and value > 0, do: raise("invalid load result")
        load_loop(fun, x, started, count + 1)
    end
  end

  def run(opts) do
    duration = opts[:seconds] || 3
    rows = opts[:rows] || 8
    unless duration >= 1 and rows >= 2, do: raise("seconds >= 1 and rows >= 2 required")
    {:ok, export} = ExPhil.Training.load_policy(Keyword.fetch!(opts, :policy))
    {:ok, metadata} = Peppi.metadata(Keyword.fetch!(opts, :replay))

    own =
      Enum.find(metadata.players, &(String.downcase(&1.character_name || "") == opts[:character]))

    unless own, do: raise("character missing")
    opp = Enum.find(metadata.players, &(&1.port != own.port))
    {:ok, replay} = Peppi.parse(opts[:replay], player_port: own.port)

    frames =
      replay
      |> Peppi.to_training_frames(
        player_port: own.port,
        opponent_port: opp.port,
        remap_ports: true
      )
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.dedup_by(& &1.game_state.frame)
      |> Enum.take(3600)
      |> Enum.map(& &1.game_state)
      |> List.to_tuple()

    history = 2 * export.config.window_size + 10
    true = tuple_size(frames) > history

    frame = fn tick, row ->
      Map.put(elem(frames, rem(tick + row * 37, tuple_size(frames))), :frame, tick)
    end

    bptt = export.config[:bptt] == true
    spread = opts[:resync_spread] || false
    if spread and bptt, do: raise("BPTT carry must not be periodically reset")

    agent_opts =
      [
        policy: export,
        deterministic: false,
        stateful_step: true,
        harness: :sync_runner,
        frame_delay: 0,
        reaction_delay: 0,
        af_convention: :parsed
      ] ++
        if(bptt, do: [], else: [stateful_resync: 30])

    # Same saturated GPU competitor for host and CUDA actors; not a training proxy.
    x = Nx.broadcast(Nx.tensor(0.001, backend: {EXLA.Backend, client: :cuda}), {2048, 2048})

    load =
      Nx.Defn.jit(fn a -> Nx.sum(Nx.dot(a, a)) end,
        compiler: EXLA,
        client: :cuda,
        precision: :highest
      )

    _ = load.(x) |> Nx.to_number()

    {setup_us, groups} =
      :timer.tc(fn ->
        for mode <- if(spread, do: [:actors, :batch], else: [:single, :actors, :batch]) do
          n = if mode == :single, do: 1, else: rows
          count = if mode == :batch, do: 1, else: n

          agents =
            for _ <- 1..count do
              {:ok, a} = PolicyAgent.start_link(agent_opts)
              {:ok, before} = PolicyAgent.snapshot_state(a)
              {:ok, _} = PolicyAgent.warmup(a)
              {:ok, ^before} = PolicyAgent.snapshot_state(a)
              a
            end

          if mode == :batch, do: :ok = PolicyAgent.batch_init(hd(agents), n)

          calls =
            if mode == :batch do
              [
                fn tick ->
                  {:ok, controllers} =
                    PolicyAgent.batch_get_controllers(
                      hd(agents),
                      for(row <- 0..(n - 1), do: frame.(tick, row))
                    )

                  true = length(controllers) == n
                end
              ]
            else
              for {a, row} <- Enum.with_index(agents) do
                fn tick -> {:ok, _} = PolicyAgent.get_controller(a, frame.(tick, row)) end
              end
            end

          for tick <- 0..(history - 1), call <- calls, do: call.(tick)
          bases = Enum.map(agents, &:sys.get_state/1)
          %{mode: mode, rows: n, agents: agents, calls: calls, bases: bases}
        end
      end)

    groups =
      if spread, do: Enum.map(groups, &ExPhil.ResyncSpread.prepare(&1, history)), else: groups

    history = if spread, do: history + 1, else: history
    validation = if spread, do: ExPhil.ResyncSpread.validate(groups, history), else: nil

    phases =
      if spread do
        for round <- [1, 2],
            schedule <- if(round == 1, do: [:aligned, :spread], else: [:spread, :aligned]),
            group <- groups,
            contention <- [false, true],
            do: {group, :paced, contention, schedule, round}
      else
        for group <- groups,
            rate <- [:saturated, :paced],
            contention <- [false, true],
            do: {group, rate, contention, nil, nil}
      end

    results =
      for {group, rate, contention, schedule, round} <- phases do
        if spread do
          ExPhil.ResyncSpread.restore(group, schedule)
        else
          for {a, base} <- Enum.zip(group.agents, group.bases),
              do: :sys.replace_state(a, fn _ -> base end)
        end

        competitor = if contention, do: Task.async(fn -> load_loop(load, x, now(), 0) end)
        start = now() + 100_000

        tasks =
          for call <- group.calls do
            Task.async(fn ->
              wait_until(start)
              work(fn tick -> call.(history + tick) end, rate, start, duration, 0, [])
            end)
          end

        streams = Enum.map(tasks, &Task.await(&1, 180_000))
        # Include the complete offered interval, even when the final paced
        # request finishes before its 16.67 ms slot ends.
        wait_until(start + duration * 1_000_000)
        finished = now()

        load_stats =
          if competitor do
            send(competitor.pid, :stop)
            Task.await(competitor, 60_000)
          end

        samples = Enum.concat(streams)
        multiplier = if group.mode == :batch, do: group.rows, else: 1
        service = Enum.map(samples, &Enum.at(&1, 1))
        response = Enum.map(samples, &Enum.at(&1, 2))

        result = %{
          mode: group.mode,
          rows: group.rows,
          rate: rate,
          contention: contention,
          elapsed_us: finished - start,
          calls: length(samples),
          actions: length(samples) * multiplier,
          actions_per_second: length(samples) * multiplier * 1_000_000 / (finished - start),
          service: stats(service),
          response: stats(response),
          deadline_misses:
            if(rate == :paced, do: Enum.count(response, &(&1 > 1_000_000 / 60)), else: nil),
          worker_calls: Enum.map(streams, &length/1),
          competitor: load_stats,
          streams: streams
        }

        result =
          if spread, do: Map.merge(result, %{schedule: schedule, round: round}), else: result

        IO.inspect(Map.drop(result, [:streams, :competitor]), label: "PROFILE", limit: :infinity)
        result
      end

    for group <- groups, a <- group.agents, do: GenServer.stop(a)
    out = Keyword.fetch!(opts, :out)
    File.write!(out <> ".samples.json.gz", :zlib.gzip(Jason.encode!(results)))

    report = %{
      client: System.get_env("RECURRENT_CLIENT"),
      policy: opts[:policy],
      replay: opts[:replay],
      bptt: bptt,
      resync: if(bptt, do: nil, else: 30),
      seconds: duration,
      setup_us: setup_us,
      competitor: "saturated 2048x2048 f32 CUDA matmul + sum, synchronized each iteration",
      samples_file: Path.basename(out <> ".samples.json.gz"),
      results: Enum.map(results, &Map.delete(&1, :streams))
    }

    report = if spread, do: Map.put(report, :resync_validation, validation), else: report
    File.write!(out, Jason.encode!(report, pretty: true) <> "\n")
  end
end

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      replay: :string,
      character: :string,
      rows: :integer,
      seconds: :integer,
      resync_spread: :boolean,
      out: :string
    ]
  )

ActorThroughputProfile.run(opts)
