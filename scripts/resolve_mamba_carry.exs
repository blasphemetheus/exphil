# AI-generated; method and authorship: docs/guides/MAMBA_CARRY_RESOLUTION.md.
# Run through scripts/run_recurrent_validation.exs.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
Code.require_file("lib/mamba_precision_experiment.exs", __DIR__)
Code.compile_file("lib/exphil/embeddings/player.ex")
Code.compile_file("lib/exphil/embeddings/game/spatial.ex")

defmodule MambaCarryResolution do
  import Nx.Defn
  alias ExPhil.Agents.Agent, as: PolicyAgent
  alias ExPhil.Data.Peppi
  alias ExPhil.Training.Output

  deftransformp(step(p, s, x), do: Edifice.SSM.Mamba.step(p, s, x))

  deftransformp(replay(p, s, x),
    do: Edifice.Stateful.Replay.run(p, s, x, module: Edifice.SSM.Mamba)
  )

  defn trajectory(p, initial, frames, resets, opts \\ []) do
    opts = keyword!(opts, [:window, :width])
    window = opts[:window]
    n = Nx.axis_size(frames, 0)
    count = Nx.axis_size(frames, 1)

    convs =
      Nx.broadcast(
        Nx.tensor(0, type: Nx.type(frames)),
        {count, n, Nx.axis_size(initial.conv, 1), Nx.axis_size(initial.conv, 2),
         Nx.axis_size(initial.conv, 3)}
      )

    outputs = Nx.broadcast(Nx.tensor(0, type: Nx.type(frames)), {count, n, opts[:width]})

    {_, carry, _, _, _, _, convs, outputs} =
      while {i = 0, carry = initial, p, frames, resets, initial, convs, outputs}, i < count do
        carry =
          if Nx.take(resets, i) do
            history = Nx.slice_along_axis(frames, i - window + 1, window - 1, axis: 1)
            replay(p, initial, history)
          else
            carry
          end

        {y, carry} = step(p, carry, Nx.take(frames, i, axis: 1))
        convs = Nx.put_slice(convs, [i, 0, 0, 0, 0], Nx.new_axis(carry.conv, 0))
        outputs = Nx.put_slice(outputs, [i, 0, 0], Nx.new_axis(y, 0))
        {i + 1, carry, p, frames, resets, initial, convs, outputs}
      end

    %{
      conv: Nx.slice_along_axis(convs, window - 1, count - window + 1, axis: 0),
      output: Nx.slice_along_axis(outputs, window - 1, count - window + 1, axis: 0),
      h: carry.h
    }
  end

  def cast(x, type), do: Nx.Defn.Composite.traverse(x, &Nx.as_type(&1, type))
  def row(x, i), do: Nx.slice_along_axis(x, i, 1, axis: 0)
  def host(x), do: Nx.Defn.Composite.traverse(x, &Nx.backend_copy(&1, Nx.BinaryBackend))

  def metrics(actual, expected) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      # Assess in f64 so comparison rounding cannot hide small discrepancies.
      a = Nx.as_type(actual, :f64)
      b = Nx.as_type(expected, :f64)
      diff = Nx.abs(Nx.subtract(a, b))
      bound = Nx.add(1.0e-4, Nx.multiply(1.0e-4, Nx.abs(b)))

      %{
        max_abs: Nx.to_number(Nx.reduce_max(diff)),
        max_error_over_bound: Nx.to_number(Nx.reduce_max(Nx.divide(diff, bound))),
        failed_values: Nx.to_number(Nx.sum(Nx.greater(diff, bound))),
        values: Nx.size(a),
        rms: Nx.to_number(Nx.sqrt(Nx.mean(Nx.multiply(diff, diff))))
      }
    end)
  end

  def compare(a, b), do: Map.new(a, fn {key, v} -> {key, metrics(v, b[key])} end)

  def capture(ticks, n) do
    {_, policy, replay_path, _} =
      Enum.find(ExPhil.BenchmarkSweep.policies(), &(elem(&1, 0) == "mamba"))

    {:ok, export} = ExPhil.Training.load_policy(policy)
    {:ok, metadata} = Peppi.metadata(replay_path)
    own = Enum.find(metadata.players, &(String.downcase(&1.character_name || "") == "fox"))
    opponent = Enum.find(metadata.players, &(&1.port != own.port))
    {:ok, parsed} = Peppi.parse(replay_path, player_port: own.port)

    games =
      Peppi.to_training_frames(parsed,
        player_port: own.port,
        opponent_port: opponent.port,
        remap_ports: true
      )
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.dedup_by(& &1.game_state.frame)
      |> Enum.map(& &1.game_state)
      |> List.to_tuple()

    {:ok, a} =
      PolicyAgent.start_link(
        policy: export,
        deterministic: true,
        stateful_step: true,
        stateful_resync: 30,
        harness: :sync_runner,
        frame_delay: 0,
        reaction_delay: 0,
        af_convention: :parsed
      )

    :ok = PolicyAgent.batch_init(a, n)
    initial = :sys.get_state(a)

    records =
      for tick <- 0..(ticks - 1) do
        inputs = for i <- 0..(n - 1), do: Map.put(elem(games, tick + i * 37), :frame, tick)
        {:ok, _, features} = PolicyAgent.batch_get_controllers(a, inputs, return_features: true)
        b = :sys.get_state(a).batch
        if rem(tick, 64) == 0, do: Output.puts("Captured real Agent frame #{tick}")

        %{
          frame: host(b.last_frames),
          conv: host(b.trunk_state.conv),
          y: host(features),
          reset: Enum.all?(b.resync_steps, &(&1 == 0))
        }
      end

    final = :sys.get_state(a).batch.trunk_state
    GenServer.stop(a)
    w = export.config.window_size
    real = Enum.map(records, & &1.frame) |> Nx.stack(axis: 1)
    padding = real |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.tile([1, w - 1, 1])

    %{
      params: host(initial.trunk_step_params),
      initial: host(initial.batch.trunk_state),
      frames: Nx.concatenate([padding, real], axis: 1),
      resets:
        Nx.tensor(List.duplicate(0, w - 1) ++ Enum.map(records, &if(&1.reset, do: 1, else: 0)),
          type: :u8
        ),
      actual: %{
        conv: Enum.map(records, & &1.conv) |> Nx.stack(),
        output: Enum.map(records, & &1.y) |> Nx.stack(),
        h: host(final.h)
      },
      window: w,
      width: Nx.axis_size(hd(records).y, 1)
    }
  end

  def run(opts) do
    ticks = opts[:ticks] || 256
    n = opts[:rows] || 8
    out = Keyword.fetch!(opts, :out)
    File.mkdir_p!(out)

    data =
      if opts[:capture],
        do: opts[:capture] |> File.read!() |> Nx.deserialize(),
        else: capture(ticks, n)

    data =
      if opts[:capture],
        do: %{data | window: Nx.to_number(data.window), width: Nx.to_number(data.width)},
        else: data

    n = Nx.axis_size(data.frames, 0)
    ticks = Nx.axis_size(data.frames, 1) - data.window + 1
    unless opts[:capture], do: File.write!(Path.join(out, "capture.nx"), Nx.serialize(data))
    MambaPrecisionExperiment.install(opts[:mode] || "baseline")
    compiled = Nx.Defn.jit(&trajectory/5, compiler: EXLA, precision: :highest)

    run = fn p, s, x ->
      compiled.(p, s, x, data.resets, window: data.window, width: data.width) |> host()
    end

    Output.puts("Compiling whole f32 and f64 trajectories")
    batch32 = run.(data.params, data.initial, data.frames)
    batch64 = run.(cast(data.params, :f64), cast(data.initial, :f64), cast(data.frames, :f64))

    rows =
      for i <- 0..(n - 1) do
        Output.puts("Reference row #{i}")
        s = Map.new(data.initial, fn {k, v} -> {k, row(v, i)} end)
        x = row(data.frames, i)
        one32 = run.(data.params, s, x)
        one64 = run.(cast(data.params, :f64), cast(s, :f64), cast(x, :f64))

        select = fn results ->
          Map.new(results, fn {k, v} ->
            {k, Nx.slice_along_axis(v, i, 1, axis: if(k == :h, do: 0, else: 1))}
          end)
        end

        b32 = select.(batch32)
        b64 = select.(batch64)

        %{
          row: i,
          f32_batch_vs_single: compare(b32, one32),
          f64_batch_vs_single: compare(b64, one64),
          batch32_vs_reference: compare(b32, one64),
          single32_vs_reference: compare(one32, one64)
        }
      end

    mode = opts[:mode] || "baseline"

    comparisons = [
      :f32_batch_vs_single,
      :f64_batch_vs_single,
      :batch32_vs_reference,
      :single32_vs_reference
    ]

    passed =
      Enum.all?(rows, fn row ->
        Enum.all?(comparisons, fn key ->
          Enum.all?(row[key], fn {_, metric} -> metric.failed_values == 0 end)
        end)
      end)

    report = %{
      client: EXLA.Client.default_name(),
      ticks: ticks,
      batch_size: n,
      mode: mode,
      atol: 1.0e-4,
      rtol: 1.0e-4,
      reference_checks_passed: passed,
      agent_comparison_is_baseline_reproduction: mode == "baseline",
      rows: rows,
      actual_agent_vs_compiled_trajectory: compare(data.actual, batch32)
    }

    File.write!(Path.join(out, "report.json"), Jason.encode!(report, pretty: true))
    Output.puts("Saved #{out}/report.json")
  end
end

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [out: :string, ticks: :integer, rows: :integer, capture: :string, mode: :string]
  )

MambaCarryResolution.run(opts)
