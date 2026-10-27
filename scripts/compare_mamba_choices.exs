# AI-generated; authorship and method: docs/guides/MAMBA_CONTROLLER_CHOICES.md.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
Code.require_file("lib/mamba_precision_experiment.exs", __DIR__)
Code.require_file("lib/mamba_carry_resolution.exs", __DIR__)
Code.require_file("lib/mamba_choice_metrics.exs", __DIR__)

defmodule MambaChoiceComparison do
  alias MambaCarryResolution, as: Carry
  alias ExPhil.Networks.Policy.Sampling
  alias ExPhil.Training.Output

  def run(opts) do
    capture = Keyword.fetch!(opts, :capture)

    meta =
      Path.join(Path.dirname(capture), "capture_metadata.json") |> File.read!() |> Jason.decode!()

    policy = meta["policy"]
    unless meta["capture_mode"] == "baseline", do: raise("Use a baseline Agent capture")
    expected_hash = meta["hashes"][policy]
    actual_hash = :crypto.hash(:sha256, File.read!(policy)) |> Base.encode16(case: :lower)
    unless expected_hash == actual_hash, do: raise("Checkpoint hash mismatch")
    {:ok, export} = ExPhil.Training.load_policy(policy)
    config = export.config

    for flag <- [:button_events, :stick_events, :danger_context, :stick_duration] do
      if Map.get(config, flag, false) not in [false, nil],
        do: raise("Unsupported context: #{flag}")
    end

    data = capture |> File.read!() |> Nx.deserialize()
    window = Nx.to_number(data.window)
    width = Nx.to_number(data.width)
    Output.puts("Computing baseline f32 and f64 trajectories")
    run = Nx.Defn.jit(&Carry.trajectory/5, compiler: EXLA, precision: :highest)

    outputs =
      for type <- [:f32, :f64] do
        run.(
          Carry.cast(data.params, type),
          Carry.cast(data.initial, type),
          Carry.cast(data.frames, type),
          data.resets,
          window: window,
          width: width
        ).output
        |> Carry.host()
      end

    [baseline, reference] = outputs
    params = Nx.Defn.Composite.traverse(export.params, &Nx.backend_copy(&1, EXLA.Backend))

    decode =
      if Map.get(config, :head, :independent) == :autoregressive do
        fn x, opts -> Sampling.sample_autoregressive_from_features(params, x, opts) end
      else
        model =
          Axon.input("features", shape: {nil, width})
          |> ExPhil.Networks.Policy.Heads.build_controller_head(
            config.axis_buckets,
            config.shoulder_buckets
          )

        {_, predict} = Axon.build(model, compiler: EXLA)
        fn x, opts -> Sampling.sample(params, predict, x, opts) end
      end

    ticks = Nx.axis_size(baseline, 0)
    rows = Nx.axis_size(baseline, 1)

    resets =
      data.resets
      |> Nx.slice_along_axis(window - 1, ticks, axis: 0)
      |> Nx.to_flat_list()
      |> Enum.chunk_every(rows)

    results =
      for mode <- [:deterministic, :stochastic] do
        records =
          for tick <- 0..(ticks - 1) do
            opts = [
              deterministic: mode == :deterministic,
              key: Nx.Random.key(10_000 + tick),
              temperature: 1.0
            ]

            decoded =
              for features <- [data.actual.output, baseline, reference] do
                # Keep production heads f32; isolate trunk error, including its f32 interface.
                x = Nx.take(features, tick) |> Nx.as_type(:f32) |> Nx.backend_copy(EXLA.Backend)
                decode.(x, opts) |> Carry.host()
              end

            [agent, batch, ref] = decoded
            if rem(tick, 128) == 0, do: Output.puts("#{mode} frame #{tick}")

            for {name, candidate} <- [{:agent, agent}, {:replay, batch}], row <- 0..(rows - 1) do
              MambaChoiceMetrics.record(candidate, ref, row)
              |> Map.merge(%{
                candidate: name,
                tick: tick,
                row: row,
                resync: Enum.at(Enum.at(resets, tick), row) == 1
              })
            end
          end
          |> List.flatten()

        {mode,
         Map.new([:agent, :replay], fn name ->
           selected = Enum.filter(records, &(&1.candidate == name))
           {name, MambaChoiceMetrics.summarize(selected)}
         end)}
      end
      |> Map.new()

    report = %{
      client: EXLA.Client.default_name(),
      capture: capture,
      policy: policy,
      hashes: meta["hashes"],
      ticks: ticks,
      rows: rows,
      head_precision: "f32 for all comparisons",
      reference: "f64 trunk rounded to f32 at production head interface",
      stochastic: "temperature 1, shared key 10000 + tick, one draw per row/frame",
      feature_metrics: %{
        agent: Carry.metrics(data.actual.output, reference),
        replay: Carry.metrics(baseline, reference)
      },
      choices: results
    }

    out = Keyword.fetch!(opts, :out)
    File.mkdir_p!(Path.dirname(out))
    File.write!(out, Jason.encode!(report, pretty: true))
    Output.puts("Saved #{out}")
  end
end

{opts, [], []} = OptionParser.parse(System.argv(), strict: [capture: :string, out: :string])
MambaChoiceComparison.run(opts)
