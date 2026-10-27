# AI-generated diagnostic. See docs/guides/MAMBA_BATCH_PARITY.md for attribution and method.
# Run through scripts/run_recurrent_validation.exs.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
Code.compile_file("lib/exphil/embeddings/player.ex")
Code.compile_file("lib/exphil/embeddings/game/spatial.ex")

defmodule MambaBatchDiagnostic do
  alias ExPhil.Agents.Agent, as: PolicyAgent
  alias ExPhil.Data.Peppi
  alias ExPhil.Training.Output
  alias Edifice.Stateful.Ops

  def row(x, i), do: Nx.slice_along_axis(x, i, 1, axis: 0)
  def slice(state, i), do: Map.new(state, fn {k, v} -> {k, row(v, i)} end)

  def embedding_error(a, b) do
    differences =
      Enum.zip(Nx.to_flat_list(a), Nx.to_flat_list(b))
      |> Enum.with_index()
      |> Enum.reject(fn {{x, y}, _} -> x == y end)
      |> Enum.map(fn {{x, y}, i} -> %{index: i, batch: x, single: y} end)

    Map.put(error(a, b), :differences, differences)
  end

  def error(a, b) do
    a = Nx.backend_copy(a, Nx.BinaryBackend)
    b = Nx.backend_copy(b, Nx.BinaryBackend)

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      %{
        max_abs: Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(a, b)))),
        close: Nx.to_number(Nx.all_close(a, b, atol: 1.0e-4, rtol: 1.0e-4)) == 1
      }
    end)
  end

  # Expose existing primitives without changing their equations. Returning
  # intermediates can change compiler fusion; check the complete step separately.
  def stages(params, carry, frame) do
    params = Ops.unwrap_params(params)

    x =
      if params["input_projection"], do: Ops.dense(frame, params["input_projection"]), else: frame

    f64 =
      if params["input_projection"] do
        weights = Nx.Defn.Composite.traverse(params["input_projection"], &Nx.as_type(&1, :f64))
        Ops.dense(Nx.as_type(frame, :f64), weights)
      else
        Nx.as_type(frame, :f64)
      end

    {_, values} =
      Enum.reduce(
        1..Nx.axis_size(carry.h, 1),
        {x, [{"input_projection", x}, {"input_projection_f64", f64}]},
        fn i, {x, values} ->
          name = "mamba_block_#{i}"
          h = Nx.squeeze(Nx.slice_along_axis(carry.h, i - 1, 1, axis: 1), axes: [1])
          conv = Nx.squeeze(Nx.slice_along_axis(carry.conv, i - 1, 1, axis: 1), axes: [1])
          norm = Ops.layer_norm(x, params[name <> "_norm"])
          xz = Ops.dense(norm, params[name <> "_in_proj"])
          inner = div(Nx.axis_size(xz, 1), 2)
          branch = Nx.slice_along_axis(xz, 0, inner, axis: 1)
          z = Nx.slice_along_axis(xz, inner, inner, axis: 1)
          {conv_out, _} = Ops.conv1d_step(branch, conv, params[name <> "_conv_dw_conv"])
          activated = Ops.silu(conv_out)
          {y, new_h} = Edifice.SSM.Mamba.step_selective_ssm(params, name <> "_ssm", activated, h)
          gated = Nx.multiply(y, Ops.silu(z))
          out = Nx.add(x, Ops.dense(gated, params[name <> "_out_proj"]))

          parts = [
            {"norm", norm},
            {"in_proj", xz},
            {"conv", conv_out},
            {"silu", activated},
            {"ssm_h", new_h},
            {"ssm_y", y},
            {"gate", gated},
            {"residual", out}
          ]

          {out, values ++ Enum.map(parts, fn {k, v} -> {name <> "/" <> k, v} end)}
        end
      )

    Map.new(values)
  end

  def run(opts) do
    {_, policy, replay_path, _} =
      Enum.find(ExPhil.BenchmarkSweep.policies(), &(elem(&1, 0) == "mamba"))

    sources =
      [
        policy,
        replay_path,
        __ENV__.file,
        "lib/exphil/embeddings/player.ex",
        "lib/exphil/embeddings/game/spatial.ex"
      ] ++
        ExPhil.BenchmarkSweep.inference_sources()

    hashes = ExPhil.BenchmarkSweep.hashes(sources)
    {:ok, export} = ExPhil.Training.load_policy(policy)
    {:ok, metadata} = Peppi.metadata(replay_path)
    own = Enum.find(metadata.players, &(String.downcase(&1.character_name || "") == "fox"))
    opp = Enum.find(metadata.players, &(&1.port != own.port))
    {:ok, replay} = Peppi.parse(replay_path, player_port: own.port)

    frames =
      Peppi.to_training_frames(replay,
        player_port: own.port,
        opponent_port: opp.port,
        remap_ports: true
      )
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.dedup_by(& &1.game_state.frame)
      |> Enum.map(& &1.game_state)
      |> List.to_tuple()

    n = 8
    ticks = opts[:ticks] || 160
    phase? = opts[:stagger] || false

    base = [
      policy: export,
      deterministic: true,
      stateful_step: true,
      stateful_resync: 30,
      harness: :sync_runner,
      frame_delay: 0,
      reaction_delay: 0,
      af_convention: :parsed
    ]

    agents =
      for i <- 0..(n - 1) do
        {:ok, a} =
          PolicyAgent.start_link(
            base ++ [stateful_resync_phase: if(phase?, do: div(i * 31, n), else: 0)]
          )

        a
      end

    {:ok, batch} = PolicyAgent.start_link(base ++ [stateful_resync_stagger: phase?])
    :ok = PolicyAgent.batch_init(batch, n)
    step = Nx.Defn.jit(&Edifice.SSM.Mamba.step/3, compiler: EXLA, precision: :highest)
    probe = Nx.Defn.jit(&stages/3, compiler: EXLA, precision: :highest)

    Output.puts(
      "Comparing #{ticks} replay frames, eight rows, #{EXLA.Client.default_name()}; first calls compile"
    )

    events =
      for tick <- 0..(ticks - 1) do
        inputs = for i <- 0..(n - 1), do: Map.put(elem(frames, tick + i * 37), :frame, tick)

        {:ok, actual, features} =
          PolicyAgent.batch_get_controllers(batch, inputs, return_features: true)

        singles =
          for {a, input} <- Enum.zip(agents, inputs) do
            {:ok, controller} = PolicyAgent.get_controller(a, input)
            {controller, :sys.get_state(a)}
          end

        bs = :sys.get_state(batch)

        rows =
          for {{controller, s}, i} <- Enum.with_index(singles) do
            %{
              row: i,
              controller_equal: controller == Enum.at(actual, i),
              embedding: embedding_error(row(bs.batch.last_frames, i), hd(s.step_frame_buffer)),
              carry:
                Map.new(s.trunk_state, fn {k, v} ->
                  {k, error(row(bs.batch.trunk_state[k], i), v)}
                end)
            }
          end

        # At selected frames, hold inputs AND incoming carry identical. This
        # isolates batch-shape arithmetic from accumulated recurrent differences.
        isolated =
          if tick in [0, 31, 63, ticks - 1] do
            p = bs.trunk_step_params
            c = bs.batch.trunk_state
            x = bs.batch.last_frames
            {by, bc} = step.(p, c, x)
            stages_b = probe.(p, c, x)

            for i <- 0..(n - 1) do
              {sy, sc} = step.(p, slice(c, i), row(x, i))
              stages_s = probe.(p, slice(c, i), row(x, i))

              %{
                row: i,
                output: error(row(by, i), sy),
                carry: Map.new(sc, fn {k, v} -> {k, error(row(bc[k], i), v)} end),
                stages: Map.new(stages_s, fn {k, v} -> {k, error(row(stages_b[k], i), v)} end)
              }
            end
          end

        if rem(tick, 32) == 0, do: Output.puts("Compared frame #{tick}")

        %{
          tick: tick,
          rows: rows,
          isolated: isolated,
          features: if(isolated, do: Nx.to_flat_list(features)),
          controllers: actual |> :erlang.term_to_binary() |> Base.encode64()
        }
      end

    for a <- [batch | agents], do: GenServer.stop(a)

    versions =
      Map.new([:nx, :exla, :axon, :edifice, :xla], fn app ->
        Application.load(app)

        {app,
         %{
           version: to_string(Application.spec(app, :vsn) || "unknown"),
           path: to_string(:code.lib_dir(app))
         }}
      end)

    native = :exla |> :code.priv_dir() |> to_string() |> Path.join("libexla.so")

    runtime_hashes =
      ExPhil.BenchmarkSweep.hashes(
        [native] ++
          Enum.map([Nx, EXLA, Axon, Edifice.SSM.Mamba], &(&1 |> :code.which() |> to_string()))
      )

    report = %{
      client: EXLA.Client.default_name(),
      versions: versions,
      runtime_hashes: runtime_hashes,
      stagger: phase?,
      events: events,
      hashes: hashes
    }

    unless hashes == ExPhil.BenchmarkSweep.hashes(sources),
      do: raise("sources changed during validation")

    out = Keyword.fetch!(opts, :out)
    File.mkdir_p!(Path.dirname(out))
    File.write!(out, Jason.encode!(report, pretty: true))
    rows = Enum.flat_map(events, & &1.rows)

    Output.puts(
      "Controller differences: #{Enum.count(rows, &(not &1.controller_equal))}/#{length(rows)}"
    )

    Output.puts(
      "Embedding differences: #{Enum.count(rows, &(&1.embedding.max_abs != 0))}/#{length(rows)}"
    )

    Output.puts("Saved #{out}")
  end
end

{opts, [], []} =
  OptionParser.parse(System.argv(), strict: [out: :string, ticks: :integer, stagger: :boolean])

MambaBatchDiagnostic.run(opts)
