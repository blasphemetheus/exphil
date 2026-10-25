# Offline, fixed-input validation: recurrence parity is a correctness gate;
# carry/window disagreement is a behavioral measurement, not a failure.
# devenv shell -- env EXLA_TARGET=host EXPHIL_EXLA_PRECISION=highest mix run \
#   scripts/validate_recurrent_carry.exs --policy PATH --replay PATH --out PATH
defmodule RecurrentCarryValidation do
  alias ExPhil.Training.{Data, Utils}

  def run(opts) do
    client =
      case Keyword.get(opts, :client, "host") do
        "host" -> :host
        "cuda" -> :cuda
        other -> raise("unsupported client #{other}")
      end

    Application.put_env(
      :exla,
      :clients,
      Keyword.put(Application.get_env(:exla, :clients, []), :host, platform: :host)
    )

    Nx.default_backend({EXLA.Backend, client: client})
    Nx.Defn.default_options(compiler: EXLA, client: client, precision: :highest)
    path = Keyword.fetch!(opts, :policy)
    {:ok, export} = ExPhil.Training.load_policy(path)
    c = export.config
    arch = if c.backbone == :mamba_2, do: :mamba_ssd, else: c.backbone
    module = Edifice.module_for(arch)
    # These exports use raw categorical IDs, without learned preprocessing.
    if c[:action_embed_size] || c[:character_embed_size],
      do: raise("learned embedding preprocessing is not supported by this probe")

    if c[:button_events] || c[:stick_events],
      do: raise("event-head masking is not supported by this probe")

    replay_path = Keyword.fetch!(opts, :replay)
    {:ok, meta} = ExPhil.Data.Peppi.metadata(replay_path)
    character = Keyword.get(opts, :character, "fox")

    own =
      Enum.find(meta.players, &(String.downcase(&1.character_name || "") == character)) ||
        raise("replay has no #{character}")

    opp = Enum.find(meta.players, &(&1.port != own.port))
    {:ok, replay} = ExPhil.Data.Peppi.parse(replay_path, player_port: own.port)
    n = Keyword.get(opts, :frames, 320)

    frames =
      replay
      |> ExPhil.Data.Peppi.to_training_frames(
        player_port: own.port,
        opponent_port: opp.port,
        remap_ports: true
      )
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.drop(Keyword.get(opts, :offset, 0))
      |> Enum.take(n)

    sidecar = Path.join(Path.dirname(path), "model_config.json")
    json = if File.exists?(sidecar), do: Jason.decode!(File.read!(sidecar)), else: %{}
    ds = Data.from_frames(frames)

    cfg = %{
      ds.embed_config
      | stage_internals: c[:stage_internals] || false,
        with_projectiles: Map.get(c, :with_projectiles, true),
        queue_depth: c[:queue_depth] || 1,
        with_delay_id: c[:with_delay_id] || false
    }

    frames = Enum.map(frames, &Map.put(&1, :delay_id, Keyword.get(opts, :delay_id, 2)))

    ds =
      %{Data.from_frames(frames) | embed_config: cfg}
      |> Data.precompute_frame_embeddings(
        use_prev_action: c[:use_prev_action] || false,
        prev_action_quantize: json["prev_action_quantize"] in [true, "true"],
        show_progress: false
      )

    x =
      if is_list(ds.embedded_frames),
        do: Nx.concatenate(ds.embedded_frames),
        else: ds.embedded_frames

    {count, width} = Nx.shape(x)
    unless width == c.embed_size, do: raise("embedding width #{width} != #{c.embed_size}")
    if count < c.window_size, do: raise("replay shorter than training window")
    x = Nx.new_axis(x, 0)

    params =
      export.params
      |> Utils.ensure_model_state()
      |> Nx.backend_copy({EXLA.Backend, client: client})

    build_opts = [
      embed_dim: width,
      hidden_size: c.hidden_size,
      num_layers: c.num_layers,
      dropout: 0.0,
      state_size: c[:state_size] || 16,
      expand_factor: c[:expand_factor] || 2,
      conv_size: c[:conv_size] || 4,
      recurrent_state: if(c[:bptt], do: :zeros, else: :legacy_random)
    ]

    initial =
      Edifice.init_state(arch, params, build_opts)
      |> Nx.backend_copy({EXLA.Backend, client: client})

    step = Edifice.Stateful.jit_step(module, EXLA, client: client, precision: :highest)

    full = fn len ->
      model = Edifice.build(arch, Keyword.put(build_opts, :window_size, len))

      {_, predict} =
        Axon.build(model, mode: :inference, compiler: EXLA, client: client, precision: :highest)

      predict
    end

    window_predict = full.(c.window_size)
    # Match deployed cold padding for window-trained policies; BPTT consumes once.
    pad = if c[:bptt], do: 0, else: c.window_size - 1

    padded =
      if pad > 0,
        do:
          Nx.concatenate(
            [Nx.broadcast(Nx.slice_along_axis(x, 0, 1, axis: 1), {1, pad, width}), x],
            axis: 1
          ),
        else: x

    # Full scans are unrolled by some backends. Bound compilation size while
    # still measuring carried behavior over arbitrarily longer replay clips.
    prefix_limit = max(c.window_size, Keyword.get(opts, :prefix_limit, 320))

    checkpoints =
      Enum.uniq([
        c.window_size,
        min(2 * c.window_size, min(count, prefix_limit)),
        min(count, prefix_limit)
      ])

    prefix_predictions =
      Map.new(checkpoints, fn t ->
        IO.puts("Compiling #{arch} prefix #{t + pad}")
        {t, full.(t + pad).(params, Nx.slice_along_axis(padded, 0, t + pad, axis: 1))}
      end)

    warmed =
      Enum.reduce(0..(pad - 1)//1, initial, fn _, s ->
        elem(step.(params, s, Nx.take(x, 0, axis: 1)), 1)
      end)

    head_model =
      ExPhil.Networks.Policy.Heads.build_controller_head(
        Axon.input("features", shape: {nil, c.hidden_size}),
        c[:axis_buckets] || 16,
        c[:shoulder_buckets] || 4
      )

    {_, head_predict} =
      Axon.build(head_model,
        mode: :inference,
        compiler: EXLA,
        client: client,
        precision: :highest
      )

    sample = fn features ->
      action =
        if c[:head] == :autoregressive do
          ExPhil.Networks.Policy.sample_autoregressive_from_features(params, features,
            deterministic: true
          )
        else
          ExPhil.Networks.Policy.sample(params, head_predict, features, deterministic: true)
        end

      action
      |> Map.take([:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder])
      |> Map.new(fn {k, v} -> {k, Nx.to_flat_list(v)} end)
    end

    {rows, _} =
      Enum.map_reduce(1..count, {warmed, warmed}, fn t, {carry, resync} ->
        f = Nx.take(x, t - 1, axis: 1)
        resync_now = t > c.window_size and rem(t - c.window_size - 1, 30) == 0

        resync =
          if resync_now do
            Enum.reduce((t - c.window_size)..(t - 2), initial, fn i, s ->
              elem(step.(params, s, Nx.take(x, i, axis: 1)), 1)
            end)
          else
            resync
          end

        {a, carry} = step.(params, carry, f)
        {b, resync} = step.(params, resync, f)

        row =
          if t >= c.window_size do
            w =
              window_predict.(
                params,
                Nx.slice_along_axis(x, t - c.window_size, c.window_size, axis: 1)
              )

            if resync_now, do: Nx.Testing.assert_all_close(b, w, atol: 1.0e-4, rtol: 1.0e-4)

            prefix_error =
              if p = prefix_predictions[t] do
                Nx.Testing.assert_all_close(a, p, atol: 1.0e-4, rtol: 1.0e-4)
                delta(a, p)
              end

            # State snapshot must replay identically, including convolution history.
            {r1, _} = step.(params, carry, f)
            {r2, _} = step.(params, Edifice.Stateful.snapshot(carry), f)
            unless Nx.to_binary(r1) == Nx.to_binary(r2), do: raise("rollback mismatch")
            wa = sample.(w)

            %{
              frame: t,
              prefix_max_abs: prefix_error,
              carry_max_abs: delta(a, w),
              resync_max_abs: delta(b, w),
              carry_action_equal: if(wa, do: sample.(a) == wa),
              resync_action_equal: if(wa, do: sample.(b) == wa)
            }
          end

        {row, {carry, resync}}
      end)

    rows = Enum.reject(rows, &is_nil/1)

    result = %{
      policy: path,
      replay: replay_path,
      character: character,
      offset: Keyword.get(opts, :offset, 0),
      architecture: arch,
      frames: count,
      window: c.window_size,
      bptt: c[:bptt] || false,
      precision: "highest",
      client: client,
      padding: pad,
      prefix_lengths: Enum.map(checkpoints, &(&1 + pad)),
      note:
        "Recorded observations and expert previous actions fixed across modes; not closed-loop gameplay. Resync counterfactual rebuilds every 30 frames after first full window. For BPTT, window/resync truncate trained memory and are diagnostic only. Action agreement uses deterministic decoding of the checkpoint head.",
      rollback_exact: true,
      rows: rows,
      summary: %{
        prefix_max_abs: rows |> Enum.map(&(&1.prefix_max_abs || 0)) |> Enum.max(),
        carry_max_abs: rows |> Enum.map(& &1.carry_max_abs) |> Enum.max(),
        resync_max_abs: rows |> Enum.map(& &1.resync_max_abs) |> Enum.max(),
        carry_action_agreement: Enum.count(rows, &(&1.carry_action_equal == true)) / length(rows),
        resync_action_agreement:
          Enum.count(rows, &(&1.resync_action_equal == true)) / length(rows)
      }
    }

    out = Keyword.fetch!(opts, :out)
    File.mkdir_p!(Path.dirname(out))
    File.write!(out, Jason.encode!(result, pretty: true), [:exclusive])
    IO.inspect(result.summary, label: "#{arch} carried-state validation")
  end

  defp delta(a, b), do: a |> Nx.subtract(b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
end

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      client: :string,
      prefix_limit: :integer,
      policy: :string,
      replay: :string,
      out: :string,
      frames: :integer,
      delay_id: :integer,
      character: :string,
      offset: :integer
    ]
  )

RecurrentCarryValidation.run(opts)
