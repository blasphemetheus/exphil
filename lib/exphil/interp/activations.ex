defmodule ExPhil.Interp.Activations do
  @moduledoc """
  Activation capture for interpretability (INTERP_ROADMAP Phase 0).

  Runs an exported policy's trunk (backbone) over replay frames and returns
  the `[n, hidden]` representation the controller heads read, aligned
  frame-for-frame with `ExPhil.Interp.GroundTruth` labels.

  The pipeline mirrors training exactly (same `Data.from_frames` →
  `precompute_frame_embeddings` → windowed sequences), minus expert
  relabeling, action-delay shifting, and prev-action dropout — activations
  must be deterministic and reflect what the policy actually sees at
  inference.

  Alignment: with `stride: 1`, sequence `i` covers frames `i..i+window-1`
  and its activation corresponds to a decision AT frame `i + window - 1`.
  `capture_replay/3` returns that mapping explicitly via `:frame_offset`.

  ## Example

      trunk = Activations.load_trunk("checkpoints/mewtwo_combo_poolgrow_r1_policy.bin")
      %{activations: acts, labels: labels} =
        Activations.capture_replay(trunk, "~/Slippi/Game_x.slp")
      # acts: {n_windows, 256} f32; labels.opp_knockdown: {n_windows} u8
  """

  alias ExPhil.Data.Peppi
  alias ExPhil.Interp.GroundTruth
  alias ExPhil.Networks
  alias ExPhil.Training
  alias ExPhil.Training.{Data, Utils}

  @doc """
  Load an exported policy `.bin` and build its trunk-only predict function.

  Returns a map with `:predict_fn`, `:params`, `:config`, `:window`,
  `:hidden_size` — pass it to `capture_replay/3`.

  Only temporal autoregressive policies are supported (all drill policies
  are); raises for non-temporal exports.
  """
  def load_trunk(policy_path, opts \\ []) do
    {:ok, %{params: params, config: config}} = Training.load_policy(policy_path)

    unless Map.get(config, :temporal, false) do
      raise ArgumentError,
            "#{policy_path} is not a temporal policy — trunk capture only supports temporal policies"
    end

    trunk =
      Networks.Policy.build_temporal_trunk(
        embed_size: Map.fetch!(config, :embed_size),
        backbone: Map.get(config, :backbone, :mlp),
        window_size: Map.get(config, :window_size, 60),
        num_heads: Map.get(config, :num_heads, 4),
        head_dim: Map.get(config, :head_dim, 64),
        hidden_size: Map.get(config, :hidden_size, 256),
        num_layers: Map.get(config, :num_layers, 2),
        state_size: Map.get(config, :state_size, 16),
        expand_factor: Map.get(config, :expand_factor, 2),
        conv_size: Map.get(config, :conv_size, 4),
        dropout: Map.get(config, :dropout, 0.1)
      )

    {init_fn, predict_fn} = Utils.build_compiled(trunk, mode: :inference)

    # init: :random — same architecture, fresh random params. The Phase 1
    # control: probe accuracy on a random-init trunk measures what the
    # (windowed) input alone + random projection gives, so trained-policy
    # probe scores must clear it to mean anything.
    params =
      case Keyword.get(opts, :init, :export) do
        :export ->
          Utils.ensure_model_state(params)

        :random ->
          window = Map.get(config, :window_size, 60)
          embed_size = Map.fetch!(config, :embed_size)
          init_fn.(Nx.template({1, window, embed_size}, :f32), Axon.ModelState.empty())
      end

    %{
      predict_fn: predict_fn,
      params: params,
      config: config,
      window: Map.get(config, :window_size, 60),
      hidden_size: Map.get(config, :hidden_size, 256)
    }
  end

  @doc """
  Load the FULL policy (trunk + six controller heads) for per-head logit
  capture (P0 "better tier"; feeds P3 attribution/ablation). The capture
  yields a map of logit tensors per decision frame instead of a single
  trunk activation matrix: buttons {n, 8} (order a,b,x,y,z,l,r,d_up),
  main_x/main_y/c_x/c_y {n, axis_buckets+1}, shoulder {n, shoulder_buckets+1}.
  """
  def load_heads(policy_path) do
    {:ok, %{params: params, config: config}} = Training.load_policy(policy_path)

    unless Map.get(config, :temporal, false) do
      raise ArgumentError,
            "#{policy_path} is not a temporal policy — head capture only supports temporal policies"
    end

    model =
      Networks.Policy.build_temporal(
        embed_size: Map.fetch!(config, :embed_size),
        backbone: Map.get(config, :backbone, :mlp),
        window_size: Map.get(config, :window_size, 60),
        num_heads: Map.get(config, :num_heads, 4),
        head_dim: Map.get(config, :head_dim, 64),
        hidden_size: Map.get(config, :hidden_size, 256),
        num_layers: Map.get(config, :num_layers, 2),
        state_size: Map.get(config, :state_size, 16),
        expand_factor: Map.get(config, :expand_factor, 2),
        conv_size: Map.get(config, :conv_size, 4),
        dropout: Map.get(config, :dropout, 0.1),
        axis_buckets: Map.get(config, :axis_buckets, 16),
        shoulder_buckets: Map.get(config, :shoulder_buckets, 4)
      )

    {_init_fn, predict_fn} = Utils.build_compiled(model, mode: :inference)

    %{
      kind: :heads,
      predict_fn: predict_fn,
      params: Utils.ensure_model_state(params),
      config: config,
      window: Map.get(config, :window_size, 60),
      hidden_size: Map.get(config, :hidden_size, 256)
    }
  end

  @doc """
  Load ONLY the six controller heads as a model over trunk-space inputs
  `{nil, hidden}` — for running heads on modified (e.g. LEACE-erased)
  trunk activations offline. Head layer names are absolute
  ("buttons_hidden" etc.), so exported params load directly.
  """
  def load_heads_only(policy_path) do
    {:ok, %{params: params, config: config}} = Training.load_policy(policy_path)

    hidden = Map.get(config, :hidden_size, 256)
    input = Axon.input("trunk", shape: {nil, hidden})

    model =
      ExPhil.Networks.Policy.Heads.build_controller_head(
        input,
        Map.get(config, :axis_buckets, 16),
        Map.get(config, :shoulder_buckets, 4)
      )

    {_init_fn, predict_fn} = Utils.build_compiled(model, mode: :inference)

    %{
      kind: :heads_only,
      predict_fn: predict_fn,
      params: Utils.ensure_model_state(params),
      config: config,
      hidden_size: hidden
    }
  end

  @doc """
  A pseudo-trunk that returns the raw last-frame embedding of each window —
  the Phase 1 input floor: what a linear probe can decode from the current
  frame's embedding with no learned representation at all.
  """
  def input_trunk(opts \\ []) do
    window = Keyword.get(opts, :window, 60)
    embed_size = Keyword.get(opts, :embed_size, 288)
    use_prev_action = Keyword.get(opts, :use_prev_action, true)

    %{
      predict_fn: fn _params, states ->
        states
        |> Nx.slice_along_axis(window - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      params: %{},
      config: %{use_prev_action: use_prev_action},
      window: window,
      hidden_size: embed_size
    }
  end

  @doc """
  Capture trunk activations + aligned ground-truth labels for one replay.

  ## Options
    - `:player_port` - the policy's perspective (default 1)
    - `:opponent_port` - default 2
    - `:batch_size` - forward-pass batch size (default 256)
    - `:labels` - compute ground-truth labels too (default true)

  Returns `%{activations, labels, frame_offset, n, replay}` where
  `activations` is `{n, hidden}` f32, each `labels` entry is `{n}`, and
  `frame_offset` is the index (into the post-`frame >= 0` frame list) of the
  decision frame for row 0 (= window - 1).
  """
  def capture_replay(trunk, replay_path, opts \\ []) do
    player_port = Keyword.get(opts, :player_port, 1)
    opponent_port = Keyword.get(opts, :opponent_port, 2)
    batch_size = Keyword.get(opts, :batch_size, 256)
    with_labels = Keyword.get(opts, :labels, true)

    path = Path.expand(replay_path)
    {:ok, replay} = Peppi.parse(path)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: player_port, opponent_port: opponent_port)
      |> Enum.reject(&(&1.game_state.frame < 0))

    window = trunk.window

    # :use_prev_action override enables the prev-action ABLATION: capturing
    # with false embeds nil prev_action = a zeroed slot of the same layout
    # (agent.ex semantics), i.e. the policy sees no self-conditioning.
    use_prev_action =
      Keyword.get(opts, :use_prev_action, Map.get(trunk.config, :use_prev_action, false))

    # Cached embeddings (task #26): keyed on (replay, embed-config,
    # prev-action regime) — we re-embedded the same dozen replays ~30x/day
    # before this. Ablated captures (use_prev_action false) get their own
    # cache entries via the key's use_prev_action component.
    dataset =
      frames
      |> Data.from_frames()
      |> Data.precompute_frame_embeddings_cached(
        cache: true,
        replay_files: [path],
        use_prev_action: use_prev_action,
        prev_action_dropout: 0.0,
        show_progress: false
      )

    # The generic batch->forward->collect loop lives in edifice now
    # (Edifice.Interpretability.Capture, the audit-gap-3 port); this
    # module keeps the melee half: replays, embeddings (+cache), labels.
    batches =
      dataset
      |> Data.batched_sequences(
        batch_size: batch_size,
        window_size: window,
        stride: 1,
        lazy: true,
        shuffle: false,
        drop_last: false
      )
      # Stream, not Enum: one batch on device at a time
      |> Stream.map(& &1.states)

    # Heads trunks return a 6-tuple; adapt to Capture's tensor-or-map
    # contract (atom keys pass through untouched)
    predict_fn =
      case Map.get(trunk, :kind, :trunk) do
        :heads ->
          fn params, states ->
            {buttons, main_x, main_y, c_x, c_y, shoulder} = trunk.predict_fn.(params, states)

            %{
              buttons: buttons,
              main_x: main_x,
              main_y: main_y,
              c_x: c_x,
              c_y: c_y,
              shoulder: shoulder
            }
          end

        _ ->
          trunk.predict_fn
      end

    activations =
      case Edifice.Interpretability.Capture.run(predict_fn, trunk.params, batches) do
        %{"output" => t} -> t
        m -> m
      end

    n =
      case activations do
        %Nx.Tensor{} = t -> Nx.axis_size(t, 0)
        m -> m |> Map.values() |> hd() |> Nx.axis_size(0)
      end

    labels =
      if with_labels do
        frames
        |> GroundTruth.frame_labels(opponent_port: opponent_port)
        |> GroundTruth.align_to_windows(window, n)
      else
        nil
      end

    %{
      activations: activations,
      labels: labels,
      frame_offset: window - 1,
      n: n,
      replay: path
    }
  end

  @doc """
  Capture several replays with one trunk and concatenate the results.

  Returns `%{activations, labels, replay_index, replays}` — `replay_index`
  is a `{n}` s64 tensor mapping each row to its source replay's position in
  `replays` (needed for by-replay train/val splits in probing: frames within
  a game are heavily correlated, so never split within a replay).
  """
  def capture(trunk, replay_paths, opts \\ []) do
    captures = Enum.map(replay_paths, &capture_replay(trunk, &1, opts))

    replay_index =
      captures
      |> Enum.with_index()
      |> Enum.flat_map(fn {c, i} -> List.duplicate(i, c.n) end)
      |> Nx.tensor(type: :s64, backend: Nx.BinaryBackend)

    labels =
      case captures do
        [%{labels: nil} | _] ->
          nil

        _ ->
          captures
          |> Enum.map(& &1.labels)
          |> Enum.reduce(fn m, acc ->
            Map.new(acc, fn {k, v} -> {k, Nx.concatenate([v, Map.fetch!(m, k)])} end)
          end)
      end

    activations =
      case captures do
        # NB: %Nx.Tensor{} guard first — bare %{} also matches tensors
        [%{activations: %Nx.Tensor{}} | _] ->
          captures |> Enum.map(& &1.activations) |> Nx.concatenate(axis: 0)

        _ ->
          # heads captures: map of per-head logit tensors
          captures
          |> Enum.map(& &1.activations)
          |> Enum.reduce(fn m, acc ->
            Map.new(acc, fn {k, v} -> {k, Nx.concatenate([v, Map.fetch!(m, k)])} end)
          end)
      end

    %{
      activations: activations,
      labels: labels,
      replay_index: replay_index,
      replays: Enum.map(captures, & &1.replay)
    }
  end
end
