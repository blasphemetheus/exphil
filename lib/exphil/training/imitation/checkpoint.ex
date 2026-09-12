defmodule ExPhil.Training.Imitation.Checkpointing do
  @moduledoc """
  Checkpoint save/load/export functions for imitation learning.

  Handles serialization of training state including:
  - Policy parameters (neural network weights)
  - Optimizer state (momentum, adaptive learning rate accumulators)
  - Training configuration
  - Step count and metrics history

  ## Tensor Backend Conversion

  EXLA tensors cannot be serialized directly. All tensors are converted to
  `Nx.BinaryBackend` before saving to ensure they can be loaded in different
  processes or sessions.

  ## See Also

  - `ExPhil.Training.Imitation` - Main imitation learning module
  - `ExPhil.Training.Checkpoint` - Low-level checkpoint utilities
  - `ExPhil.Training.AsyncCheckpoint` - Background checkpoint saving
  """

  alias ExPhil.Training.Checkpoint
  alias ExPhil.Embeddings
  alias ExPhil.Error.CheckpointError

  require Logger

  # ============================================================================
  # Save Functions
  # ============================================================================

  @doc """
  Save a training checkpoint.

  Tensors are converted to BinaryBackend before saving to ensure
  they can be loaded in a different process/session.

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `path` - File path to save checkpoint to

  ## Returns

  - `:ok` on success
  - `{:error, term()}` on failure
  """
  @spec save_checkpoint(struct(), Path.t(), keyword()) :: :ok | {:error, term()}
  def save_checkpoint(trainer, path, opts \\ []) do
    # Convert all tensors to BinaryBackend for serialization
    checkpoint = %{
      policy_params: to_binary_backend(trainer.policy_params),
      optimizer_state: to_binary_backend(trainer.optimizer_state),
      config: trainer.config,
      step: trainer.step,
      metrics: trainer.metrics
    }

    # Optional caller metadata (e.g. dagger resume snapshots store
    # %{epoch: e, fingerprint: f}); load_checkpoint/2 ignores it, callers
    # peek it via Training.Checkpoint.load/2 + Map.get(:meta).
    checkpoint =
      case Keyword.get(opts, :meta) do
        nil -> checkpoint
        meta -> Map.put(checkpoint, :meta, meta)
      end

    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    # Write-then-rename: atomic on the same filesystem, so an interrupt
    # (reboot killed r13 mid-write territory, 2026-07-18) can never leave
    # a truncated checkpoint at the published path.
    tmp = path <> ".tmp"

    case File.write(tmp, :erlang.term_to_binary(checkpoint)) do
      :ok ->
        File.rename!(tmp, path)
        Logger.info("Saved checkpoint to #{path}")
        :ok

      error ->
        File.rm(tmp)
        error
    end
  end

  @doc """
  Save a training checkpoint asynchronously.

  Like `save_checkpoint/2` but returns immediately while the checkpoint
  is written in the background. This prevents training from blocking
  on disk I/O.

  Requires `ExPhil.Training.AsyncCheckpoint` to be started (typically
  in your application's supervision tree).

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `path` - File path to save checkpoint to
  - `opts` - Options:
    - `:timeout` - Max time to wait if save queue is full (default: 5000ms)

  ## Returns

  - `:ok` if queued successfully
  - `{:error, %CheckpointError{reason: :queue_full}}` if save queue is full

  ## Example

      # Add to your application.ex supervision tree:
      children = [
        ExPhil.Training.AsyncCheckpoint,
        # ... other children
      ]

      # Then in training:
      :ok = Checkpointing.save_checkpoint_async(trainer, path)

      # At end of training, wait for pending saves:
      :ok = ExPhil.Training.AsyncCheckpoint.await_pending()
  """
  @spec save_checkpoint_async(struct(), Path.t(), keyword()) :: :ok | {:error, CheckpointError.t()}
  def save_checkpoint_async(trainer, path, opts \\ []) do
    # Build checkpoint map (no need to convert to BinaryBackend here,
    # AsyncCheckpoint does that internally to handle cross-process access)
    checkpoint = %{
      policy_params: trainer.policy_params,
      optimizer_state: trainer.optimizer_state,
      config: trainer.config,
      step: trainer.step,
      metrics: trainer.metrics
    }

    ExPhil.Training.AsyncCheckpoint.save_async(checkpoint, path, opts)
  end

  # ============================================================================
  # Load Functions
  # ============================================================================

  @doc """
  Load a training checkpoint.

  Validates embed size if the trainer was initialized with one.
  Warns if checkpoint embed size differs from current config.
  Also validates that optimizer step count matches trainer.step.

  ## Parameters

  - `trainer` - The imitation trainer struct to update
  - `path` - File path to load checkpoint from

  ## Returns

  - `{:ok, updated_trainer}` on success
  - `{:error, term()}` on failure
  """
  @spec load_checkpoint(struct(), Path.t(), keyword()) :: {:ok, struct()} | {:error, term()}
  def load_checkpoint(trainer, path, opts \\ []) do
    current_embed_size = trainer.config[:embed_size]

    # A TRAINING resume with a different embedding width cannot succeed —
    # the first batch dies inside EXLA compile (`cannot reshape {296} to
    # {1,1,288}`), 80 s and one poisoned 1.2 GB cache entry later. Fail here
    # instead. Observed 2026-08-28: the AWBC arms dropped --stage-internals
    # and the warn-only guard let B1 start. (Guards fail loud and safe.)
    case Checkpoint.load(path,
           current_embed_size: current_embed_size,
           error_on_mismatch: true
         ) do
      {:ok, checkpoint} ->
        trainer_head = normalize_head(trainer.config[:head])
        ckpt_head = normalize_head(get_in_config(checkpoint.config, :head))
        reinit_head = Keyword.get(opts, :reinit_head, false)

        if trainer_head != ckpt_head or reinit_head do
          transplant_trunk(trainer, checkpoint, path,
            trainer_head: trainer_head,
            ckpt_head: ckpt_head,
            reinit_head: reinit_head
          )
        else
          full_resume(trainer, checkpoint, path)
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp full_resume(trainer, checkpoint, path) do
    new_trainer = %{
      trainer
      | policy_params: checkpoint.policy_params,
        optimizer_state: checkpoint.optimizer_state,
        config: Map.merge(checkpoint.config, Map.take(trainer.config, [:label_delay, :frame_delay, :action_delay, :label_convention])),
        step: checkpoint.step,
        metrics: checkpoint.metrics
    }

    # Validate optimizer step matches trainer step
    case get_optimizer_step(new_trainer.optimizer_state) do
      nil ->
        Logger.warning("Could not verify optimizer step count")

      opt_step when opt_step != new_trainer.step ->
        Logger.warning(
          "Optimizer step count (#{opt_step}) differs from trainer step (#{new_trainer.step}). " <>
            "LR schedule may not continue correctly."
        )

      _ ->
        :ok
    end

    Logger.info("Loaded checkpoint from #{path} at step #{new_trainer.step}")
    {:ok, new_trainer}
  end

  # Layer-name prefixes that belong to a controller HEAD (either kind).
  # Everything else is trunk (+ embeddings) and is safe to transplant.
  @head_prefixes ~w(buttons_ main_x_ main_y_ c_x_ c_y_ shoulder_ ar_)

  # Trunk transplant: resume the TRUNK from a checkpoint whose head differs
  # from the trainer's (e.g. `--resume ep10.axon --head autoregressive`), or
  # whose head is being deliberately re-initialised (`--reinit-head`, the
  # v1.1-IND control of AUTOREGRESSIVE_HEAD_PLAN item 9).
  #
  # Semantics (all deliberate, all logged):
  # - Trunk/embedding params: loaded from the checkpoint (name + shape match).
  # - Head params: keep the trainer's fresh init (the checkpoint either lacks
  #   them entirely — ar_* — or is being intentionally re-rolled).
  # - Optimizer state: FRESH. The checkpoint's optimizer tree matches the old
  #   param tree, not the new one; loading it would crash or silently misapply.
  # - config/step/metrics: keep the trainer's. This is the fix for the silent
  #   head clobber (checkpoint.config[:head] overwriting `--head`) — guard #6's
  #   flag-drop class, caught in the 08-30 precheck before it burned a run.
  defp transplant_trunk(trainer, checkpoint, path, info) do
    fresh_params = trainer.policy_params
    fresh_data = params_data(fresh_params)
    ckpt_data = params_data(checkpoint.policy_params)

    {loadable, skipped_shape} =
      ckpt_data
      |> Enum.filter(fn {name, _} -> Map.has_key?(fresh_data, name) and not head_layer?(name) end)
      |> Enum.split_with(fn {name, params} -> shapes_match?(params, fresh_data[name]) end)

    if loadable == [] do
      {:error,
       {:transplant_no_overlap,
        "no non-head layer of #{path} matches the current model — wrong checkpoint or architecture"}}
    else
      dropped_head = Enum.count(ckpt_data, fn {name, _} -> head_layer?(name) end)
      fresh_head = Enum.count(fresh_data, fn {name, _} -> head_layer?(name) end)

      if skipped_shape != [] do
        Logger.warning(
          "[Checkpoint] transplant: #{length(skipped_shape)} trunk layer(s) SKIPPED on shape " <>
            "mismatch (#{skipped_shape |> Enum.map(&elem(&1, 0)) |> Enum.take(5) |> Enum.join(", ")}) — " <>
            "these keep their fresh init; verify this is intended"
        )
      end

      merged_params = put_params_data(fresh_params, Map.merge(fresh_data, Map.new(loadable)))

      Logger.warning(
        "[Checkpoint] TRUNK TRANSPLANT from #{path}: head #{inspect(info[:ckpt_head])} -> " <>
          "#{inspect(info[:trainer_head])}#{if info[:reinit_head], do: " (reinit-head)", else: ""} | " <>
          "#{length(loadable)} trunk layer(s) loaded, #{fresh_head} head layer(s) fresh, " <>
          "#{dropped_head} checkpoint head layer(s) dropped | optimizer FRESH, step reset to #{trainer.step}"
      )

      {:ok, %{trainer | policy_params: merged_params}}
    end
  end

  defp head_layer?(name), do: Enum.any?(@head_prefixes, &String.starts_with?(name, &1))

  defp normalize_head(nil), do: :independent
  defp normalize_head(head) when is_atom(head), do: head
  defp normalize_head(head) when is_binary(head), do: String.to_existing_atom(head)

  # Checkpoint config may be a map with atom OR string keys after round-trips.
  defp get_in_config(config, key) when is_map(config),
    do: Map.get(config, key, Map.get(config, to_string(key)))

  defp get_in_config(config, key) when is_list(config), do: Keyword.get(config, key)
  defp get_in_config(_, _), do: nil

  # Params may be an Axon.ModelState (data: %{layer => %{param => tensor}})
  # or a bare map of the same shape (older checkpoints).
  defp params_data(%Axon.ModelState{data: data}), do: data
  defp params_data(params) when is_map(params), do: params

  defp put_params_data(%Axon.ModelState{} = state, data), do: %{state | data: data}
  defp put_params_data(params, data) when is_map(params), do: data

  # NOTE: the Nx.Tensor clause must come FIRST — a tensor is a struct, and
  # structs satisfy is_map/1, so a bare is_map clause would swallow it.
  defp shapes_match?(%Nx.Tensor{} = a, %Nx.Tensor{} = b), do: Nx.shape(a) == Nx.shape(b)

  defp shapes_match?(a, b)
       when is_map(a) and is_map(b) and not is_struct(a) and not is_struct(b) do
    Map.keys(a) |> Enum.sort() == Map.keys(b) |> Enum.sort() and
      Enum.all?(a, fn {k, v} -> shapes_match?(v, b[k]) end)
  end

  defp shapes_match?(_, _), do: false

  # ============================================================================
  # Export Functions
  # ============================================================================

  @doc """
  Export just the policy parameters for inference.

  Includes full temporal config so agents can properly reconstruct
  the model architecture and handle sequence input.

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `path` - File path to export policy to

  ## Returns

  - `:ok` on success
  - `{:error, term()}` on failure
  """
  @spec export_policy(struct(), Path.t()) :: :ok | {:error, term()}
  def export_policy(trainer, path) do
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    # embed_size (guard #6, GUARDS_BACKLOG): the PARAMS are the single
    # source of truth — both the trainer.config scalar and embed_config
    # have lied before (0825 pilot: config scalar said 296 while the
    # params were 288-wide and the agent died at warmup; conversely a
    # trainer built with an explicit :embed_size carries a default
    # embed_config that overstates the width). The input layer's kernel
    # leading dim IS the trained width, so export whichever candidate
    # actually appears in the params — and refuse to write a checkpoint
    # whose metadata no param tensor can corroborate.
    computed_embed_size =
      trainer.embed_config && Embeddings.embedding_size(trainer.embed_config)

    stored_embed_size = trainer.config[:embed_size]

    embed_size =
      resolve_embed_size!(trainer.policy_params, computed_embed_size, stored_embed_size, path)

    canary = embed_canary(trainer)
    export_embed_config = trainer.embed_config || Embeddings.config([])

    if is_list(canary) and canary != [] and length(canary) != embed_size do
      require Logger

      Logger.warning(
        "[Checkpoint] embed canary length #{length(canary)} != embed_size #{embed_size} — " <>
          "the live Agent prefers the canary length as input width, so this checkpoint " <>
          "will NOT deploy at the trained width (non-default embedding opts are outside " <>
          "the canary's reconstruction surface)"
      )
    end

    config = %{
        # Discretization
        axis_buckets: trainer.config.axis_buckets,
        shoulder_buckets: trainer.config.shoulder_buckets,
        # MLP architecture
        embed_size: embed_size,
        hidden_sizes: trainer.config[:hidden_sizes] || [512, 512],
        dropout: trainer.config[:dropout] || 0.1,
        # Temporal config
        temporal: trainer.config[:temporal] || false,
        bptt: trainer.config[:bptt] || false,
        unroll: trainer.config[:unroll] || 80,
        frame_delay: trainer.config[:frame_delay] || 0,
        action_delay: trainer.config[:action_delay] || 0,
        label_delay: ExPhil.Data.LabelConvention.reaction_delay(trainer.config),
        num_player_names: export_embed_config.num_player_names,
        action_mode: export_embed_config.player.action_mode,
        character_mode: export_embed_config.player.character_mode,
        nana_mode: export_embed_config.player.nana_mode,
        stage_mode: export_embed_config.stage_mode,
        backbone: trainer.config[:backbone] || :mlp,
        window_size: trainer.config[:window_size] || 60,
        num_heads: trainer.config[:num_heads] || 4,
        head_dim: trainer.config[:head_dim] || 64,
        hidden_size: trainer.config[:hidden_size] || 256,
        num_layers: trainer.config[:num_layers] || 2,
        # Mamba-specific config
        state_size: trainer.config[:state_size] || 16,
        expand_factor: trainer.config[:expand_factor] || 2,
        conv_size: trainer.config[:conv_size] || 4,
        # Controller head (AUTOREGRESSIVE_HEAD_PLAN): the live agent must
        # dispatch sampling on this — :autoregressive checkpoints have
        # ar_* head params and NO buttons_hidden/main_x_hidden layers.
        head: trainer.config[:head] || :independent,
        # Embedding regime: the live agent must feed its own outputs back
        # into the prev-action channel iff the model trained with it
        use_prev_action: trainer.config[:use_prev_action] || false,
        # Queue-as-input layout (2026-07-31): the live agent must rebuild
        # the exact channel layout (K committed-action slots + delay
        # one-hot). Missing keys here cost a silent 288-vs-336 embed
        # mismatch that only surfaced at live warmup.
        queue_depth:
          (trainer.embed_config && Map.get(trainer.embed_config, :queue_depth)) || 1,
        with_delay_id:
          (trainer.embed_config && Map.get(trainer.embed_config, :with_delay_id)) || false,
        # Stage internals (W4 2026-08-24): FoD heights + PS transform in
        # the embedding — the live agent must rebuild the same layout.
        stage_internals:
          (trainer.embed_config && Map.get(trainer.embed_config, :stage_internals)) || false,
        # Bucketized action frame (player-level layout key, 2026-09-09):
        # N one-hot dims per player; the Agent must rebuild the same layout.
        action_frame_buckets: embed_action_frame_buckets(trainer.embed_config),
        # INVARIANTS.md item 4 — the source-channel stamps belong HERE (the
        # policy's own metadata is what the Agent, the probes and
        # eval_model read; found missing on v16f 2026-09-09: every consumer
        # defaulted to a 296-wide layout for a 264-wide model).
        with_projectiles:
          (trainer.embed_config && Map.get(trainer.embed_config, :with_projectiles)) || false,
        with_items: (trainer.embed_config && Map.get(trainer.embed_config, :with_items)) || false,
        provided_channels: ExPhil.Data.Peppi.provides(),
        # Embedding fingerprint (GUARDS_BACKLOG #1): the canary state
        # embedded through the BATCHED path with the config the agent
        # will reconstruct; the agent re-embeds through the LIVE path
        # at load and refuses on divergence.
        embed_canary: canary,
        # Trained delay-id set (2026-08-24, the untrained-id trap): the
        # live Agent refuses to deploy a delay-conditioned policy at an
        # id outside this set (bare --frame-delay 4 silently ran id4 —
        # untrained — and collapsed chaining for three decider games).
        train_delays: train_delays(trainer.config),
        # INVARIANTS.md item 1: the label pairing these delays are counted
        # in (unstamped = legacy :producing). See ExPhil.Data.LabelConvention.
        label_convention: ExPhil.Data.LabelConvention.current()
      }

    # Edifice manifest format (task #16): Nx.serialize params + embedded
    # Edifice.Spec. Self-describing (the spec carries the build opts the
    # network was ACTUALLY built with — no silent-default fallbacks at
    # load), shape-validatable via Edifice.Checkpoint.validate_shapes!,
    # and ~3-5x faster to (de)serialize than Erlang term format.
    # Training.Checkpoint.load_policy reads BOTH this and the legacy
    # term_to_binary format (r1-r10 checkpoints stay loadable).
    # external: exphil's policy (backbone trunk + 6 autoregressive heads)
    # is a composite owned by exphil, not an edifice registry arch — exphil
    # rebuilds it from config; the spec still carries opts + provenance.
    spec = Edifice.Spec.new(:exphil_policy, Map.to_list(config), external: true)

    Edifice.Checkpoint.save(to_binary_backend(trainer.policy_params), path,
      spec: spec,
      metadata: %{config: config}
    )
  end

  # ============================================================================
  # Optimizer State Utilities
  # ============================================================================

  @doc """
  Extract the optimizer's internal step count.

  The optimizer state tracks steps internally for LR scheduling.
  This should match `trainer.step` after proper save/load.

  Returns the step count or nil if the state structure is unexpected.

  ## Optimizer State Structure

  When using gradient clipping with an optimizer (via `Polaris.Updates.compose`),
  the state is wrapped in an extra tuple:

      {{clip_state, optimizer_state}}

  Where:
  - `clip_state` has `:count` for clip step tracking
  - `optimizer_state` (e.g., AdamW) has `:count`, `:mu`, `:nu`
  """
  @spec get_optimizer_step(tuple()) :: non_neg_integer() | nil
  def get_optimizer_step(optimizer_state) do
    case optimizer_state do
      # Composed optimizer (gradient clipping + base optimizer)
      {{_clip_state, inner_state}} when is_map(inner_state) ->
        case inner_state[:count] do
          %Nx.Tensor{} = count -> Nx.to_number(count)
          _ -> nil
        end

      # Direct optimizer (no composition)
      %{count: %Nx.Tensor{} = count} ->
        Nx.to_number(count)

      _ ->
        nil
    end
  end

  # ============================================================================
  # Private Helpers - Binary Backend Conversion
  # ============================================================================

  # Recursively convert all tensors to BinaryBackend for serialization
  # Canary fingerprint with the RECONSTRUCTED config (default struct +
  # the flat keys the agent reads back) — v1 scope: covers the
  # reconstruction surface, which is every historical burn (block
  # order, id spaces, queue layout, stage gating). Failure to embed
  # must never fail a SAVE — store nil and let load skip.
  defp embed_canary(trainer) do
    ec = trainer.embed_config || %{}

    default = ExPhil.Embeddings.Game.Config.default()

    # INVARIANTS.md item 4 (found 2026-09-09 on v16f: canary 296 vs model
    # 264): the source-channel keys MUST be in the canary config too, or
    # the stored fingerprint is a different width than the model and the
    # Agent refuses the checkpoint at its trained width.
    config = %{
      default
      | queue_depth: Map.get(ec, :queue_depth) || 1,
        with_delay_id: Map.get(ec, :with_delay_id) || false,
        stage_internals: Map.get(ec, :stage_internals) || false,
        with_projectiles: Map.get(ec, :with_projectiles, true),
        with_items: Map.get(ec, :with_items, false),
        player: %{default.player | action_frame_buckets: embed_action_frame_buckets(ec)}
    }

    ExPhil.Embeddings.Canary.fingerprint_batched(config)
  rescue
    e ->
      require Logger
      Logger.warning("[Checkpoint] embed canary failed at save (#{inspect(e)}) — storing nil")
      nil
  end

  # Guard #6 core: collect the leading dim of every rank>=2 tensor in
  # the params tree — the input layer's kernel leading dim is the true
  # trained width, so the exported embed_size MUST appear in this set.
  # Prefer the embed_config-computed candidate, fall back to the
  # trainer.config scalar, refuse the export if neither matches (an
  # export with lying metadata is worse than a failed export; the
  # trainer state is still live and re-exportable after a fix).
  defp resolve_embed_size!(params, computed, stored, path) do
    widths = tensor_leading_dims(params, MapSet.new())
    require Logger

    cond do
      MapSet.size(widths) == 0 ->
        # No rank>=2 tensors to corroborate against — nothing to lint
        computed || stored

      is_integer(computed) and MapSet.member?(widths, computed) ->
        if is_integer(stored) and stored != computed do
          Logger.warning(
            "[Checkpoint] trainer.config embed_size #{stored} disagrees with " <>
              "embed_config's #{computed} — exporting the computed value (params agree)"
          )
        end

        computed

      is_integer(stored) and MapSet.member?(widths, stored) ->
        if is_integer(computed) and computed != stored do
          Logger.warning(
            "[Checkpoint] embed_config claims width #{computed} but the params were " <>
              "built #{stored} wide (explicit :embed_size trainer) — exporting #{stored}"
          )
        end

        stored

      true ->
        raise ArgumentError,
              "[Checkpoint guard #6] refusing to export #{path}: neither embed_config's " <>
                "width (#{inspect(computed)}) nor trainer.config's (#{inspect(stored)}) is " <>
                "the leading dim of ANY param tensor (observed leading dims: " <>
                "#{widths |> Enum.sort() |> Enum.join(", ")}) — the params were built at a " <>
                "different input width than the metadata claims (0825 pilot class: config " <>
                "said 296, params were 288)"
    end
  end

  defp tensor_leading_dims(%Nx.Tensor{} = t, acc) do
    case Nx.shape(t) do
      shape when tuple_size(shape) >= 2 -> MapSet.put(acc, elem(shape, 0))
      _ -> acc
    end
  end

  defp tensor_leading_dims(%Axon.ModelState{data: data}, acc),
    do: tensor_leading_dims(data, acc)

  defp tensor_leading_dims(map, acc) when is_map(map) and not is_struct(map) do
    Enum.reduce(map, acc, fn {_k, v}, a -> tensor_leading_dims(v, a) end)
  end

  defp tensor_leading_dims(list, acc) when is_list(list) do
    Enum.reduce(list, acc, &tensor_leading_dims/2)
  end

  defp tensor_leading_dims(tuple, acc) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.reduce(acc, &tensor_leading_dims/2)
  end

  defp tensor_leading_dims(_other, acc), do: acc

  # The delay-id set this training run exposed the policy to. Priority:
  # an explicit :train_delays (dagger_drill's --multi-delay list — the
  # 0824 replicate sweeps all gate-FAILED because this helper missed
  # that pathway and stamped [0], which the untrained-id guard then
  # dutifully enforced), else the frame-delay-augment range, else the
  # single configured delay.
  # Player-level layout key: lives under embed_config.player (a
  # %Game.Config{} struct at train time, or a flattened map on old exports).
  defp embed_action_frame_buckets(nil), do: 0

  defp embed_action_frame_buckets(ec) do
    case Map.get(ec, :player) do
      %{action_frame_buckets: n} when is_integer(n) -> n
      _ -> Map.get(ec, :action_frame_buckets) || 0
    end
  end

  defp train_delays(config) do
    cond do
      is_list(config[:train_delays]) and config[:train_delays] != [] ->
        config[:train_delays]

      config[:frame_delay_augment] ->
        base = ExPhil.Data.LabelConvention.reaction_delay(config)
        Enum.map((config[:frame_delay_min] || 0)..(config[:frame_delay_max] || 0), &(&1 + base))

      true ->
        [ExPhil.Data.LabelConvention.reaction_delay(config)]
    end
  end

  defp to_binary_backend(%Nx.Tensor{} = tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  defp to_binary_backend(%Axon.ModelState{data: data, state: state} = ms) do
    %{ms | data: to_binary_backend(data), state: to_binary_backend(state)}
  end

  defp to_binary_backend(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, to_binary_backend(v)} end)
  end

  defp to_binary_backend(list) when is_list(list) do
    Enum.map(list, &to_binary_backend/1)
  end

  defp to_binary_backend(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&to_binary_backend/1)
    |> List.to_tuple()
  end

  defp to_binary_backend(other), do: other
end
