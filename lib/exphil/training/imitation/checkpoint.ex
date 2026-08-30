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
  @spec load_checkpoint(struct(), Path.t()) :: {:ok, struct()} | {:error, term()}
  def load_checkpoint(trainer, path) do
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
        new_trainer = %{
          trainer
          | policy_params: checkpoint.policy_params,
            optimizer_state: checkpoint.optimizer_state,
            config: checkpoint.config,
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

      {:error, reason} ->
        {:error, reason}
    end
  end

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
        # Embedding fingerprint (GUARDS_BACKLOG #1): the canary state
        # embedded through the BATCHED path with the config the agent
        # will reconstruct; the agent re-embeds through the LIVE path
        # at load and refuses on divergence.
        embed_canary: canary,
        # Trained delay-id set (2026-08-24, the untrained-id trap): the
        # live Agent refuses to deploy a delay-conditioned policy at an
        # id outside this set (bare --frame-delay 4 silently ran id4 —
        # untrained — and collapsed chaining for three decider games).
        train_delays: train_delays(trainer.config)
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

    config = %{
      ExPhil.Embeddings.Game.Config.default()
      | queue_depth: Map.get(ec, :queue_depth) || 1,
        with_delay_id: Map.get(ec, :with_delay_id) || false,
        stage_internals: Map.get(ec, :stage_internals) || false
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
  defp train_delays(config) do
    cond do
      is_list(config[:train_delays]) and config[:train_delays] != [] ->
        config[:train_delays]

      config[:frame_delay_augment] ->
        Enum.to_list((config[:frame_delay_min] || 0)..(config[:frame_delay_max] || 0))

      true ->
        [config[:frame_delay] || 0]
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
