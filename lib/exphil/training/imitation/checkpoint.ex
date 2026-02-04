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
  @spec save_checkpoint(struct(), Path.t()) :: :ok | {:error, term()}
  def save_checkpoint(trainer, path) do
    # Convert all tensors to BinaryBackend for serialization
    checkpoint = %{
      policy_params: to_binary_backend(trainer.policy_params),
      optimizer_state: to_binary_backend(trainer.optimizer_state),
      config: trainer.config,
      step: trainer.step,
      metrics: trainer.metrics
    }

    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    case File.write(path, :erlang.term_to_binary(checkpoint)) do
      :ok ->
        Logger.info("Saved checkpoint to #{path}")
        :ok

      error ->
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
  - `{:error, :queue_full}` if save queue is full

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
  @spec save_checkpoint_async(struct(), Path.t(), keyword()) :: :ok | {:error, :queue_full}
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

    case Checkpoint.load(path, current_embed_size: current_embed_size) do
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

    # Extract embed_size from config or compute from embed_config
    embed_size =
      trainer.config[:embed_size] ||
        (trainer.embed_config && Embeddings.embedding_size(trainer.embed_config))

    export = %{
      # Convert params to BinaryBackend for serialization
      params: to_binary_backend(trainer.policy_params),
      config: %{
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
        conv_size: trainer.config[:conv_size] || 4
      }
    }

    File.write(path, :erlang.term_to_binary(export))
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
