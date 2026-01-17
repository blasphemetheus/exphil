defmodule ExPhil.Training.Imitation do
  @moduledoc """
  Imitation learning (behavioral cloning) from Melee replays.

  Behavioral cloning trains a policy to mimic expert actions by minimizing
  the cross-entropy loss between the policy's action distribution and the
  actions taken in replay data.

  ## Training Pipeline

  ```
  Replays (.slp) ──> Parser ──> Dataset ──> Batches ──> Training
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   Embed State   │
                                    │        +        │
                                    │  Compute Loss   │
                                    │        +        │
                                    │ Update Weights  │
                                    └─────────────────┘
  ```

  ## Usage

      # Create trainer
      trainer = Imitation.new(
        embed_size: 1024,
        hidden_sizes: [512, 512],
        learning_rate: 1.0e-4
      )

      # Train on dataset
      {:ok, trained} = Imitation.train(trainer, dataset, epochs: 10)

      # Save checkpoint
      Imitation.save_checkpoint(trained, "checkpoints/imitation_v1.axon")

  ## Frame Unrolling

  Melee has inherent frame delay (reaction time + network latency). During
  training, we can optionally unroll multiple frames to help the model learn
  to predict actions that will be executed several frames in the future.
  """

  alias ExPhil.Networks.Policy
  alias ExPhil.Embeddings

  require Logger

  defstruct [
    :policy_model,
    :policy_params,
    :optimizer,
    :optimizer_state,
    :embed_config,
    :config,
    :step,
    :metrics,
    # Cached functions for performance (built once, reused every step)
    :predict_fn,
    :apply_updates_fn
  ]

  @type t :: %__MODULE__{
    policy_model: Axon.t(),
    policy_params: map(),
    optimizer: Polaris.Updates.t(),
    optimizer_state: map(),
    embed_config: map(),
    config: map(),
    step: non_neg_integer(),
    metrics: map(),
    predict_fn: function() | nil,
    apply_updates_fn: function() | nil
  }

  # Default training configuration
  @default_config %{
    learning_rate: 1.0e-4,
    batch_size: 64,
    max_grad_norm: 1.0,
    weight_decay: 1.0e-5,
    warmup_steps: 1000,
    frame_stack: 1,        # Number of frames to stack
    frame_delay: 0,        # Frames between state and action
    log_interval: 100,
    checkpoint_interval: 1000,
    axis_buckets: 16,
    shoulder_buckets: 4,
    # MLP architecture
    hidden_sizes: [512, 512],     # MLP hidden layer sizes
    dropout: 0.1,                 # Dropout rate
    # Temporal training options
    temporal: false,              # Enable temporal/sequence training
    backbone: :sliding_window,    # :sliding_window, :hybrid, :lstm, :mlp
    window_size: 60,              # Frames in attention window
    num_heads: 4,                 # Attention heads
    head_dim: 64,                 # Dimension per head
    hidden_size: 256,             # LSTM/hybrid hidden size
    num_layers: 2                 # Attention/recurrent layers
  }

  @doc """
  Create a new imitation learning trainer.

  ## Options
    - `:embed_size` - Size of state embedding (required, or provide :embed_config)
    - `:embed_config` - Embedding configuration (alternative to embed_size)
    - `:hidden_sizes` - Policy network hidden layer sizes (for MLP backbone)
    - `:learning_rate` - Initial learning rate
    - `:batch_size` - Training batch size
    - `:max_grad_norm` - Gradient clipping threshold
    - `:temporal` - Enable temporal/sequence training (default: false)
    - `:backbone` - Temporal backbone type: :sliding_window, :hybrid, :lstm, :mlp
    - `:window_size` - Attention window size for temporal (default: 60)
    - Other options merged into config
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    # Get embedding configuration
    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config(Keyword.take(opts, [:with_speeds, :with_nana, :with_projectiles]))
    end)

    embed_size = Keyword.get(opts, :embed_size, Embeddings.embedding_size(embed_config))

    # Build config - include embed_size for export
    config = @default_config
    |> Map.merge(Map.new(Keyword.take(opts, Map.keys(@default_config))))
    |> Map.put(:embed_size, embed_size)

    # Build policy model - temporal or regular
    policy_model = if config.temporal do
      Policy.build_temporal(
        embed_size: embed_size,
        backbone: config.backbone,
        window_size: config.window_size,
        num_heads: config.num_heads,
        head_dim: config.head_dim,
        hidden_size: config.hidden_size,
        num_layers: config.num_layers,
        hidden_sizes: config.hidden_sizes,
        dropout: config.dropout,
        axis_buckets: config.axis_buckets,
        shoulder_buckets: config.shoulder_buckets
      )
    else
      Policy.build(
        embed_size: embed_size,
        hidden_sizes: config.hidden_sizes,
        dropout: config.dropout,
        axis_buckets: config.axis_buckets,
        shoulder_buckets: config.shoulder_buckets
      )
    end

    # Initialize parameters using newer Axon API
    # Use mode: :train to ensure all parameters (including dropout state) are initialized
    {init_fn, _predict_fn} = Axon.build(policy_model, mode: :train)

    # Input shape depends on temporal mode
    input_shape = if config.temporal do
      {1, config.window_size, embed_size}
    else
      {1, embed_size}
    end

    policy_params = init_fn.(Nx.template(input_shape, :f32), Axon.ModelState.empty())

    # Create optimizer with gradient clipping
    {optimizer_init, optimizer_update} = create_optimizer(config)

    # Initialize optimizer state with just the parameter data (not full ModelState)
    # This ensures consistency when calling optimizer during training
    params_data = get_params_data(policy_params)
    optimizer_state = optimizer_init.(params_data)

    # Pre-build cached functions for performance
    # These are reused every training step instead of being rebuilt
    {_init_fn, predict_fn} = Axon.build(policy_model, mode: :inference)
    apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2)

    %__MODULE__{
      policy_model: policy_model,
      policy_params: policy_params,
      optimizer: optimizer_update,
      optimizer_state: optimizer_state,
      embed_config: embed_config,
      config: config,
      step: 0,
      metrics: %{
        loss: [],
        button_loss: [],
        stick_loss: [],
        learning_rate: []
      },
      predict_fn: predict_fn,
      apply_updates_fn: apply_updates_fn
    }
  end

  @doc """
  Create optimizer with learning rate schedule and gradient clipping.

  Returns `{init_fn, update_fn}` tuple.
  """
  @spec create_optimizer(map()) :: {function(), function()}
  def create_optimizer(config) do
    # Use AdamW optimizer with constant learning rate
    # (Polaris schedules are applied differently in newer versions)
    Polaris.Optimizers.adamw(
      learning_rate: config.learning_rate,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-8,
      decay: config.weight_decay
    )
  end

  @doc """
  Train on a dataset for multiple epochs.

  ## Parameters
    - `trainer` - The trainer struct
    - `dataset` - Enumerable of {state, action} batches
    - `opts` - Options:
      - `:epochs` - Number of epochs (default: 1)
      - `:callback` - Function called after each step with metrics

  ## Returns
    Updated trainer struct with trained parameters.
  """
  @spec train(t(), Enumerable.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def train(trainer, dataset, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 1)
    callback = Keyword.get(opts, :callback, fn _metrics -> :ok end)

    try do
      final_trainer = Enum.reduce(1..epochs, trainer, fn epoch, acc ->
        Logger.info("Starting epoch #{epoch}/#{epochs}")
        train_epoch(acc, dataset, epoch, callback)
      end)

      {:ok, final_trainer}
    rescue
      e -> {:error, e}
    end
  end

  @doc """
  Train for a single epoch.
  """
  @spec train_epoch(t(), Enumerable.t(), non_neg_integer(), function()) :: t()
  def train_epoch(trainer, dataset, epoch, callback) do
    {_predict_fn, loss_fn} = build_loss_fn(trainer.policy_model)

    Enum.reduce(dataset, trainer, fn batch, acc ->
      {new_trainer, metrics} = train_step(acc, batch, loss_fn)

      # Call callback
      full_metrics = Map.merge(metrics, %{epoch: epoch, step: new_trainer.step})
      callback.(full_metrics)

      # Log periodically
      if rem(new_trainer.step, acc.config.log_interval) == 0 do
        Logger.info("Step #{new_trainer.step}: loss=#{Float.round(metrics.loss, 4)}")
      end

      new_trainer
    end)
  end

  @doc """
  Perform a single training step.

  Uses Nx.Defn.value_and_grad for proper gradient computation with Axon models.
  """
  @spec train_step(t(), map(), function()) :: {t(), map()}
  def train_step(trainer, batch, _loss_fn) do
    %{states: states, actions: actions} = batch

    # Transfer ALL tensors to default backend to avoid EXLA/Defn.Expr mismatch
    # This is necessary because Nx.Defn.value_and_grad traces with Expr tensors,
    # which cannot mix with concrete EXLA tensors captured in closures
    states = Nx.backend_copy(states)
    actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)

    # Use cached predict_fn (built once in new/1, reused every step)
    predict_fn = trainer.predict_fn

    # Copy model parameters to avoid EXLA/Expr mismatch in closures
    # The original params are stored in trainer and will be used to restore structure
    model_state = deep_backend_copy(trainer.policy_params)

    # Build a loss function that takes ModelState as argument
    loss_fn = fn params ->
      # Forward pass through the model
      # In inference mode, predict_fn returns the tuple directly
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, states)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      # Compute imitation loss
      Policy.imitation_loss(logits, actions)
    end

    # Compute loss and gradients using Nx.Defn.value_and_grad
    # ModelState implements Nx.Container so this works correctly
    {loss, grads} = Nx.Defn.value_and_grad(loss_fn).(model_state)

    # Extract data for optimizer (grads has same structure as ModelState)
    grads_data = get_params_data(grads)
    params_data = get_params_data(model_state)

    # Update parameters using the optimizer
    {updates, new_optimizer_state} = trainer.optimizer.(
      grads_data,
      trainer.optimizer_state,
      params_data
    )

    # Use cached apply_updates_fn (built once in new/1, reused every step)
    new_params_data = trainer.apply_updates_fn.(params_data, updates)
    new_params = put_params_data(model_state, new_params_data)

    # Update metrics
    loss_val = Nx.to_number(loss)
    new_metrics = %{
      loss: [loss_val | Enum.take(trainer.metrics.loss, 99)],
      button_loss: trainer.metrics.button_loss,
      stick_loss: trainer.metrics.stick_loss,
      learning_rate: trainer.metrics.learning_rate
    }

    new_trainer = %{trainer |
      policy_params: new_params,
      optimizer_state: new_optimizer_state,
      step: trainer.step + 1,
      metrics: new_metrics
    }

    {new_trainer, %{loss: loss_val, step: new_trainer.step}}
  end

  @doc """
  Build the loss function for training.
  """
  @spec build_loss_fn(Axon.t()) :: {function(), function()}
  def build_loss_fn(policy_model) do
    {_init_fn, predict_fn} = Axon.build(policy_model)

    loss_fn = fn params, states, actions ->
      # Forward pass
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, states)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      # Compute loss
      Policy.imitation_loss(logits, actions)
    end

    {predict_fn, loss_fn}
  end

  @doc """
  Evaluate on a validation dataset.
  """
  @spec evaluate(t(), Enumerable.t()) :: map()
  def evaluate(trainer, dataset) do
    {_predict_fn, loss_fn} = build_loss_fn(trainer.policy_model)

    {total_loss, count} = Enum.reduce(dataset, {0.0, 0}, fn batch, {acc_loss, acc_count} ->
      %{states: states, actions: actions} = batch
      loss = loss_fn.(trainer.policy_params, states, actions)
      {acc_loss + Nx.to_number(loss), acc_count + 1}
    end)

    avg_loss = if count > 0, do: total_loss / count, else: 0.0

    %{
      loss: avg_loss,
      num_batches: count
    }
  end

  # ============================================================================
  # Checkpointing
  # ============================================================================

  @doc """
  Save a training checkpoint.

  Tensors are converted to BinaryBackend before saving to ensure
  they can be loaded in a different process/session.
  """
  @spec save_checkpoint(t(), Path.t()) :: :ok | {:error, term()}
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

  @doc """
  Load a training checkpoint.
  """
  @spec load_checkpoint(t(), Path.t()) :: {:ok, t()} | {:error, term()}
  def load_checkpoint(trainer, path) do
    case File.read(path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)

        new_trainer = %{trainer |
          policy_params: checkpoint.policy_params,
          optimizer_state: checkpoint.optimizer_state,
          step: checkpoint.step,
          metrics: checkpoint.metrics
        }

        Logger.info("Loaded checkpoint from #{path} at step #{new_trainer.step}")
        {:ok, new_trainer}

      error ->
        error
    end
  end

  @doc """
  Export just the policy parameters for inference.

  Includes full temporal config so agents can properly reconstruct
  the model architecture and handle sequence input.
  """
  @spec export_policy(t(), Path.t()) :: :ok | {:error, term()}
  def export_policy(trainer, path) do
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    # Extract embed_size from config or compute from embed_config
    embed_size = trainer.config[:embed_size] ||
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
        num_layers: trainer.config[:num_layers] || 2
      }
    }

    File.write(path, :erlang.term_to_binary(export))
  end

  # ============================================================================
  # Inference
  # ============================================================================

  @doc """
  Get action from the trained policy.
  """
  @spec get_action(t(), Nx.Tensor.t(), keyword()) :: map()
  def get_action(trainer, state, opts \\ []) do
    {_init_fn, predict_fn} = Axon.build(trainer.policy_model)

    Policy.sample(
      trainer.policy_params,
      predict_fn,
      state,
      Keyword.merge([
        axis_buckets: trainer.config.axis_buckets,
        shoulder_buckets: trainer.config.shoulder_buckets
      ], opts)
    )
  end

  @doc """
  Get action as ControllerState for game input.
  """
  @spec get_controller_action(t(), Nx.Tensor.t(), keyword()) :: ExPhil.Bridge.ControllerState.t()
  def get_controller_action(trainer, state, opts \\ []) do
    samples = get_action(trainer, state, opts)

    Policy.to_controller_state(samples,
      axis_buckets: trainer.config.axis_buckets,
      shoulder_buckets: trainer.config.shoulder_buckets
    )
  end

  # ============================================================================
  # Metrics
  # ============================================================================

  @doc """
  Get current training metrics summary.
  """
  @spec metrics_summary(t()) :: map()
  def metrics_summary(trainer) do
    recent_losses = Enum.take(trainer.metrics.loss, 100)

    %{
      step: trainer.step,
      avg_loss: safe_mean(recent_losses),
      min_loss: safe_min(recent_losses),
      max_loss: safe_max(recent_losses)
    }
  end

  defp safe_mean([]), do: 0.0
  defp safe_mean(list), do: Enum.sum(list) / length(list)

  defp safe_min([]), do: 0.0
  defp safe_min(list), do: Enum.min(list)

  defp safe_max([]), do: 0.0
  defp safe_max(list), do: Enum.max(list)

  # Helpers for handling Axon.ModelState
  defp get_params_data(%Axon.ModelState{data: data}), do: data
  defp get_params_data(params) when is_map(params), do: params

  defp put_params_data(%Axon.ModelState{} = state, data), do: %{state | data: data}
  defp put_params_data(_original, data), do: data

  # Deep copy all tensors in a nested map/ModelState to avoid EXLA/Expr mismatch
  # NOTE: Clause order matters! Nx.Tensor and Axon.ModelState are structs (i.e. maps),
  # so they must be pattern-matched BEFORE the generic is_map guard clause.
  defp deep_backend_copy(%Nx.Tensor{} = tensor), do: Nx.backend_copy(tensor)

  defp deep_backend_copy(%Axon.ModelState{data: data} = state) do
    %{state | data: deep_backend_copy(data)}
  end

  defp deep_backend_copy(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_backend_copy(v)} end)
  end

  defp deep_backend_copy(other), do: other
end
