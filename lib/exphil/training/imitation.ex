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
    :metrics
  ]

  @type t :: %__MODULE__{
    policy_model: Axon.t(),
    policy_params: map(),
    optimizer: Polaris.Updates.t(),
    optimizer_state: map(),
    embed_config: map(),
    config: map(),
    step: non_neg_integer(),
    metrics: map()
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
    shoulder_buckets: 4
  }

  @doc """
  Create a new imitation learning trainer.

  ## Options
    - `:embed_size` - Size of state embedding (required, or provide :embed_config)
    - `:embed_config` - Embedding configuration (alternative to embed_size)
    - `:hidden_sizes` - Policy network hidden layer sizes
    - `:learning_rate` - Initial learning rate
    - `:batch_size` - Training batch size
    - `:max_grad_norm` - Gradient clipping threshold
    - Other options merged into config
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    # Get embedding configuration
    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config(Keyword.take(opts, [:with_speeds, :with_nana, :with_projectiles]))
    end)

    embed_size = Keyword.get(opts, :embed_size, Embeddings.embedding_size(embed_config))

    # Build config
    config = @default_config
    |> Map.merge(Map.new(Keyword.take(opts, Map.keys(@default_config))))

    # Build policy model
    policy_model = Policy.build(
      embed_size: embed_size,
      hidden_sizes: Keyword.get(opts, :hidden_sizes, [512, 512]),
      axis_buckets: config.axis_buckets,
      shoulder_buckets: config.shoulder_buckets
    )

    # Initialize parameters using newer Axon API
    {init_fn, _predict_fn} = Axon.build(policy_model)
    policy_params = init_fn.(Nx.template({1, embed_size}, :f32), Axon.ModelState.empty())

    # Create optimizer with gradient clipping
    {optimizer_init, optimizer_update} = create_optimizer(config)

    # Initialize optimizer state
    optimizer_state = optimizer_init.(policy_params)

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
      }
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
  """
  @spec train_step(t(), map(), function()) :: {t(), map()}
  def train_step(trainer, batch, loss_fn) do
    %{states: states, actions: actions} = batch

    # Extract params data from ModelState if needed
    params_data = get_params_data(trainer.policy_params)

    # Compute gradients using value_and_grad
    grad_fn = fn params ->
      # Reconstruct ModelState for forward pass
      full_params = put_params_data(trainer.policy_params, params)
      loss_fn.(full_params, states, actions)
    end

    {loss, grads} = Nx.Defn.value_and_grad(grad_fn).(params_data)

    # Update parameters using Polaris
    {updates, new_optimizer_state} = trainer.optimizer.(
      grads,
      trainer.optimizer_state,
      params_data
    )

    new_params_data = Polaris.Updates.apply_updates(params_data, updates)
    new_params = put_params_data(trainer.policy_params, new_params_data)

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
  """
  @spec save_checkpoint(t(), Path.t()) :: :ok | {:error, term()}
  def save_checkpoint(trainer, path) do
    checkpoint = %{
      policy_params: trainer.policy_params,
      optimizer_state: trainer.optimizer_state,
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
  """
  @spec export_policy(t(), Path.t()) :: :ok | {:error, term()}
  def export_policy(trainer, path) do
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    export = %{
      params: trainer.policy_params,
      config: %{
        axis_buckets: trainer.config.axis_buckets,
        shoulder_buckets: trainer.config.shoulder_buckets
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
end
