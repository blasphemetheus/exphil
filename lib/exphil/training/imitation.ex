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

  ## Module Organization

  This module orchestrates imitation learning, delegating to submodules:

  - `Imitation.Optimizer` - Optimizer creation and LR schedules
  - `Imitation.Loss` - Loss function builders
  - `Imitation.TrainLoop` - Training loop and gradient handling
  - `Imitation.Validation` - Evaluation functions
  - `Imitation.Checkpointing` - Save/load/export functions

  ## See Also

  - `ExPhil.Training.Config` - Configuration parsing and presets
  - `ExPhil.Training.Data` - Data loading and preprocessing
  - `ExPhil.Networks.Policy` - The policy network architecture
  - `ExPhil.Embeddings.Game` - Game state embedding
  """

  alias ExPhil.Networks.Policy
  alias ExPhil.Embeddings
  alias ExPhil.Training.MixedPrecision
  alias ExPhil.Training.Imitation.{Checkpointing, Loss, Optimizer, TrainLoop, Validation}

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
    :apply_updates_fn,
    # Compiled loss+grad function - avoids closure creation and deep_backend_copy every batch
    :loss_and_grad_fn,
    # Compiled eval loss function - for validation without gradient computation
    :eval_loss_fn,
    # Mixed precision state (FP32 master weights + BF16 compute params)
    :mixed_precision_state
  ]

  @type t :: %__MODULE__{
          policy_model: Axon.t(),
          policy_params: map(),
          optimizer: term(),
          optimizer_state: map(),
          embed_config: map(),
          config: map(),
          step: non_neg_integer(),
          metrics: map(),
          predict_fn: function() | nil,
          apply_updates_fn: function() | nil,
          eval_loss_fn: function() | nil,
          loss_and_grad_fn: function() | nil,
          mixed_precision_state: MixedPrecision.t() | nil
        }

  # Default training configuration
  @default_config %{
    learning_rate: 1.0e-4,
    batch_size: 64,
    max_grad_norm: 1.0,
    weight_decay: 1.0e-5,
    warmup_steps: 1000,
    # Number of frames to stack
    frame_stack: 1,
    # Frames between state and action
    frame_delay: 0,
    log_interval: 100,
    checkpoint_interval: 1000,
    axis_buckets: 16,
    shoulder_buckets: 4,
    # MLP architecture
    # MLP hidden layer sizes
    hidden_sizes: [512, 512],
    # Dropout rate
    dropout: 0.1,
    # Precision (bf16 = ~2x faster, minimal accuracy loss)
    precision: :bf16,
    # Mixed precision training (FP32 master weights + BF16 compute)
    # More numerically stable than pure BF16 - preserves small gradients
    mixed_precision: false,
    # Temporal training options
    # Enable temporal/sequence training
    temporal: false,
    # :sliding_window, :hybrid, :lstm, :mlp
    backbone: :sliding_window,
    # Frames in attention window
    window_size: 60,
    # Attention heads
    num_heads: 4,
    # Dimension per head
    head_dim: 64,
    # LSTM/hybrid hidden size
    hidden_size: 256,
    # Attention/recurrent layers
    num_layers: 2,
    # nil = full BPTT, integer = truncate to last N steps
    truncate_bptt: nil,
    # Mamba-specific options
    # Mamba SSM state dimension
    state_size: 16,
    # Mamba expansion factor
    expand_factor: 2,
    # Mamba conv kernel size
    conv_size: 4,
    # Jamba-specific (Mamba + Attention hybrid)
    # Insert attention layer every N layers
    attention_every: 3,
    # Gradient checkpointing (trade compute for memory)
    # Enable gradient checkpointing for memory efficiency
    gradient_checkpoint: false,
    # Checkpoint every N layers (1 = every layer)
    checkpoint_every: 1,
    # Layer normalization for MLP backbone
    # Add layer norm after each dense layer
    layer_norm: false,
    # Gradient accumulation
    # 1 = no accumulation, N = effective batch = batch_size * N
    accumulation_steps: 1,
    # Label smoothing
    # 0.0 = no smoothing, 0.1 = typical value
    label_smoothing: 0.0,
    # Optimizer selection
    # :adam, :adamw, :lamb, :radam, :sgd, :rmsprop
    optimizer: :adamw,
    # K-means stick discretization
    # Path or tensor of K-means centers (nil = uniform buckets)
    kmeans_centers: nil
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
    - `:precision` - Tensor precision: :bf16 (default, ~2x faster) or :f32
    - `:temporal` - Enable temporal/sequence training (default: false)
    - `:backbone` - Temporal backbone type: :sliding_window, :hybrid, :lstm, :mlp
    - `:window_size` - Attention window size for temporal (default: 60)
    - Other options merged into config
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    # Get embedding configuration
    embed_config =
      Keyword.get_lazy(opts, :embed_config, fn ->
        Embeddings.config(Keyword.take(opts, [:with_speeds, :with_nana, :with_projectiles]))
      end)

    embed_size = Keyword.get(opts, :embed_size, Embeddings.embedding_size(embed_config))

    # Build config - include embed_size for export
    config =
      @default_config
      |> Map.merge(Map.new(Keyword.take(opts, Map.keys(@default_config))))
      |> Map.put(:embed_size, embed_size)

    # Load K-means centers if path provided
    config = load_kmeans_centers(config)

    # Build policy model - temporal or regular
    policy_model =
      if config.temporal do
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
          shoulder_buckets: config.shoulder_buckets,
          truncate_bptt: config.truncate_bptt,
          layer_norm: config.layer_norm,
          # Jamba-specific
          attention_every: config.attention_every,
          # Jamba stability options (Pre-LN + QK LayerNorm prevent NaN)
          pre_norm: Map.get(config, :pre_norm, true),
          qk_layernorm: Map.get(config, :qk_layernorm, true),
          # Gradient checkpointing for memory efficiency
          gradient_checkpoint: Map.get(config, :gradient_checkpoint, false),
          checkpoint_every: Map.get(config, :checkpoint_every, 1)
        )
      else
        Policy.build(
          embed_size: embed_size,
          hidden_sizes: config.hidden_sizes,
          dropout: config.dropout,
          axis_buckets: config.axis_buckets,
          shoulder_buckets: config.shoulder_buckets,
          layer_norm: config.layer_norm
        )
      end

    # Initialize parameters using newer Axon API
    # Use mode: :train to ensure all parameters (including dropout state) are initialized
    {init_fn, _predict_fn} = Axon.build(policy_model, mode: :train)

    # Input shape depends on temporal mode
    input_shape =
      if config.temporal do
        {1, config.window_size, embed_size}
      else
        {1, embed_size}
      end

    # When using mixed precision, initialize params in FP32 (master weights)
    # Otherwise use configured precision
    init_precision = if config.mixed_precision, do: :f32, else: config.precision
    policy_params = init_fn.(Nx.template(input_shape, init_precision), Axon.ModelState.empty())

    # Initialize mixed precision state if enabled
    # This maintains FP32 master weights while computing in BF16
    mixed_precision_state =
      if config.mixed_precision do
        compute_precision = config.precision || :bf16
        MixedPrecision.init(get_params_data(policy_params), precision: compute_precision)
      else
        nil
      end

    # Create optimizer with gradient clipping
    {optimizer_init, optimizer_update} = Optimizer.create_optimizer(config)

    # Initialize optimizer state with just the parameter data (not full ModelState)
    # For mixed precision, use FP32 master weights; otherwise use policy_params
    params_data =
      if mixed_precision_state do
        MixedPrecision.get_master_params(mixed_precision_state)
      else
        get_params_data(policy_params)
      end

    optimizer_state = optimizer_init.(params_data)

    # Pre-build cached functions for performance
    # These are reused every training step instead of being rebuilt
    {_init_fn, predict_fn} = Axon.build(policy_model, mode: :inference)

    # JIT compile optimizer and update functions with EXLA for GPU execution
    # Without compiler: EXLA, these default to CPU which causes 0% GPU utilization
    optimizer_fn = Nx.Defn.jit(optimizer_update, compiler: EXLA)
    apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2, compiler: EXLA)

    # Build compiled loss+grad function (avoids deep_backend_copy every batch)
    # This function is JITted once and reused for all training steps
    loss_and_grad_fn = Loss.build_loss_and_grad_fn(predict_fn, config)

    # Build compiled eval loss function (for validation - no gradients needed)
    # JITted once and reused for all validation batches
    eval_loss_fn = Loss.build_eval_loss_fn(predict_fn, config)

    %__MODULE__{
      policy_model: policy_model,
      policy_params: policy_params,
      optimizer: optimizer_fn,
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
      apply_updates_fn: apply_updates_fn,
      loss_and_grad_fn: loss_and_grad_fn,
      eval_loss_fn: eval_loss_fn,
      mixed_precision_state: mixed_precision_state
    }
  end

  @doc """
  Warm up JIT-compiled functions to avoid latency during training/inference.

  EXLA/XLA compiles functions lazily on first use with specific tensor shapes.
  This can cause 5-15 second delays during training when a new code path is hit.
  Calling warmup/2 before training triggers all JIT compilations upfront.

  ## What gets warmed up

  - `:all` (default) - All functions below
  - `:training` - loss_and_grad_fn (forward + backward pass)
  - `:validation` - eval_loss_fn via evaluate() (forward pass + accumulation)
  - `:inference` - predict_fn (forward pass only)

  ## Example

      trainer = Imitation.new(opts)
      sample_batch = Enum.take(dataset, 1) |> hd()

      # Warm up everything before training
      {:ok, warmup_times} = Imitation.warmup(trainer, sample_batch)
      # => {:ok, %{training: 8500, validation: 5200, inference: 1200}}

      # Or warm up specific functions
      {:ok, _} = Imitation.warmup(trainer, sample_batch, only: [:validation])

  ## Returns

  `{:ok, timing_map}` where timing_map has milliseconds for each warmed function.
  """
  @spec warmup(t(), map(), keyword()) :: {:ok, map()} | {:error, term()}
  def warmup(trainer, sample_batch, opts \\ []) do
    only = Keyword.get(opts, :only, [:all])
    show_progress = Keyword.get(opts, :show_progress, true)

    targets =
      if :all in only do
        [:training, :validation, :inference]
      else
        only
      end

    timings =
      Enum.reduce(targets, %{}, fn target, acc ->
        {time_ms, _result} = :timer.tc(fn -> warmup_target(trainer, sample_batch, target) end, :millisecond)

        if show_progress do
          IO.write(:stderr, "    ✓ #{target} JIT compiled (#{Float.round(time_ms / 1000, 1)}s)\n")
        end

        Map.put(acc, target, time_ms)
      end)

    {:ok, timings}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Quick warmup with just validation (most common use case).

  Equivalent to `warmup(trainer, sample_batch, only: [:validation])`.
  """
  @spec warmup_validation(t(), list(), keyword()) :: {:ok, map()} | {:error, term()}
  def warmup_validation(trainer, validation_batches, opts \\ []) when is_list(validation_batches) do
    # Use same max_concurrency as actual validation to warm up the right code path
    # Default to 4 which matches evaluate/3 default
    max_concurrency = Keyword.get(opts, :max_concurrency, 4)

    # full: true processes ALL validation batches, which warms up not just JIT but also
    # whatever GPU state/memory effects occur after training JIT compilation.
    # This moves the ~3.5s overhead from first validation to warmup phase.
    full = Keyword.get(opts, :full, false)

    warmup_batches =
      if full do
        validation_batches
      else
        # Take enough batches to exercise the parallel path (at least max_concurrency batches)
        Enum.take(validation_batches, max(2, max_concurrency))
      end

    {time_ms, _result} = :timer.tc(fn ->
      evaluate(trainer, warmup_batches, show_progress: false, max_concurrency: max_concurrency)
    end, :millisecond)

    {:ok, %{validation: time_ms}}
  end

  # Private warmup implementations for each target
  defp warmup_target(trainer, batch, :training) do
    %{states: states, actions: actions} = batch

    if trainer.loss_and_grad_fn do
      # Run one forward+backward pass
      {_loss, _grads} = trainer.loss_and_grad_fn.(trainer.policy_params, states, actions)
    end

    :ok
  end

  defp warmup_target(trainer, batch, :validation) do
    # Run through evaluate() with 1 batch to warm up entire validation path
    # including closure wrapper, Nx.add accumulation, Nx.to_number
    evaluate(trainer, [batch], show_progress: false, max_concurrency: 1)
    :ok
  end

  defp warmup_target(trainer, batch, :inference) do
    %{states: states} = batch

    if trainer.predict_fn do
      # Run one forward pass
      _predictions = trainer.predict_fn.(trainer.policy_params, states)
    end

    :ok
  end

  # Delegate optimizer creation to Optimizer submodule
  defdelegate create_optimizer(config), to: Optimizer

  # Delegate training functions to TrainLoop submodule
  defdelegate train(trainer, dataset, opts \\ []), to: TrainLoop
  defdelegate train_epoch(trainer, dataset, epoch, callback), to: TrainLoop

  defdelegate train_step(trainer, batch, loss_fn), to: TrainLoop

  # Delegate loss function builders to Loss submodule
  # Note: build_loss_fn has optional opts argument
  def build_loss_fn(policy_model, opts \\ []), do: Loss.build_loss_fn(policy_model, opts)
  defdelegate build_loss_and_grad_fn(predict_fn, config), to: Loss
  defdelegate build_eval_loss_fn(predict_fn, config), to: Loss

  # Delegate evaluation functions to Validation submodule
  def evaluate(trainer, dataset, opts \\ []), do: Validation.evaluate(trainer, dataset, opts)
  defdelegate evaluate_batch(trainer, batch), to: Validation

  # ============================================================================
  # Checkpointing - delegated to Checkpointing submodule
  # ============================================================================

  defdelegate save_checkpoint(trainer, path), to: Checkpointing
  def save_checkpoint_async(trainer, path, opts \\ []),
    do: Checkpointing.save_checkpoint_async(trainer, path, opts)
  defdelegate get_optimizer_step(optimizer_state), to: Checkpointing
  defdelegate load_checkpoint(trainer, path), to: Checkpointing
  defdelegate export_policy(trainer, path), to: Checkpointing

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
      Keyword.merge(
        [
          axis_buckets: trainer.config.axis_buckets,
          shoulder_buckets: trainer.config.shoulder_buckets
        ],
        opts
      )
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
      shoulder_buckets: trainer.config.shoulder_buckets,
      kmeans_centers: trainer.config[:kmeans_centers_tensor]
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

  # Helper for handling Axon.ModelState
  defp get_params_data(%Axon.ModelState{data: data}), do: data
  defp get_params_data(params) when is_map(params), do: params

  # Load K-means centers from file path and update config with tensor and axis_size
  defp load_kmeans_centers(%{kmeans_centers: nil} = config), do: config

  defp load_kmeans_centers(%{kmeans_centers: path} = config) when is_binary(path) do
    alias ExPhil.Embeddings.KMeans

    case KMeans.load(path) do
      {:ok, centers} ->
        k = Nx.axis_size(centers, 0)
        # K-means gives us k clusters directly (no +1 like buckets)
        config
        |> Map.put(:kmeans_centers_tensor, centers)
        |> Map.put(:axis_size, k)
        # For compatibility with existing code
        |> Map.put(:axis_buckets, k - 1)

      {:error, reason} ->
        raise "Failed to load K-means centers from #{path}: #{inspect(reason)}"
    end
  end

  defp load_kmeans_centers(%{kmeans_centers: %Nx.Tensor{} = centers} = config) do
    k = Nx.axis_size(centers, 0)

    config
    |> Map.put(:kmeans_centers_tensor, centers)
    |> Map.put(:axis_size, k)
    |> Map.put(:axis_buckets, k - 1)
  end

  defp load_kmeans_centers(config), do: config
end
