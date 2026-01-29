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
  alias ExPhil.Training.{Checkpoint, MixedPrecision, Utils}

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
    # Mixed precision state (FP32 master weights + BF16 compute params)
    :mixed_precision_state
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
          apply_updates_fn: function() | nil,
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
    {optimizer_init, optimizer_update} = create_optimizer(config)

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
    loss_and_grad_fn = build_loss_and_grad_fn(predict_fn, config)

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
      mixed_precision_state: mixed_precision_state
    }
  end

  @doc """
  Create optimizer with learning rate schedule and gradient clipping.

  Supports multiple optimizers:
  - `:adam` - Standard Adam optimizer
  - `:adamw` - AdamW (decoupled weight decay, default)
  - `:lamb` - LAMB (good for large batch training)
  - `:radam` - Rectified Adam (more stable early training)
  - `:sgd` - Stochastic gradient descent with momentum
  - `:rmsprop` - RMSprop optimizer

  Supports multiple learning rate schedules:
  - `:constant` - Fixed learning rate (default)
  - `:cosine` - Cosine decay from initial LR to 0
  - `:exponential` - Exponential decay by factor of 0.95 per epoch
  - `:linear` - Linear decay to 0 over decay_steps

  Returns `{init_fn, update_fn}` tuple.
  """
  @spec create_optimizer(map()) :: {function(), function()}
  def create_optimizer(config) do
    lr_schedule = build_lr_schedule(config)
    max_grad_norm = config[:max_grad_norm] || 1.0
    optimizer_type = config[:optimizer] || :adamw

    # Build the base optimizer
    base_optimizer = build_base_optimizer(optimizer_type, lr_schedule, config)

    # Optionally compose with gradient clipping
    if max_grad_norm > 0 do
      clip = Polaris.Updates.clip_by_global_norm(max_norm: max_grad_norm)
      Polaris.Updates.compose(clip, base_optimizer)
    else
      base_optimizer
    end
  end

  defp build_base_optimizer(:adam, lr_schedule, _config) do
    Polaris.Optimizers.adam(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-8
    )
  end

  defp build_base_optimizer(:adamw, lr_schedule, config) do
    Polaris.Optimizers.adamw(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-8,
      decay: config.weight_decay
    )
  end

  defp build_base_optimizer(:lamb, lr_schedule, config) do
    Polaris.Optimizers.lamb(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-6,
      decay: config.weight_decay
    )
  end

  defp build_base_optimizer(:radam, lr_schedule, _config) do
    Polaris.Optimizers.radam(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-8
    )
  end

  defp build_base_optimizer(:sgd, lr_schedule, _config) do
    Polaris.Optimizers.sgd(
      learning_rate: lr_schedule,
      momentum: 0.9
    )
  end

  defp build_base_optimizer(:rmsprop, lr_schedule, _config) do
    Polaris.Optimizers.rmsprop(
      learning_rate: lr_schedule,
      centered: false,
      momentum: 0.0
    )
  end

  # Build the learning rate schedule based on config
  # Polaris schedules are defn functions that can't be composed directly,
  # so we only use them when there's no warmup. With warmup, we create
  # custom schedule functions.
  # Cosine restarts always uses custom schedule since Polaris doesn't support it.
  defp build_lr_schedule(config) do
    base_lr = config.learning_rate
    schedule_type = config[:lr_schedule] || :constant
    warmup_steps = config[:warmup_steps] || 0
    decay_steps = config[:decay_steps]
    restart_period = config[:restart_period] || 1000
    restart_mult = config[:restart_mult] || 2

    # Cosine restarts needs custom implementation regardless of warmup
    if warmup_steps > 0 or schedule_type == :cosine_restarts do
      # Use custom schedule with warmup and/or restarts
      build_custom_schedule(
        base_lr,
        schedule_type,
        warmup_steps,
        decay_steps,
        restart_period,
        restart_mult
      )
    else
      # Use Polaris schedules directly (no warmup, no restarts)
      build_polaris_schedule(base_lr, schedule_type, decay_steps)
    end
  end

  # Use simple schedule functions (Polaris.Schedules has compatibility issues with Nx 0.10)
  defp build_polaris_schedule(base_lr, schedule_type, decay_steps) do
    # Pre-convert to tensors with BinaryBackend to avoid EXLA closure issues
    base_lr_t = Nx.tensor(base_lr, type: :f32, backend: Nx.BinaryBackend)

    case schedule_type do
      :constant ->
        fn _step -> base_lr_t end

      :cosine ->
        steps = decay_steps || 10_000
        steps_t = Nx.tensor(steps, type: :f32, backend: Nx.BinaryBackend)
        pi_t = Nx.tensor(:math.pi(), type: :f32, backend: Nx.BinaryBackend)
        half_t = Nx.tensor(0.5, type: :f32, backend: Nx.BinaryBackend)
        one_t = Nx.tensor(1.0, type: :f32, backend: Nx.BinaryBackend)

        fn step ->
          step_f = Nx.as_type(step, :f32)
          progress = Nx.min(Nx.divide(step_f, steps_t), one_t)
          cosine_decay = Nx.multiply(half_t, Nx.add(one_t, Nx.cos(Nx.multiply(pi_t, progress))))
          Nx.multiply(base_lr_t, cosine_decay)
        end

      :exponential ->
        transition_steps = decay_steps || 1000
        transition_t = Nx.tensor(transition_steps, type: :f32, backend: Nx.BinaryBackend)
        rate_t = Nx.tensor(0.95, type: :f32, backend: Nx.BinaryBackend)

        fn step ->
          step_f = Nx.as_type(step, :f32)
          num_decays = Nx.floor(Nx.divide(step_f, transition_t))
          decay_factor = Nx.pow(rate_t, num_decays)
          Nx.multiply(base_lr_t, decay_factor)
        end

      :linear ->
        steps = decay_steps || 10_000
        steps_t = Nx.tensor(steps, type: :f32, backend: Nx.BinaryBackend)
        one_t = Nx.tensor(1.0, type: :f32, backend: Nx.BinaryBackend)
        zero_t = Nx.tensor(0.0, type: :f32, backend: Nx.BinaryBackend)

        fn step ->
          step_f = Nx.as_type(step, :f32)
          progress = Nx.min(Nx.divide(step_f, steps_t), one_t)
          decay_factor = Nx.subtract(one_t, progress)
          Nx.max(Nx.multiply(base_lr_t, decay_factor), zero_t)
        end
    end
  end

  # Build a custom schedule function that implements warmup + decay
  # This is needed because Polaris schedules can't be composed
  defp build_custom_schedule(
         base_lr,
         schedule_type,
         warmup_steps,
         decay_steps,
         restart_period,
         restart_mult
       ) do
    # Pre-convert to tensors with Nx.BinaryBackend to avoid EXLA closure issues
    base_lr_t = Nx.tensor(base_lr, type: :f32, backend: Nx.BinaryBackend)
    warmup_steps_t = Nx.tensor(warmup_steps, type: :f32, backend: Nx.BinaryBackend)
    decay_steps_t = Nx.tensor(decay_steps || 10_000, type: :f32, backend: Nx.BinaryBackend)
    zero_t = Nx.tensor(0.0, type: :f32, backend: Nx.BinaryBackend)
    one_t = Nx.tensor(1.0, type: :f32, backend: Nx.BinaryBackend)
    two_t = Nx.tensor(2.0, type: :f32, backend: Nx.BinaryBackend)
    pi_t = Nx.tensor(:math.pi(), type: :f32, backend: Nx.BinaryBackend)
    restart_period_t = Nx.tensor(restart_period, type: :f32, backend: Nx.BinaryBackend)
    restart_mult_t = Nx.tensor(restart_mult, type: :f32, backend: Nx.BinaryBackend)

    fn step ->
      step_f = Nx.as_type(step, :f32)

      # Warmup phase: linear ramp from 0 to base_lr
      warmup_progress = Nx.divide(step_f, Nx.max(warmup_steps_t, one_t))
      warmup_lr = Nx.multiply(warmup_progress, base_lr_t)

      # Post-warmup step (offset by warmup_steps)
      post_warmup_step = Nx.max(Nx.subtract(step_f, warmup_steps_t), zero_t)

      # Decay phase based on schedule type
      decay_lr =
        case schedule_type do
          :constant ->
            base_lr_t

          :cosine ->
            # Cosine decay: lr * (1 + cos(pi * step / decay_steps)) / 2
            progress = Nx.divide(post_warmup_step, decay_steps_t)
            clamped_progress = Nx.min(progress, one_t)

            cosine_factor =
              Nx.divide(
                Nx.add(one_t, Nx.cos(Nx.multiply(pi_t, clamped_progress))),
                two_t
              )

            Nx.multiply(base_lr_t, cosine_factor)

          :cosine_restarts ->
            # Cosine Annealing with Warm Restarts (SGDR)
            # Periods: T_0, T_0*T_mult, T_0*T_mult^2, ...
            # lr = 0.5 * lr_max * (1 + cos(pi * T_cur / T_i))
            #
            # To find current period and position within it:
            # - If T_mult == 1: period_idx = floor(step / T_0), T_cur = step % T_0
            # - If T_mult > 1: sum of geometric series = T_0 * (T_mult^n - 1) / (T_mult - 1)
            #   Solve for n: n = log(step * (T_mult - 1) / T_0 + 1) / log(T_mult)
            compute_cosine_restarts_lr(
              post_warmup_step,
              base_lr_t,
              restart_period_t,
              restart_mult_t,
              pi_t,
              one_t,
              two_t,
              zero_t
            )

          :exponential ->
            # Exponential decay: lr * rate^(step / transition_steps)
            rate_t = Nx.tensor(0.95, type: :f32, backend: Nx.BinaryBackend)
            exponent = Nx.divide(post_warmup_step, decay_steps_t)
            decay_factor = Nx.pow(rate_t, exponent)
            Nx.multiply(base_lr_t, decay_factor)

          :linear ->
            # Linear decay: lr * max(0, 1 - step / decay_steps)
            progress = Nx.divide(post_warmup_step, decay_steps_t)
            decay_factor = Nx.max(Nx.subtract(one_t, progress), zero_t)
            Nx.multiply(base_lr_t, decay_factor)
        end

      # Select warmup or decay based on current step
      in_warmup =
        Nx.greater(warmup_steps_t, zero_t) |> Nx.logical_and(Nx.less(step_f, warmup_steps_t))

      Nx.select(in_warmup, warmup_lr, decay_lr)
    end
  end

  # Compute learning rate for cosine annealing with warm restarts
  # Uses the SGDR algorithm where periods grow geometrically
  defp compute_cosine_restarts_lr(step, base_lr, t0, t_mult, pi, one, two, _zero) do
    # For T_mult == 1: simple periodic restarts
    # For T_mult > 1: geometric growth of periods
    #
    # When T_mult == 1:
    #   period_idx = floor(step / T_0)
    #   T_cur = step - period_idx * T_0
    #   T_i = T_0
    #
    # When T_mult > 1:
    #   Total steps through n full periods = T_0 * (T_mult^n - 1) / (T_mult - 1)
    #   Solve: step = T_0 * (T_mult^n - 1) / (T_mult - 1)
    #   n = log(step * (T_mult - 1) / T_0 + 1) / log(T_mult)

    # Handle T_mult == 1 vs T_mult > 1
    mult_is_one =
      Nx.less_equal(
        Nx.abs(Nx.subtract(t_mult, one)),
        Nx.tensor(1.0e-6, type: :f32, backend: Nx.BinaryBackend)
      )

    # Case 1: T_mult == 1 (fixed period)
    period_idx_fixed = Nx.floor(Nx.divide(step, t0))
    t_cur_fixed = Nx.subtract(step, Nx.multiply(period_idx_fixed, t0))
    t_i_fixed = t0

    # Case 2: T_mult > 1 (growing periods)
    # n = floor(log(step * (t_mult - 1) / t0 + 1) / log(t_mult))
    # Guard against division by zero when t_mult = 1 (even though we won't use this branch)
    eps = Nx.tensor(1.0e-6, type: :f32, backend: Nx.BinaryBackend)
    mult_minus_one = Nx.max(Nx.subtract(t_mult, one), eps)
    ratio = Nx.add(Nx.divide(Nx.multiply(step, mult_minus_one), t0), one)
    # Clamp ratio to >= 1 to avoid log(0)
    ratio_clamped = Nx.max(ratio, one)
    # Guard against log(1) = 0 in denominator
    log_t_mult = Nx.max(Nx.log(t_mult), eps)
    n_continuous = Nx.divide(Nx.log(ratio_clamped), log_t_mult)
    period_idx_grow = Nx.floor(n_continuous)

    # Total steps through period_idx complete periods
    # sum = T_0 * (T_mult^n - 1) / (T_mult - 1)
    completed_steps =
      Nx.divide(
        Nx.multiply(t0, Nx.subtract(Nx.pow(t_mult, period_idx_grow), one)),
        mult_minus_one
      )

    t_cur_grow = Nx.subtract(step, completed_steps)
    t_i_grow = Nx.multiply(t0, Nx.pow(t_mult, period_idx_grow))

    # Select based on t_mult
    t_cur = Nx.select(mult_is_one, t_cur_fixed, t_cur_grow)
    t_i = Nx.select(mult_is_one, t_i_fixed, t_i_grow)

    # Cosine annealing within current period
    # lr = 0.5 * lr_max * (1 + cos(pi * T_cur / T_i))
    progress = Nx.divide(t_cur, Nx.max(t_i, one))
    clamped_progress = Nx.min(progress, one)

    cosine_factor =
      Nx.divide(
        Nx.add(one, Nx.cos(Nx.multiply(pi, clamped_progress))),
        two
      )

    Nx.multiply(base_lr, cosine_factor)
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
      final_trainer =
        Enum.reduce(1..epochs, trainer, fn epoch, acc ->
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

  Supports gradient accumulation via config.accumulation_steps.
  With accumulation_steps=N, gradients are averaged over N mini-batches
  before applying updates, effectively training with batch_size*N.
  """
  @spec train_epoch(t(), Enumerable.t(), non_neg_integer(), function()) :: t()
  def train_epoch(trainer, dataset, epoch, callback) do
    accumulation_steps = trainer.config[:accumulation_steps] || 1

    if accumulation_steps == 1 do
      # Fast path: no accumulation, original behavior
      train_epoch_no_accumulation(trainer, dataset, epoch, callback)
    else
      # Gradient accumulation path
      train_epoch_with_accumulation(trainer, dataset, epoch, callback, accumulation_steps)
    end
  end

  defp train_epoch_no_accumulation(trainer, dataset, epoch, callback) do
    {_predict_fn, loss_fn} = build_loss_fn(trainer.policy_model)
    gc_every = trainer.config[:gc_every] || 0

    Enum.reduce(dataset, trainer, fn batch, acc ->
      {new_trainer, metrics} = train_step(acc, batch, loss_fn)

      # Call callback
      full_metrics = Map.merge(metrics, %{epoch: epoch, step: new_trainer.step})
      callback.(full_metrics)

      # Log periodically
      if rem(new_trainer.step, acc.config.log_interval) == 0 do
        Logger.info("Step #{new_trainer.step}: loss=#{Float.round(metrics.loss, 4)}")
      end

      # Periodic garbage collection to prevent memory buildup
      if gc_every > 0 and rem(new_trainer.step, gc_every) == 0 do
        :erlang.garbage_collect()
      end

      new_trainer
    end)
  end

  defp train_epoch_with_accumulation(trainer, dataset, epoch, callback, accumulation_steps) do
    gc_every = trainer.config[:gc_every] || 0

    # Track accumulated gradients and losses
    init_accum = %{
      trainer: trainer,
      grads: nil,
      losses: [],
      count: 0
    }

    final_accum =
      Enum.reduce(dataset, init_accum, fn batch, accum ->
        # Compute gradients without applying updates
        {grads, loss} = compute_gradients(accum.trainer, batch)

        # Accumulate gradients (sum them)
        new_grads =
          if accum.grads == nil do
            grads
          else
            add_gradients(accum.grads, grads)
          end

        new_accum = %{
          accum
          | grads: new_grads,
            losses: [loss | accum.losses],
            count: accum.count + 1
        }

        # Check if we should apply update
        if new_accum.count >= accumulation_steps do
          # Average gradients and apply update
          avg_grads = scale_gradients(new_accum.grads, 1.0 / accumulation_steps)
          avg_loss = Enum.sum(new_accum.losses) / accumulation_steps

          new_trainer = apply_gradients(new_accum.trainer, avg_grads)

          # Call callback
          metrics = %{loss: avg_loss, step: new_trainer.step}
          full_metrics = Map.merge(metrics, %{epoch: epoch, step: new_trainer.step})
          callback.(full_metrics)

          # Log periodically
          if rem(new_trainer.step, new_trainer.config.log_interval) == 0 do
            Logger.info(
              "Step #{new_trainer.step}: loss=#{Float.round(avg_loss, 4)} (accum=#{accumulation_steps})"
            )
          end

          # Periodic garbage collection to prevent memory buildup
          if gc_every > 0 and rem(new_trainer.step, gc_every) == 0 do
            :erlang.garbage_collect()
          end

          # Reset accumulation state
          %{trainer: new_trainer, grads: nil, losses: [], count: 0}
        else
          new_accum
        end
      end)

    # Handle remaining batches if dataset size isn't divisible by accumulation_steps
    if final_accum.count > 0 and final_accum.grads != nil do
      avg_grads = scale_gradients(final_accum.grads, 1.0 / final_accum.count)
      avg_loss = Enum.sum(final_accum.losses) / final_accum.count

      new_trainer = apply_gradients(final_accum.trainer, avg_grads)

      # Log final partial accumulation
      Logger.info(
        "Step #{new_trainer.step}: loss=#{Float.round(avg_loss, 4)} (partial accum=#{final_accum.count})"
      )

      new_trainer
    else
      final_accum.trainer
    end
  end

  # Compute gradients for a batch without applying updates
  defp compute_gradients(trainer, batch) do
    %{states: states, actions: actions} = batch

    states = Nx.backend_copy(states)
    actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)

    # Convert states to training precision (bf16 for ~2x speedup)
    states = Nx.as_type(states, trainer.config.precision)

    predict_fn = trainer.predict_fn
    model_state = deep_backend_copy(trainer.policy_params)
    label_smoothing = trainer.config[:label_smoothing] || 0.0
    focal_loss = trainer.config[:focal_loss] || false
    focal_gamma = trainer.config[:focal_gamma] || 2.0

    loss_fn = fn params ->
      {buttons, main_x, main_y, c_x, c_y, shoulder} =
        predict_fn.(Utils.ensure_model_state(params), states)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      Policy.imitation_loss(logits, actions,
        label_smoothing: label_smoothing,
        focal_loss: focal_loss,
        focal_gamma: focal_gamma
      )
    end

    {loss, grads} = Nx.Defn.value_and_grad(loss_fn).(model_state)
    grads_data = get_params_data(grads)
    loss_val = Nx.to_number(loss)

    {grads_data, loss_val}
  end

  # Add two gradient maps element-wise
  defp add_gradients(grads1, grads2) do
    deep_map2(grads1, grads2, fn t1, t2 -> Nx.add(t1, t2) end)
  end

  # Scale gradients by a factor
  defp scale_gradients(grads, factor) do
    factor_t = Nx.tensor(factor, type: :f32)
    deep_map(grads, fn t -> Nx.multiply(t, factor_t) end)
  end

  # Apply averaged gradients to update parameters
  defp apply_gradients(trainer, grads) do
    params_data = get_params_data(trainer.policy_params)

    {updates, new_optimizer_state} =
      trainer.optimizer.(
        grads,
        trainer.optimizer_state,
        params_data
      )

    new_params_data = trainer.apply_updates_fn.(params_data, updates)
    new_params = put_params_data(trainer.policy_params, new_params_data)

    %{
      trainer
      | policy_params: new_params,
        optimizer_state: new_optimizer_state,
        step: trainer.step + 1
    }
  end

  # Deep map over nested gradient structures
  defp deep_map(%Nx.Tensor{} = t, fun), do: fun.(t)

  defp deep_map(map, fun) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_map(v, fun)} end)
  end

  defp deep_map(other, _fun), do: other

  # Deep map2 over two nested gradient structures
  defp deep_map2(%Nx.Tensor{} = t1, %Nx.Tensor{} = t2, fun), do: fun.(t1, t2)

  defp deep_map2(map1, map2, fun) when is_map(map1) and is_map(map2) and not is_struct(map1) do
    Map.new(map1, fn {k, v1} ->
      v2 = Map.fetch!(map2, k)
      {k, deep_map2(v1, v2, fun)}
    end)
  end

  defp deep_map2(other, _other2, _fun), do: other

  @doc """
  Perform a single training step.

  Uses cached loss_and_grad_fn built in new/1 to avoid per-batch overhead.
  No more deep_backend_copy or closure creation every batch.

  When mixed precision is enabled:
  - Forward/backward pass uses BF16 compute params for speed
  - Gradients are cast to FP32 before accumulation (preserves small updates)
  - FP32 master weights maintain full precision across many steps
  """
  @spec train_step(t(), map(), function()) :: {t(), map()}
  def train_step(%{mixed_precision_state: nil} = trainer, batch, _loss_fn) do
    # Standard training path (no mixed precision)
    train_step_standard(trainer, batch)
  end

  def train_step(%{mixed_precision_state: mp_state} = trainer, batch, _loss_fn) do
    # Mixed precision training path
    train_step_mixed_precision(trainer, batch, mp_state)
  end

  # Standard training without mixed precision
  defp train_step_standard(trainer, batch) do
    %{states: states, actions: actions} = batch

    # Use cached loss+grad function (built once in new/1)
    {loss, grads} = trainer.loss_and_grad_fn.(trainer.policy_params, states, actions)

    # Extract data for optimizer (grads has same structure as ModelState)
    grads_data = get_params_data(grads)
    params_data = get_params_data(trainer.policy_params)

    # Update parameters using the optimizer
    {updates, new_optimizer_state} =
      trainer.optimizer.(
        grads_data,
        trainer.optimizer_state,
        params_data
      )

    # Use cached apply_updates_fn (built once in new/1, reused every step)
    new_params_data = trainer.apply_updates_fn.(params_data, updates)
    new_params = put_params_data(trainer.policy_params, new_params_data)

    new_trainer = %{
      trainer
      | policy_params: new_params,
        optimizer_state: new_optimizer_state,
        step: trainer.step + 1
    }

    {new_trainer, %{loss: loss, step: new_trainer.step}}
  end

  # Mixed precision training with FP32 master weights
  defp train_step_mixed_precision(trainer, batch, mp_state) do
    %{states: states, actions: actions} = batch

    # Get BF16 compute params for forward/backward pass (fast on tensor cores)
    compute_params = MixedPrecision.get_compute_params(mp_state)
    compute_model_state = put_params_data(trainer.policy_params, compute_params)

    # Forward + backward in BF16 (compute precision)
    {loss, grads} = trainer.loss_and_grad_fn.(compute_model_state, states, actions)
    grads_data = get_params_data(grads)

    # Cast gradients to FP32 (preserves small gradient updates)
    grads_f32 = MixedPrecision.cast_grads_to_f32(mp_state, grads_data)

    # Get FP32 master weights for optimizer
    master_params = MixedPrecision.get_master_params(mp_state)

    # Apply optimizer to FP32 master weights (same call pattern as non-mixed precision)
    {updates, new_optimizer_state} =
      trainer.optimizer.(
        grads_f32,
        trainer.optimizer_state,
        master_params
      )

    # Apply updates to FP32 master weights
    new_params_data = trainer.apply_updates_fn.(master_params, updates)

    # Update mixed precision state with new master params
    new_mp_state = MixedPrecision.set_master_params(mp_state, new_params_data)

    # Also update policy_params (for checkpointing)
    new_params = put_params_data(trainer.policy_params, new_params_data)

    new_trainer = %{
      trainer
      | policy_params: new_params,
        optimizer_state: new_optimizer_state,
        mixed_precision_state: new_mp_state,
        step: trainer.step + 1
    }

    {new_trainer, %{loss: loss, step: new_trainer.step}}
  end

  @doc """
  Build the loss function for training.

  ## Options
    - `:label_smoothing` - Label smoothing factor (default: 0.0)
  """
  @spec build_loss_fn(Axon.t(), keyword()) :: {function(), function()}
  def build_loss_fn(policy_model, opts \\ []) do
    label_smoothing = Keyword.get(opts, :label_smoothing, 0.0)
    focal_loss = Keyword.get(opts, :focal_loss, false)
    focal_gamma = Keyword.get(opts, :focal_gamma, 2.0)
    {_init_fn, predict_fn} = Axon.build(policy_model)

    loss_fn = fn params, states, actions ->
      # Forward pass
      {buttons, main_x, main_y, c_x, c_y, shoulder} =
        predict_fn.(Utils.ensure_model_state(params), states)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      # Compute loss with optional label smoothing and focal loss
      Policy.imitation_loss(logits, actions,
        label_smoothing: label_smoothing,
        focal_loss: focal_loss,
        focal_gamma: focal_gamma
      )
    end

    {predict_fn, loss_fn}
  end

  @doc """
  Build a compiled loss+gradient function for efficient training.

  This function is built ONCE in `new/1` and reused for all training steps.
  It avoids the need for `deep_backend_copy` every batch by:
  1. Taking all inputs (params, states, actions) as explicit arguments
  2. Using JIT compilation to cache the computation graph
  3. Not capturing any tensors in closures

  The returned function takes `(params, states, actions)` and returns `{loss, grads}`.
  """
  @spec build_loss_and_grad_fn(function(), map()) :: function()
  def build_loss_and_grad_fn(predict_fn, config) do
    # Extract config options that affect loss computation
    # These are captured once when building the function, not every batch
    label_smoothing = config[:label_smoothing] || 0.0
    focal_loss = config[:focal_loss] || false
    focal_gamma = config[:focal_gamma] || 2.0
    precision = config[:precision] || :bf16

    # Build the loss+grad function using JIT compilation
    # predict_fn is captured here (once), not in train_step (every batch)
    #
    # Strategy: We JIT compile a function that takes (params, states, actions) as
    # explicit arguments. By using Nx.Defn.jit on the outer function, all tensors
    # flow through as arguments and get properly traced together.
    #
    # The inner value_and_grad closure is fine because when the outer function is
    # JIT compiled, states/actions become Defn.Expr during tracing (not EXLA tensors).

    inner_fn = fn params, states, actions ->
      # Convert states to training precision
      states = Nx.as_type(states, precision)

      # Build loss function - states/actions are already Defn.Expr from outer JIT
      loss_fn = fn p ->
        {buttons, main_x, main_y, c_x, c_y, shoulder} =
          predict_fn.(Utils.ensure_model_state(p), states)

        logits = %{
          buttons: buttons,
          main_x: main_x,
          main_y: main_y,
          c_x: c_x,
          c_y: c_y,
          shoulder: shoulder
        }

        Policy.imitation_loss(logits, actions,
          label_smoothing: label_smoothing,
          focal_loss: focal_loss,
          focal_gamma: focal_gamma
        )
      end

      # Compute loss and gradients
      Nx.Defn.value_and_grad(loss_fn).(params)
    end

    # JIT compile the entire function - this makes states/actions flow as Defn.Expr
    # during tracing, avoiding the EXLA/Defn.Expr conflict
    Nx.Defn.jit(inner_fn, compiler: EXLA)
  end

  @doc """
  Evaluate on a validation dataset.

  ## Options
    - `:show_progress` - Show progress bar during evaluation (default: true)
    - `:progress_interval` - Update progress every N batches (default: 10)
  """
  @spec evaluate(t(), Enumerable.t(), keyword()) :: map()
  def evaluate(trainer, dataset, opts \\ []) do
    show_progress = Keyword.get(opts, :show_progress, true)
    progress_interval = Keyword.get(opts, :progress_interval, 10)

    # Use same label smoothing as training for consistent loss comparison
    label_smoothing = trainer.config[:label_smoothing] || 0.0
    {_predict_fn, loss_fn} = build_loss_fn(trainer.policy_model, label_smoothing: label_smoothing)

    # Try to get total count for progress bar (works for lists, not all enumerables)
    total_batches =
      case Enumerable.count(dataset) do
        {:ok, count} -> count
        {:error, _} -> nil
      end

    if show_progress and total_batches && total_batches > 0 do
      IO.write(:stderr, "    Validating: 0/#{total_batches} batches...\e[K")
    end

    # Accumulate losses as tensors, convert only once at the end
    # This avoids blocking GPU→CPU transfer after every batch
    {losses, count} =
      Enum.reduce(dataset, {[], 0}, fn batch, {acc_losses, acc_count} ->
        %{states: states, actions: actions} = batch
        loss = loss_fn.(trainer.policy_params, states, actions)
        new_count = acc_count + 1

        # Show progress
        if show_progress and total_batches && total_batches > 0 and rem(new_count, progress_interval) == 0 do
          pct = round(new_count / total_batches * 100)
          IO.write(:stderr, "\r    Validating: #{new_count}/#{total_batches} batches (#{pct}%)...\e[K")
        end

        {[loss | acc_losses], new_count}
      end)

    # Clear progress line
    if show_progress and total_batches && total_batches > 0 do
      IO.write(:stderr, "\r\e[K")
    end

    avg_loss =
      if count > 0 do
        # Single GPU→CPU transfer at the end instead of per-batch
        total = losses |> Nx.stack() |> Nx.sum() |> Nx.to_number()
        total / count
      else
        0.0
      end

    %{
      loss: avg_loss,
      num_batches: count
    }
  end

  @doc """
  Evaluate on a single batch. Returns loss as a tensor (not number) for efficiency.
  Caller should accumulate tensors and convert to number once at epoch end.
  """
  @spec evaluate_batch(t(), map()) :: %{loss: Nx.Tensor.t()}
  def evaluate_batch(trainer, batch) do
    label_smoothing = trainer.config[:label_smoothing] || 0.0
    {_predict_fn, loss_fn} = build_loss_fn(trainer.policy_model, label_smoothing: label_smoothing)

    %{states: states, actions: actions} = batch
    loss = loss_fn.(trainer.policy_params, states, actions)

    %{loss: loss}
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

  @doc """
  Save a training checkpoint asynchronously.

  Like `save_checkpoint/2` but returns immediately while the checkpoint
  is written in the background. This prevents training from blocking
  on disk I/O.

  Requires `ExPhil.Training.AsyncCheckpoint` to be started (typically
  in your application's supervision tree).

  ## Options
    - `:timeout` - Max time to wait if save queue is full (default: 5000ms)

  ## Example

      # Add to your application.ex supervision tree:
      children = [
        ExPhil.Training.AsyncCheckpoint,
        # ... other children
      ]

      # Then in training:
      :ok = Imitation.save_checkpoint_async(trainer, path)

      # At end of training, wait for pending saves:
      :ok = ExPhil.Training.AsyncCheckpoint.await_pending()
  """
  @spec save_checkpoint_async(t(), Path.t(), keyword()) :: :ok | {:error, :queue_full}
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

  @doc """
  Load a training checkpoint.

  Validates embed size if the trainer was initialized with one.
  Warns if checkpoint embed size differs from current config.
  Also validates that optimizer step count matches trainer.step.
  """
  @spec load_checkpoint(t(), Path.t()) :: {:ok, t()} | {:error, term()}
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
