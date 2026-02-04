defmodule ExPhil.Training.Imitation.Optimizer do
  @moduledoc """
  Optimizer creation and learning rate schedules for imitation learning.

  Provides functions to create optimizers with:
  - Multiple optimizer types (Adam, AdamW, LAMB, RAdam, SGD, RMSProp)
  - Learning rate schedules (constant, cosine, exponential, linear)
  - Cosine annealing with warm restarts (SGDR)
  - Warmup periods
  - Gradient clipping

  ## Supported Optimizers

  | Type | Description | Best For |
  |------|-------------|----------|
  | `:adamw` | AdamW (default) | General training with weight decay |
  | `:adam` | Standard Adam | Simple training |
  | `:lamb` | LAMB | Large batch training |
  | `:radam` | Rectified Adam | Stable early training |
  | `:sgd` | SGD + momentum | Fine-tuning |
  | `:rmsprop` | RMSprop | Alternative to Adam |

  ## Learning Rate Schedules

  | Schedule | Description |
  |----------|-------------|
  | `:constant` | Fixed learning rate |
  | `:cosine` | Cosine decay to 0 |
  | `:cosine_restarts` | SGDR with growing periods |
  | `:exponential` | Decay by 0.95 per epoch |
  | `:linear` | Linear decay to 0 |

  ## See Also

  - `ExPhil.Training.Imitation` - Main imitation learning module
  - `Polaris.Optimizers` - Underlying optimizer implementations
  """

  @doc """
  Create optimizer with learning rate schedule and gradient clipping.

  ## Config Options
    - `:learning_rate` - Base learning rate
    - `:optimizer` - Optimizer type (default: :adamw)
    - `:max_grad_norm` - Gradient clipping threshold (default: 1.0)
    - `:weight_decay` - L2 regularization (default: 1.0e-5)
    - `:lr_schedule` - Schedule type (default: :constant)
    - `:warmup_steps` - LR warmup steps (default: 0)
    - `:decay_steps` - Steps over which to decay
    - `:restart_period` - SGDR initial period (default: 1000)
    - `:restart_mult` - SGDR period multiplier (default: 2)

  ## Returns
    `{init_fn, update_fn}` tuple compatible with Polaris.
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

  @doc """
  Build a base optimizer with the given learning rate schedule.
  """
  @spec build_base_optimizer(atom(), function() | Nx.Tensor.t(), map()) ::
          {function(), function()}
  def build_base_optimizer(:adam, lr_schedule, _config) do
    Polaris.Optimizers.adam(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-8
    )
  end

  def build_base_optimizer(:adamw, lr_schedule, config) do
    Polaris.Optimizers.adamw(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-8,
      decay: config[:weight_decay] || 1.0e-5
    )
  end

  def build_base_optimizer(:lamb, lr_schedule, config) do
    Polaris.Optimizers.lamb(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-6,
      decay: config[:weight_decay] || 1.0e-5
    )
  end

  def build_base_optimizer(:radam, lr_schedule, _config) do
    Polaris.Optimizers.radam(
      learning_rate: lr_schedule,
      b1: 0.9,
      b2: 0.999,
      eps: 1.0e-8
    )
  end

  def build_base_optimizer(:sgd, lr_schedule, _config) do
    Polaris.Optimizers.sgd(
      learning_rate: lr_schedule,
      momentum: 0.9
    )
  end

  def build_base_optimizer(:rmsprop, lr_schedule, _config) do
    Polaris.Optimizers.rmsprop(
      learning_rate: lr_schedule,
      centered: false,
      momentum: 0.0
    )
  end

  @doc """
  Build a learning rate schedule function based on config.

  Returns a function that takes step count and returns learning rate tensor.
  """
  @spec build_lr_schedule(map()) :: function()
  def build_lr_schedule(config) do
    base_lr = config[:learning_rate] || 1.0e-4
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

  # ============================================================================
  # Private Helpers - Polaris Schedules
  # ============================================================================

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

  # ============================================================================
  # Private Helpers - Custom Schedules (warmup + restarts)
  # ============================================================================

  # Build a custom schedule function that implements warmup + decay
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
    #   Solve for n: n = log(step * (T_mult - 1) / T_0 + 1) / log(T_mult)

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
    # Guard against division by zero when t_mult = 1
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
  Extract the optimizer's internal step count.

  The optimizer state tracks steps internally for LR scheduling.
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
end
