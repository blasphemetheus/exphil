defmodule ExPhil.Training.Config.Validator do
  @moduledoc """
  Validation functions for training configuration.

  Validates training options and collects errors/warnings. Designed to be
  called by `ExPhil.Training.Config.validate/1` with the necessary context
  (allowlists for backbones, optimizers, etc.).

  ## Validation Categories

  - **Numeric bounds**: epochs, batch_size, window_size, etc.
  - **Type checking**: hidden_sizes must be list of positive integers
  - **Range validation**: val_split in [0, 1), probabilities in [0, 1]
  - **Consistency**: frame_delay_min <= frame_delay_max
  - **File existence**: replays dir, resume checkpoint

  ## Warnings vs Errors

  - **Errors** cause validation to fail with `{:error, errors}`
  - **Warnings** are logged but don't fail validation

  ## See Also

  - `ExPhil.Training.Config` - Main configuration module
  - `ExPhil.Training.Help` - Help text and error message formatting
  """

  alias ExPhil.Constants
  alias ExPhil.Training.Help

  @type validation_context :: %{
          valid_backbones: [atom()],
          valid_optimizers: [atom()],
          valid_lr_schedules: [atom()]
        }

  @doc """
  Validate training options and return errors/warnings.

  Returns `{:ok, opts}` if valid, or `{:error, errors}` if invalid.
  Warnings are logged but don't cause validation to fail.

  ## Parameters

  - `opts` - Keyword list of training options
  - `context` - Map with allowlists for validation:
    - `:valid_backbones` - List of valid backbone atoms
    - `:valid_optimizers` - List of valid optimizer atoms
    - `:valid_lr_schedules` - List of valid LR schedule atoms

  ## Examples

      iex> context = %{valid_backbones: [:mamba, :lstm], valid_optimizers: [:adam], valid_lr_schedules: [:cosine]}
      iex> Validator.validate([epochs: 10, batch_size: 64], context)
      {:ok, [epochs: 10, batch_size: 64]}

  """
  @spec validate(keyword(), validation_context()) :: {:ok, keyword()} | {:error, [String.t()]}
  def validate(opts, context) do
    errors = collect_errors(opts, context)
    warnings = collect_warnings(opts)

    # Log warnings
    Enum.each(warnings, &IO.warn/1)

    case errors do
      [] -> {:ok, opts}
      _ -> {:error, errors}
    end
  end

  @doc """
  Validate training options, raising on errors.

  Returns opts if valid, raises `ArgumentError` if invalid.
  """
  @spec validate!(keyword(), validation_context()) :: keyword()
  def validate!(opts, context) do
    case validate(opts, context) do
      {:ok, opts} ->
        opts

      {:error, errors} ->
        raise ArgumentError, """
        Invalid training configuration:

        #{Enum.map_join(errors, "\n", &("  - " <> &1))}

        Use --help or see docs/TRAINING_FEATURES.md for valid options.
        """
    end
  end

  # ============================================================================
  # Error Collection
  # ============================================================================

  defp collect_errors(opts, context) do
    []
    |> validate_positive(opts, :epochs)
    |> validate_positive(opts, :batch_size)
    |> validate_positive_or_nil(opts, :max_files)
    |> validate_positive(opts, :window_size)
    |> validate_positive(opts, :stride)
    |> validate_positive(opts, :num_layers)
    |> validate_positive(opts, :state_size)
    |> validate_positive(opts, :expand_factor)
    |> validate_positive(opts, :conv_size)
    |> validate_non_negative(opts, :frame_delay)
    |> validate_non_negative(opts, :frame_delay_min)
    |> validate_non_negative(opts, :frame_delay_max)
    |> validate_frame_delay_range(opts)
    |> validate_hidden_sizes(opts)
    |> validate_temporal_backbone(opts, context)
    |> validate_precision(opts)
    |> validate_optimizer(opts, context)
    |> validate_replays_dir(opts)
    |> validate_positive(opts, :patience)
    |> validate_positive_float(opts, :min_delta)
    |> validate_positive_float(opts, :learning_rate)
    |> validate_lr_schedule(opts, context)
    |> validate_non_negative(opts, :warmup_steps)
    |> validate_positive(opts, :restart_period)
    |> validate_restart_mult(opts)
    |> validate_non_negative_float(opts, :max_grad_norm)
    |> validate_resume_checkpoint(opts)
    |> validate_positive(opts, :accumulation_steps)
    |> validate_val_split(opts)
    |> validate_probability(opts, :mirror_prob)
    |> validate_probability(opts, :noise_prob)
    |> validate_positive_float(opts, :noise_scale)
    |> validate_label_smoothing(opts)
    |> validate_positive_or_nil(opts, :keep_best)
    |> validate_ema_decay(opts)
    |> validate_positive_or_nil(opts, :stream_chunk_size)
  end

  # ============================================================================
  # Warning Collection
  # ============================================================================

  defp collect_warnings(opts) do
    []
    |> warn_large_window_size(opts)
    |> warn_large_batch_size(opts)
    |> warn_temporal_without_window(opts)
  end

  # ============================================================================
  # Numeric Validators
  # ============================================================================

  defp validate_positive(errors, opts, key) do
    value = opts[key]

    if is_integer(value) and value <= 0 do
      ["#{key} must be positive, got: #{value}" | errors]
    else
      errors
    end
  end

  defp validate_positive_or_nil(errors, opts, key) do
    value = opts[key]

    if value != nil and (not is_integer(value) or value <= 0) do
      ["#{key} must be a positive integer or nil, got: #{inspect(value)}" | errors]
    else
      errors
    end
  end

  defp validate_non_negative(errors, opts, key) do
    value = opts[key]

    if is_integer(value) and value < 0 do
      ["#{key} must be non-negative, got: #{value}" | errors]
    else
      errors
    end
  end

  defp validate_positive_float(errors, opts, key) do
    value = opts[key]

    cond do
      is_nil(value) -> errors
      is_number(value) and value > 0 -> errors
      is_number(value) -> ["#{key} must be positive, got: #{value}" | errors]
      true -> ["#{key} must be a positive number, got: #{inspect(value)}" | errors]
    end
  end

  defp validate_non_negative_float(errors, opts, key) do
    value = opts[key]

    cond do
      is_nil(value) -> errors
      is_number(value) and value >= 0 -> errors
      is_number(value) -> ["#{key} must be non-negative, got: #{value}" | errors]
      true -> ["#{key} must be a non-negative number, got: #{inspect(value)}" | errors]
    end
  end

  # ============================================================================
  # Complex Validators
  # ============================================================================

  defp validate_hidden_sizes(errors, opts) do
    case opts[:hidden_sizes] do
      nil ->
        errors

      sizes when is_list(sizes) ->
        if Enum.all?(sizes, &(is_integer(&1) and &1 > 0)) do
          errors
        else
          msg = "hidden_sizes must be a list of positive integers, got: #{inspect(sizes)}"
          [Help.with_link(msg, :hidden_sizes) | errors]
        end

      other ->
        msg = "hidden_sizes must be a list, got: #{inspect(other)}"
        [Help.with_link(msg, :hidden_sizes) | errors]
    end
  end

  defp validate_frame_delay_range(errors, opts) do
    min_delay = opts[:frame_delay_min] || 0
    max_delay = opts[:frame_delay_max] || Constants.online_frame_delay()

    cond do
      min_delay > max_delay ->
        [
          "frame_delay_min (#{min_delay}) cannot be greater than frame_delay_max (#{max_delay})"
          | errors
        ]

      max_delay > 60 ->
        [
          "frame_delay_max > 60 is unusually high (online play is typically #{Constants.online_frame_delay()} frames)"
          | errors
        ]

      true ->
        errors
    end
  end

  defp validate_temporal_backbone(errors, opts, context) do
    if opts[:temporal] do
      backbone = opts[:backbone]
      valid_backbones = context[:valid_backbones] || []

      if backbone not in valid_backbones do
        msg =
          "temporal training requires backbone in #{inspect(valid_backbones)}, got: #{inspect(backbone)}"

        [Help.with_link(msg, :backbone) | errors]
      else
        errors
      end
    else
      errors
    end
  end

  defp validate_precision(errors, opts) do
    case opts[:precision] do
      p when p in [:bf16, :f32] ->
        errors

      nil ->
        errors

      other ->
        msg = "precision must be :bf16 or :f32, got: #{inspect(other)}"
        [Help.with_link(msg, :precision) | errors]
    end
  end

  defp validate_optimizer(errors, opts, context) do
    valid_optimizers = context[:valid_optimizers] || []
    optimizer = opts[:optimizer]

    cond do
      is_nil(optimizer) ->
        errors

      optimizer in valid_optimizers ->
        errors

      true ->
        [
          "optimizer must be one of #{inspect(valid_optimizers)}, got: #{inspect(optimizer)}"
          | errors
        ]
    end
  end

  defp validate_replays_dir(errors, opts) do
    dir = opts[:replays]

    if dir && not File.dir?(dir) do
      msg = "replays directory does not exist: #{dir}"
      [Help.with_link(msg, :replays) | errors]
    else
      errors
    end
  end

  defp validate_lr_schedule(errors, opts, context) do
    schedule = opts[:lr_schedule]
    valid_schedules = context[:valid_lr_schedules] || []

    if schedule && schedule not in valid_schedules do
      msg =
        "lr_schedule must be one of #{inspect(valid_schedules)}, got: #{inspect(schedule)}"

      [Help.with_link(msg, :lr_schedule) | errors]
    else
      errors
    end
  end

  defp validate_resume_checkpoint(errors, opts) do
    resume_path = opts[:resume]

    if resume_path && not File.exists?(resume_path) do
      msg = "resume checkpoint does not exist: #{resume_path}"
      [Help.with_link(msg, :resume) | errors]
    else
      errors
    end
  end

  # ============================================================================
  # Range Validators
  # ============================================================================

  defp validate_val_split(errors, opts) do
    val_split = opts[:val_split]

    cond do
      is_nil(val_split) ->
        errors

      not is_number(val_split) ->
        ["val_split must be a number, got: #{inspect(val_split)}" | errors]

      val_split < 0.0 or val_split >= 1.0 ->
        ["val_split must be in [0.0, 1.0), got: #{val_split}" | errors]

      true ->
        errors
    end
  end

  defp validate_probability(errors, opts, key) do
    value = opts[key]

    cond do
      is_nil(value) -> errors
      not is_number(value) -> ["#{key} must be a number, got: #{inspect(value)}" | errors]
      value < 0.0 or value > 1.0 -> ["#{key} must be in [0.0, 1.0], got: #{value}" | errors]
      true -> errors
    end
  end

  defp validate_label_smoothing(errors, opts) do
    value = opts[:label_smoothing]

    cond do
      is_nil(value) ->
        errors

      not is_number(value) ->
        msg = "label_smoothing must be a number, got: #{inspect(value)}"
        [Help.with_link(msg, :label_smoothing) | errors]

      value < 0.0 or value >= 1.0 ->
        msg = "label_smoothing must be in [0.0, 1.0), got: #{value}"
        [Help.with_link(msg, :label_smoothing) | errors]

      true ->
        errors
    end
  end

  defp validate_ema_decay(errors, opts) do
    value = opts[:ema_decay]

    cond do
      is_nil(value) -> errors
      not is_number(value) -> ["ema_decay must be a number, got: #{inspect(value)}" | errors]
      value <= 0.0 or value >= 1.0 -> ["ema_decay must be in (0.0, 1.0), got: #{value}" | errors]
      true -> errors
    end
  end

  defp validate_restart_mult(errors, opts) do
    value = opts[:restart_mult]

    cond do
      is_nil(value) -> errors
      not is_number(value) -> ["restart_mult must be a number, got: #{inspect(value)}" | errors]
      value < 1.0 -> ["restart_mult must be >= 1.0, got: #{value}" | errors]
      true -> errors
    end
  end

  # ============================================================================
  # Warning Collectors
  # ============================================================================

  defp warn_large_window_size(warnings, opts) do
    if opts[:window_size] && opts[:window_size] > 120 do
      msg = "window_size #{opts[:window_size]} > 120 may cause memory issues"
      [Help.warning_with_help(msg, :window_size) | warnings]
    else
      warnings
    end
  end

  defp warn_large_batch_size(warnings, opts) do
    # Only warn about large batch sizes on CPU - GPUs handle large batches fine
    if opts[:batch_size] && opts[:batch_size] > 256 && not gpu_available?() do
      msg = "batch_size #{opts[:batch_size]} > 256 may cause memory issues on CPU"
      [Help.warning_with_help(msg, :batch_size) | warnings]
    else
      warnings
    end
  end

  defp gpu_available? do
    # Check if EXLA with CUDA is available
    case System.get_env("EXLA_TARGET") do
      "cuda" ->
        true

      _ ->
        # Also check if EXLA detected a GPU
        try do
          ExPhil.Training.GPUUtils.gpu_available?()
        rescue
          _ -> false
        end
    end
  end

  defp warn_temporal_without_window(warnings, opts) do
    temporal = opts[:temporal] || false
    window_size = opts[:window_size] || 60

    if temporal and window_size < 30 do
      msg = "temporal training with window_size < 30 may miss important temporal patterns"
      [Help.warning_with_help(msg, :temporal) | warnings]
    else
      warnings
    end
  end
end
