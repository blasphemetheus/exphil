defmodule ExPhil.Training.Config do
  @moduledoc """
  Training configuration parsing and generation.

  Extracts the configuration logic from training scripts into a testable module.
  Handles:
  - Command-line argument parsing
  - Timestamped checkpoint name generation
  - Training config JSON structure
  """

  @default_replays_dir "/home/dori/git/melee/replays"
  @default_hidden_sizes [64, 64]

  @valid_presets [:quick, :standard, :full, :full_cpu, :mewtwo, :ganondorf, :link, :gameandwatch, :zelda]

  @doc """
  List of available preset names.
  """
  def available_presets, do: @valid_presets

  @doc """
  Default training options.
  """
  def defaults do
    [
      replays: @default_replays_dir,
      epochs: 10,
      batch_size: 64,
      hidden_sizes: @default_hidden_sizes,
      max_files: nil,
      checkpoint: nil,
      player_port: 1,
      wandb: false,
      wandb_project: "exphil",
      wandb_name: nil,
      temporal: false,
      backbone: :sliding_window,
      window_size: 60,
      stride: 1,
      num_layers: 2,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      truncate_bptt: nil,
      precision: :bf16,
      frame_delay: 0,
      preset: nil,
      character: nil,
      # Early stopping
      early_stopping: false,
      patience: 5,
      min_delta: 0.01,
      # Checkpointing
      save_best: true,
      save_every: nil,
      # Learning rate
      learning_rate: 1.0e-4,
      lr_schedule: :constant,
      warmup_steps: 0,
      decay_steps: nil,
      # Resumption
      resume: nil,
      # Model naming
      name: nil,
      # Gradient accumulation
      accumulation_steps: 1,
      # Validation split
      val_split: 0.0
    ]
  end

  # ============================================================================
  # Training Presets
  # ============================================================================

  @doc """
  Get training options for a preset.

  ## Available Presets

  ### Speed Presets
  - `:quick` - Fast iteration (1 epoch, 5 files, small MLP)
  - `:standard` - Balanced training (10 epochs, 50 files, medium MLP)
  - `:full` - Maximum quality (50 epochs, all files, Mamba temporal)
  - `:full_cpu` - Full training optimized for CPU (no temporal)

  ### Character Presets
  - `:mewtwo` - Optimized for Mewtwo (longer sequences for teleport recovery)
  - `:ganondorf` - Optimized for Ganondorf (spacing-focused)
  - `:link` - Optimized for Link (projectile tracking)
  - `:gameandwatch` - Optimized for Mr. Game & Watch (shorter sequences)
  - `:zelda` - Optimized for Zelda (transform mechanics)

  ## Examples

      iex> Config.preset(:quick)
      [epochs: 1, max_files: 5, hidden_sizes: [32, 32], ...]

      iex> Config.preset(:mewtwo)
      [character: :mewtwo, epochs: 50, hidden_sizes: [256, 256], window_size: 90, ...]

  """
  def preset(:quick) do
    [
      epochs: 1,
      max_files: 5,
      batch_size: 64,
      hidden_sizes: [32, 32],
      temporal: false,
      preset: :quick
    ]
  end

  def preset(:standard) do
    [
      epochs: 10,
      max_files: 50,
      batch_size: 64,
      hidden_sizes: [64, 64],
      temporal: false,
      preset: :standard
    ]
  end

  def preset(:full) do
    [
      epochs: 50,
      max_files: nil,
      batch_size: 128,
      hidden_sizes: [256, 256],
      temporal: true,
      backbone: :mamba,
      window_size: 60,
      num_layers: 2,
      preset: :full
    ]
  end

  def preset(:full_cpu) do
    [
      epochs: 20,
      max_files: 100,
      batch_size: 64,
      hidden_sizes: [128, 128],
      temporal: false,
      preset: :full_cpu
    ]
  end

  # Character-specific presets (built on :full)
  def preset(:mewtwo) do
    Keyword.merge(preset(:full), [
      character: :mewtwo,
      window_size: 90,  # Longer sequences for teleport recovery tracking
      preset: :mewtwo
    ])
  end

  def preset(:ganondorf) do
    Keyword.merge(preset(:full), [
      character: :ganondorf,
      window_size: 60,  # Standard - spacing-focused
      preset: :ganondorf
    ])
  end

  def preset(:link) do
    Keyword.merge(preset(:full), [
      character: :link,
      window_size: 75,  # Longer for projectile tracking
      preset: :link
    ])
  end

  def preset(:gameandwatch) do
    Keyword.merge(preset(:full), [
      character: :gameandwatch,
      window_size: 45,  # Shorter - no L-cancel simplifies timing
      preset: :gameandwatch
    ])
  end

  def preset(:zelda) do
    Keyword.merge(preset(:full), [
      character: :zelda,
      window_size: 60,  # Standard - transform tracking handled separately
      preset: :zelda
    ])
  end

  def preset(name) when is_binary(name) do
    preset(String.to_atom(name))
  end

  def preset(invalid) do
    raise ArgumentError, """
    Unknown preset: #{inspect(invalid)}

    Available presets:
      Speed:     quick, standard, full, full_cpu
      Character: mewtwo, ganondorf, link, gameandwatch, zelda

    Usage: mix run scripts/train_from_replays.exs --preset quick
    """
  end

  # ============================================================================
  # Validation
  # ============================================================================

  @valid_backbones [:lstm, :gru, :mamba, :sliding_window, :hybrid]

  @doc """
  Validate training options and return errors/warnings.

  Returns `{:ok, opts}` if valid, or `{:error, errors}` if invalid.
  Warnings are logged but don't cause validation to fail.

  ## Examples

      iex> Config.validate(epochs: 10, batch_size: 64)
      {:ok, [epochs: 10, batch_size: 64]}

      iex> Config.validate(epochs: -1)
      {:error, ["epochs must be positive, got: -1"]}

  """
  def validate(opts) do
    errors = collect_errors(opts)
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
  Warnings are logged but don't cause validation to fail.

  ## Examples

      iex> Config.validate!(epochs: 10, batch_size: 64)
      [epochs: 10, batch_size: 64]

      iex> Config.validate!(epochs: -1)
      ** (ArgumentError) Invalid training configuration...

  """
  def validate!(opts) do
    case validate(opts) do
      {:ok, opts} -> opts
      {:error, errors} ->
        raise ArgumentError, """
        Invalid training configuration:

        #{Enum.map_join(errors, "\n", &("  - " <> &1))}

        Use --help or see docs/TRAINING_FEATURES.md for valid options.
        """
    end
  end

  defp collect_errors(opts) do
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
    |> validate_hidden_sizes(opts)
    |> validate_temporal_backbone(opts)
    |> validate_precision(opts)
    |> validate_replays_dir(opts)
    |> validate_positive(opts, :patience)
    |> validate_positive_float(opts, :min_delta)
    |> validate_positive_float(opts, :learning_rate)
    |> validate_lr_schedule(opts)
    |> validate_non_negative(opts, :warmup_steps)
    |> validate_resume_checkpoint(opts)
    |> validate_positive(opts, :accumulation_steps)
    |> validate_val_split(opts)
  end

  defp collect_warnings(opts) do
    []
    |> warn_large_window_size(opts)
    |> warn_large_batch_size(opts)
    |> warn_many_epochs_without_wandb(opts)
    |> warn_temporal_without_window(opts)
  end

  # Error validators
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

  defp validate_hidden_sizes(errors, opts) do
    case opts[:hidden_sizes] do
      nil -> errors
      sizes when is_list(sizes) ->
        if Enum.all?(sizes, &(is_integer(&1) and &1 > 0)) do
          errors
        else
          ["hidden_sizes must be a list of positive integers, got: #{inspect(sizes)}" | errors]
        end
      other ->
        ["hidden_sizes must be a list, got: #{inspect(other)}" | errors]
    end
  end

  defp validate_temporal_backbone(errors, opts) do
    if opts[:temporal] do
      backbone = opts[:backbone]
      if backbone not in @valid_backbones do
        ["temporal training requires backbone in #{inspect(@valid_backbones)}, got: #{inspect(backbone)}" | errors]
      else
        errors
      end
    else
      errors
    end
  end

  defp validate_precision(errors, opts) do
    case opts[:precision] do
      p when p in [:bf16, :f32] -> errors
      nil -> errors
      other -> ["precision must be :bf16 or :f32, got: #{inspect(other)}" | errors]
    end
  end

  defp validate_replays_dir(errors, opts) do
    dir = opts[:replays]
    if dir && not File.dir?(dir) do
      ["replays directory does not exist: #{dir}" | errors]
    else
      errors
    end
  end

  @valid_lr_schedules [:constant, :cosine, :exponential, :linear]

  defp validate_lr_schedule(errors, opts) do
    schedule = opts[:lr_schedule]
    if schedule && schedule not in @valid_lr_schedules do
      ["lr_schedule must be one of #{inspect(@valid_lr_schedules)}, got: #{inspect(schedule)}" | errors]
    else
      errors
    end
  end

  defp validate_resume_checkpoint(errors, opts) do
    resume_path = opts[:resume]
    if resume_path && not File.exists?(resume_path) do
      ["resume checkpoint does not exist: #{resume_path}" | errors]
    else
      errors
    end
  end

  defp validate_val_split(errors, opts) do
    val_split = opts[:val_split]
    cond do
      is_nil(val_split) -> errors
      not is_number(val_split) -> ["val_split must be a number, got: #{inspect(val_split)}" | errors]
      val_split < 0.0 or val_split >= 1.0 -> ["val_split must be in [0.0, 1.0), got: #{val_split}" | errors]
      true -> errors
    end
  end

  # Warning collectors
  defp warn_large_window_size(warnings, opts) do
    if opts[:window_size] && opts[:window_size] > 120 do
      ["window_size #{opts[:window_size]} > 120 may cause memory issues" | warnings]
    else
      warnings
    end
  end

  defp warn_large_batch_size(warnings, opts) do
    if opts[:batch_size] && opts[:batch_size] > 256 do
      ["batch_size #{opts[:batch_size]} > 256 may cause memory issues on CPU" | warnings]
    else
      warnings
    end
  end

  defp warn_many_epochs_without_wandb(warnings, opts) do
    epochs = opts[:epochs] || 0
    wandb = opts[:wandb] || false
    if epochs >= 20 and not wandb do
      ["training #{epochs} epochs without --wandb; consider enabling for metrics tracking" | warnings]
    else
      warnings
    end
  end

  defp warn_temporal_without_window(warnings, opts) do
    temporal = opts[:temporal] || false
    window_size = opts[:window_size] || 60
    if temporal and window_size < 30 do
      ["temporal training with window_size < 30 may miss important temporal patterns" | warnings]
    else
      warnings
    end
  end

  @doc """
  Apply a preset to options, allowing CLI args to override preset values.

  Preset values serve as defaults, but any explicitly provided CLI arguments
  take precedence.

  ## Examples

      # Preset provides epochs: 1, but CLI overrides with epochs: 5
      iex> opts = Config.parse_args(["--preset", "quick", "--epochs", "5"])
      iex> opts[:epochs]
      5
      iex> opts[:hidden_sizes]
      [32, 32]  # From preset

  """
  def apply_preset(opts, args) do
    case get_arg_value(args, "--preset") do
      nil ->
        opts

      preset_name ->
        preset_opts = preset(preset_name)

        # Merge: defaults < preset < CLI args
        # We need to identify which opts were explicitly set via CLI
        cli_overrides = get_cli_overrides(args)

        defaults()
        |> Keyword.merge(preset_opts)
        |> Keyword.merge(cli_overrides)
    end
  end

  # Get only the options that were explicitly provided via CLI
  defp get_cli_overrides(args) do
    []
    |> maybe_add_override(args, "--epochs", :epochs, &String.to_integer/1)
    |> maybe_add_override(args, "--batch-size", :batch_size, &String.to_integer/1)
    |> maybe_add_override(args, "--max-files", :max_files, &String.to_integer/1)
    |> maybe_add_override(args, "--hidden-sizes", :hidden_sizes, &parse_hidden_sizes/1)
    |> maybe_add_override(args, "--window-size", :window_size, &String.to_integer/1)
    |> maybe_add_override(args, "--backbone", :backbone, &String.to_atom/1)
    |> maybe_add_override(args, "--num-layers", :num_layers, &String.to_integer/1)
    |> maybe_add_override(args, "--frame-delay", :frame_delay, &String.to_integer/1)
    |> maybe_add_override(args, "--replays", :replays, & &1)
    |> maybe_add_override(args, "--checkpoint", :checkpoint, & &1)
    |> maybe_add_flag_override(args, "--temporal", :temporal)
    |> maybe_add_flag_override(args, "--wandb", :wandb)
  end

  defp maybe_add_override(opts, args, flag, key, parser) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, parser.(value))
    end
  end

  defp maybe_add_flag_override(opts, args, flag, key) do
    if has_flag?(args, flag) do
      Keyword.put(opts, key, true)
    else
      opts
    end
  end

  @doc """
  Parse command-line arguments into a keyword list of options.

  If `--preset` is provided, the preset values are used as a base,
  with any explicit CLI arguments overriding the preset.

  ## Examples

      iex> Config.parse_args(["--epochs", "5", "--temporal"])
      [epochs: 5, temporal: true, ...]

      iex> Config.parse_args(["--preset", "quick"])
      [epochs: 1, max_files: 5, hidden_sizes: [32, 32], ...]

      iex> Config.parse_args(["--preset", "quick", "--epochs", "3"])
      [epochs: 3, max_files: 5, hidden_sizes: [32, 32], ...]  # epochs overridden

  """
  def parse_args(args) when is_list(args) do
    # Check if preset is specified - if so, use apply_preset flow
    if has_flag_value?(args, "--preset") do
      apply_preset(defaults(), args)
    else
      # No preset - standard parsing flow
      parse_args_standard(args)
    end
  end

  defp parse_args_standard(args) do
    defaults()
    |> parse_string_arg(args, "--replays", :replays)
    |> parse_int_arg(args, "--epochs", :epochs)
    |> parse_int_arg(args, "--batch-size", :batch_size)
    |> parse_hidden_sizes_arg(args)
    |> parse_optional_int_arg(args, "--max-files", :max_files)
    |> parse_string_arg(args, "--checkpoint", :checkpoint)
    |> parse_int_arg(args, "--player", :player_port)
    |> parse_flag(args, "--wandb", :wandb)
    |> parse_string_arg(args, "--wandb-project", :wandb_project)
    |> parse_string_arg(args, "--wandb-name", :wandb_name)
    |> parse_flag(args, "--temporal", :temporal)
    |> parse_atom_arg(args, "--backbone", :backbone)
    |> parse_int_arg(args, "--window-size", :window_size)
    |> parse_int_arg(args, "--stride", :stride)
    |> parse_int_arg(args, "--num-layers", :num_layers)
    |> parse_int_arg(args, "--state-size", :state_size)
    |> parse_int_arg(args, "--expand-factor", :expand_factor)
    |> parse_int_arg(args, "--conv-size", :conv_size)
    |> parse_optional_int_arg(args, "--truncate-bptt", :truncate_bptt)
    |> parse_precision_arg(args)
    |> parse_int_arg(args, "--frame-delay", :frame_delay)
    |> parse_flag(args, "--early-stopping", :early_stopping)
    |> parse_int_arg(args, "--patience", :patience)
    |> parse_float_arg(args, "--min-delta", :min_delta)
    |> parse_flag(args, "--save-best", :save_best)
    |> parse_optional_int_arg(args, "--save-every", :save_every)
    |> parse_float_arg(args, "--lr", :learning_rate)
    |> parse_atom_arg(args, "--lr-schedule", :lr_schedule)
    |> parse_optional_int_arg(args, "--warmup-steps", :warmup_steps)
    |> parse_optional_int_arg(args, "--decay-steps", :decay_steps)
    |> parse_string_arg(args, "--resume", :resume)
    |> parse_string_arg(args, "--name", :name)
    |> parse_int_arg(args, "--accumulation-steps", :accumulation_steps)
    |> parse_float_arg(args, "--val-split", :val_split)
  end

  defp has_flag_value?(args, flag) do
    case Enum.find_index(args, &(&1 == flag)) do
      nil -> false
      idx -> Enum.at(args, idx + 1) != nil
    end
  end

  @doc """
  Generate a checkpoint name with memorable naming if not already specified.

  Format: `checkpoints/{character_}{backbone}_{name}_{timestamp}.axon`

  The name can be:
  - Explicitly set with `--name wavedashing_falcon`
  - Auto-generated if not specified (e.g., "tactical_marth")

  ## Examples

      iex> opts = [checkpoint: nil, temporal: false, character: nil, name: nil]
      iex> Config.ensure_checkpoint_name(opts) |> Keyword.get(:checkpoint)
      "checkpoints/mlp_cosmic_falcon_20260119_123456.axon"

      iex> opts = [checkpoint: nil, temporal: true, backbone: :mamba, character: :mewtwo, name: nil]
      iex> Config.ensure_checkpoint_name(opts) |> Keyword.get(:checkpoint)
      "checkpoints/mewtwo_mamba_brave_phoenix_20260119_123456.axon"

      iex> opts = [checkpoint: nil, temporal: false, name: "my_custom_name"]
      iex> Config.ensure_checkpoint_name(opts) |> Keyword.get(:checkpoint)
      "checkpoints/mlp_my_custom_name_20260119_123456.axon"

      iex> opts = [checkpoint: "my_model.axon"]
      iex> Config.ensure_checkpoint_name(opts) |> Keyword.get(:checkpoint)
      "my_model.axon"

  """
  def ensure_checkpoint_name(opts) do
    if opts[:checkpoint] do
      opts
    else
      alias ExPhil.Training.Naming

      timestamp = generate_timestamp()
      backbone = if opts[:temporal], do: opts[:backbone], else: :mlp
      name = opts[:name] || Naming.generate()
      character = opts[:character]

      checkpoint_name = if character do
        "checkpoints/#{character}_#{backbone}_#{name}_#{timestamp}.axon"
      else
        "checkpoints/#{backbone}_#{name}_#{timestamp}.axon"
      end

      # Store the generated name in opts for display
      opts
      |> Keyword.put(:checkpoint, checkpoint_name)
      |> Keyword.put(:name, name)
    end
  end

  @doc """
  Generate a timestamp string for checkpoint naming.

  Format: YYYYMMDD_HHMMSS in UTC
  """
  def generate_timestamp do
    DateTime.utc_now() |> Calendar.strftime("%Y%m%d_%H%M%S")
  end

  @doc """
  Generate a timestamp string using a specific DateTime (for testing).
  """
  def generate_timestamp(%DateTime{} = dt) do
    Calendar.strftime(dt, "%Y%m%d_%H%M%S")
  end

  @doc """
  Build the training config map that gets saved as JSON alongside the model.

  ## Parameters
  - opts: The training options keyword list
  - results: A map with training results like :embed_size, :training_frames, etc.
  """
  def build_config_json(opts, results \\ %{}) do
    %{
      # Timestamp
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),

      # Input parameters
      replays_dir: opts[:replays],
      max_files: opts[:max_files],
      player_port: opts[:player_port],

      # Model architecture
      temporal: opts[:temporal],
      backbone: if(opts[:temporal], do: to_string(opts[:backbone]), else: "mlp"),
      hidden_sizes: opts[:hidden_sizes],
      embed_size: results[:embed_size],

      # Temporal options
      window_size: opts[:window_size],
      stride: opts[:stride],
      num_layers: opts[:num_layers],
      truncate_bptt: opts[:truncate_bptt],

      # Mamba options
      state_size: opts[:state_size],
      expand_factor: opts[:expand_factor],
      conv_size: opts[:conv_size],

      # Training parameters
      epochs: opts[:epochs],
      batch_size: opts[:batch_size],
      precision: to_string(opts[:precision]),
      frame_delay: opts[:frame_delay],

      # Early stopping
      early_stopping: opts[:early_stopping],
      patience: opts[:patience],
      min_delta: opts[:min_delta],

      # Results (if provided)
      training_frames: results[:training_frames],
      validation_frames: results[:validation_frames],
      total_time_seconds: results[:total_time_seconds],
      final_training_loss: results[:final_training_loss],
      epochs_completed: results[:epochs_completed],
      stopped_early: results[:stopped_early],
      checkpoint_path: opts[:checkpoint],
      policy_path: derive_policy_path(opts[:checkpoint])
    }
  end

  @doc """
  Derive the policy path from a checkpoint path.

  ## Examples

      iex> Config.derive_policy_path("checkpoints/mlp_20260119.axon")
      "checkpoints/mlp_20260119_policy.bin"

  """
  def derive_policy_path(nil), do: nil
  def derive_policy_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_policy.bin")
  end

  @doc """
  Derive the config JSON path from a checkpoint path.

  ## Examples

      iex> Config.derive_config_path("checkpoints/mlp_20260119.axon")
      "checkpoints/mlp_20260119_config.json"

  """
  def derive_config_path(nil), do: nil
  def derive_config_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_config.json")
  end

  @doc """
  Derive the best checkpoint path from a checkpoint path.

  ## Examples

      iex> Config.derive_best_checkpoint_path("checkpoints/mlp_20260119.axon")
      "checkpoints/mlp_20260119_best.axon"

  """
  def derive_best_checkpoint_path(nil), do: nil
  def derive_best_checkpoint_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_best.axon")
  end

  @doc """
  Derive the best policy path from a checkpoint path.

  ## Examples

      iex> Config.derive_best_policy_path("checkpoints/mlp_20260119.axon")
      "checkpoints/mlp_20260119_best_policy.bin"

  """
  def derive_best_policy_path(nil), do: nil
  def derive_best_policy_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_best_policy.bin")
  end

  @doc """
  Parse hidden sizes from a comma-separated string.

  ## Examples

      iex> Config.parse_hidden_sizes("64,64")
      [64, 64]

      iex> Config.parse_hidden_sizes("128, 64, 32")
      [128, 64, 32]

  """
  def parse_hidden_sizes(str) when is_binary(str) do
    str
    |> String.split(",")
    |> Enum.map(&String.trim/1)
    |> Enum.map(&String.to_integer/1)
  end

  # ============================================================================
  # Private helpers
  # ============================================================================

  defp get_arg_value(args, flag) do
    case Enum.find_index(args, &(&1 == flag)) do
      nil -> nil
      idx -> Enum.at(args, idx + 1)
    end
  end

  defp has_flag?(args, flag) do
    Enum.member?(args, flag)
  end

  defp parse_string_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, value)
    end
  end

  defp parse_int_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, String.to_integer(value))
    end
  end

  defp parse_optional_int_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, String.to_integer(value))
    end
  end

  defp parse_float_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, String.to_float(value))
    end
  end

  defp parse_atom_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, String.to_atom(value))
    end
  end

  defp parse_flag(opts, args, flag, key) do
    if has_flag?(args, flag) do
      Keyword.put(opts, key, true)
    else
      opts
    end
  end

  defp parse_hidden_sizes_arg(opts, args) do
    case get_arg_value(args, "--hidden-sizes") do
      nil -> opts
      value -> Keyword.put(opts, :hidden_sizes, parse_hidden_sizes(value))
    end
  end

  defp parse_precision_arg(opts, args) do
    case get_arg_value(args, "--precision") do
      nil -> opts
      "f32" -> Keyword.put(opts, :precision, :f32)
      "bf16" -> Keyword.put(opts, :precision, :bf16)
      other -> raise "Unknown precision: #{other}. Use 'bf16' or 'f32'"
    end
  end
end
