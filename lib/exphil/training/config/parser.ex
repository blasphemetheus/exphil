defmodule ExPhil.Training.Config.Parser do
  @moduledoc """
  CLI argument parsing for training configuration.

  Provides utilities for:
  - Parsing command-line arguments into configuration options
  - Validating arguments against known flags
  - Suggesting corrections for typos
  - Type-safe parsing of various argument types

  ## Usage

      context = Parser.build_context(
        valid_backbones: [:lstm, :mamba],
        valid_optimizers: [:adam],
        ...
      )

      opts = Parser.parse(args, defaults, context)

  ## Context

  The context map provides allowlists for safe atom conversion:

      context = %{
        valid_backbones: [:lstm, :gru, :mamba],
        valid_optimizers: [:adam, :adamw],
        valid_lr_schedules: [:constant, :cosine],
        valid_characters: [:fox, :falco],
        valid_stages: [:battlefield],
        valid_flags: ["--epochs", "--batch-size", ...]
      }

  ## See Also

  - `ExPhil.Training.Config` - Main configuration module
  - `ExPhil.Training.Config.AtomSafety` - Safe atom conversion
  """

  alias ExPhil.Training.Config.AtomSafety

  @type parser_context :: %{
          valid_backbones: [atom()],
          valid_optimizers: [atom()],
          valid_lr_schedules: [atom()],
          valid_characters: [atom()],
          valid_stages: [atom()],
          valid_flags: [String.t()]
        }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Parse command-line arguments into configuration options.

  ## Parameters

  - `args` - List of CLI arguments
  - `base_opts` - Base options (defaults or from YAML)
  - `context` - Map with allowlists for parsing

  ## Returns

  Keyword list of parsed options.
  """
  @spec parse(list(String.t()), keyword(), parser_context()) :: keyword()
  def parse(args, base_opts, context) do
    parse_args_standard(args, base_opts, context)
  end

  @doc """
  Parse hidden sizes string into list of integers.

  ## Examples

      iex> Parser.parse_hidden_sizes("512,256,128")
      [512, 256, 128]

  """
  @spec parse_hidden_sizes(String.t()) :: [integer()]
  def parse_hidden_sizes(str) when is_binary(str) do
    str
    |> String.split(",")
    |> Enum.map(&String.trim/1)
    |> Enum.map(&String.to_integer/1)
  end

  @doc """
  Validate command-line arguments for unrecognized flags.

  Returns `{:ok, []}` if all flags are valid, or `{:ok, warnings}` with a list
  of warning messages for unrecognized flags with suggestions.

  ## Examples

      iex> Parser.validate_args(["--epochs", "10"], ["--epochs", "--batch-size"])
      {:ok, []}

  """
  @spec validate_args(list(String.t()), list(String.t())) :: {:ok, list(String.t())}
  def validate_args(args, valid_flags) when is_list(args) do
    # Extract all flags (args starting with --)
    input_flags =
      args
      |> Enum.filter(&String.starts_with?(&1, "--"))
      |> Enum.uniq()

    # Find unrecognized flags
    unrecognized = input_flags -- valid_flags

    warnings =
      Enum.map(unrecognized, fn flag ->
        case suggest_flag(flag, valid_flags) do
          nil -> "Unknown flag '#{flag}'. Run with --help to see available options."
          suggestion -> "Unknown flag '#{flag}'. Did you mean '#{suggestion}'?"
        end
      end)

    {:ok, warnings}
  end

  @doc """
  Validate args and print warnings if any.
  """
  @spec validate_args!(list(String.t()), list(String.t())) :: :ok
  def validate_args!(args, valid_flags) do
    {:ok, warnings} = validate_args(args, valid_flags)

    if warnings != [] do
      IO.puts(:stderr, "")

      Enum.each(warnings, fn warning ->
        IO.puts(:stderr, "⚠️  #{warning}")
      end)

      IO.puts(:stderr, "")
    end

    :ok
  end

  # ============================================================================
  # Core Parsing Logic
  # ============================================================================

  defp parse_args_standard(args, base_opts, ctx) do
    base_opts
    |> parse_string_arg(args, "--replays", :replays)
    |> parse_string_arg(args, "--replay-dir", :replays)
    |> parse_int_arg(args, "--epochs", :epochs)
    |> parse_int_arg(args, "--batch-size", :batch_size)
    |> parse_hidden_sizes_arg(args)
    |> parse_optional_int_arg(args, "--max-files", :max_files)
    |> parse_flag(args, "--skip-errors", :skip_errors)
    |> parse_flag(args, "--fail-fast", :fail_fast)
    |> parse_flag(args, "--show-errors", :show_errors)
    |> parse_flag(args, "--hide-errors", :hide_errors)
    |> parse_string_arg(args, "--error-log", :error_log)
    |> then(fn opts ->
      if opts[:fail_fast], do: Keyword.put(opts, :skip_errors, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:hide_errors], do: Keyword.put(opts, :show_errors, false), else: opts
    end)
    |> parse_string_arg(args, "--checkpoint", :checkpoint)
    |> parse_int_arg(args, "--player", :player_port)
    |> parse_atom_arg(args, "--train-character", :train_character, ctx[:valid_characters] || [])
    |> parse_flag(args, "--dual-port", :dual_port)
    |> parse_flag(args, "--balance-characters", :balance_characters)
    |> parse_flag(args, "--wandb", :wandb)
    |> parse_string_arg(args, "--wandb-project", :wandb_project)
    |> parse_string_arg(args, "--wandb-name", :wandb_name)
    |> parse_flag(args, "--temporal", :temporal)
    |> parse_atom_arg(args, "--backbone", :backbone, (ctx[:valid_backbones] || []) ++ [:mlp])
    # Policy type options
    |> parse_atom_arg(args, "--policy-type", :policy_type, ctx[:valid_policy_types] || [:autoregressive])
    |> parse_int_arg(args, "--action-horizon", :action_horizon)
    |> parse_int_arg(args, "--num-inference-steps", :num_inference_steps)
    |> parse_float_arg(args, "--kl-weight", :kl_weight)
    |> parse_int_arg(args, "--window-size", :window_size)
    |> parse_int_arg(args, "--stride", :stride)
    |> parse_int_arg(args, "--num-layers", :num_layers)
    |> parse_int_arg(args, "--attention-every", :attention_every)
    # Jamba stability options
    |> parse_flag(args, "--pre-norm", :pre_norm)
    |> parse_flag(args, "--no-pre-norm", :no_pre_norm)
    |> then(fn opts ->
      if opts[:no_pre_norm], do: Keyword.put(opts, :pre_norm, false), else: opts
    end)
    |> parse_flag(args, "--qk-layernorm", :qk_layernorm)
    |> parse_flag(args, "--no-qk-layernorm", :no_qk_layernorm)
    |> then(fn opts ->
      if opts[:no_qk_layernorm], do: Keyword.put(opts, :qk_layernorm, false), else: opts
    end)
    # Chunked attention
    |> parse_flag(args, "--chunked-attention", :chunked_attention)
    |> parse_flag(args, "--no-chunked-attention", :no_chunked_attention)
    |> then(fn opts ->
      if opts[:no_chunked_attention], do: Keyword.put(opts, :chunked_attention, false), else: opts
    end)
    |> parse_int_arg(args, "--chunk-size", :chunk_size)
    # Memory-efficient attention
    |> parse_flag(args, "--memory-efficient-attention", :memory_efficient_attention)
    |> parse_flag(args, "--no-memory-efficient-attention", :no_memory_efficient_attention)
    |> then(fn opts ->
      if opts[:no_memory_efficient_attention],
        do: Keyword.put(opts, :memory_efficient_attention, false),
        else: opts
    end)
    # FlashAttention NIF
    |> parse_flag(args, "--flash-attention-nif", :flash_attention_nif)
    |> parse_flag(args, "--no-flash-attention-nif", :no_flash_attention_nif)
    |> then(fn opts ->
      if opts[:no_flash_attention_nif],
        do: Keyword.put(opts, :flash_attention_nif, false),
        else: opts
    end)
    |> parse_int_arg(args, "--state-size", :state_size)
    |> parse_int_arg(args, "--expand-factor", :expand_factor)
    |> parse_int_arg(args, "--conv-size", :conv_size)
    |> parse_optional_int_arg(args, "--truncate-bptt", :truncate_bptt)
    |> parse_precision_arg(args)
    |> parse_flag(args, "--mixed-precision", :mixed_precision)
    |> parse_int_arg(args, "--frame-delay", :frame_delay)
    |> parse_flag(args, "--frame-delay-augment", :frame_delay_augment)
    |> parse_int_arg(args, "--frame-delay-min", :frame_delay_min)
    |> parse_int_arg(args, "--frame-delay-max", :frame_delay_max)
    |> parse_online_robust_flag(args)
    |> parse_flag(args, "--early-stopping", :early_stopping)
    |> parse_int_arg(args, "--patience", :patience)
    |> parse_float_arg(args, "--min-delta", :min_delta)
    |> parse_flag(args, "--save-best", :save_best)
    |> parse_optional_int_arg(args, "--save-every", :save_every)
    |> parse_optional_int_arg(args, "--save-every-batches", :save_every_batches)
    |> parse_float_arg(args, "--lr", :learning_rate)
    |> parse_float_arg(args, "--learning-rate", :learning_rate)
    |> parse_atom_arg(args, "--lr-schedule", :lr_schedule, ctx[:valid_lr_schedules] || [])
    |> parse_optional_int_arg(args, "--warmup-steps", :warmup_steps)
    |> parse_optional_int_arg(args, "--decay-steps", :decay_steps)
    |> parse_int_arg(args, "--restart-period", :restart_period)
    |> parse_float_arg(args, "--restart-mult", :restart_mult)
    |> parse_float_arg(args, "--max-grad-norm", :max_grad_norm)
    |> parse_string_arg(args, "--resume", :resume)
    |> parse_string_arg(args, "--name", :name)
    |> parse_int_arg(args, "--accumulation-steps", :accumulation_steps)
    |> parse_float_arg(args, "--val-split", :val_split)
    |> parse_flag(args, "--augment", :augment)
    |> parse_float_arg(args, "--mirror-prob", :mirror_prob)
    |> parse_float_arg(args, "--noise-prob", :noise_prob)
    |> parse_float_arg(args, "--noise-scale", :noise_scale)
    |> parse_float_arg(args, "--label-smoothing", :label_smoothing)
    |> parse_float_arg(args, "--dropout", :dropout)
    |> parse_flag(args, "--focal-loss", :focal_loss)
    |> parse_float_arg(args, "--focal-gamma", :focal_gamma)
    |> parse_float_arg(args, "--button-weight", :button_weight)
    |> parse_float_arg(args, "--stick-edge-weight", :stick_edge_weight)
    |> parse_flag(args, "--no-register", :no_register)
    |> parse_optional_int_arg(args, "--keep-best", :keep_best)
    |> parse_flag(args, "--ema", :ema)
    |> parse_float_arg(args, "--ema-decay", :ema_decay)
    |> parse_flag(args, "--precompute", :precompute)
    |> parse_flag(args, "--no-precompute", :no_precompute)
    |> parse_flag(args, "--cache-embeddings", :cache_embeddings)
    |> parse_flag(args, "--no-cache", :no_cache)
    |> parse_string_arg(args, "--cache-dir", :cache_dir)
    |> parse_flag(args, "--cache-augmented", :cache_augmented)
    |> parse_int_arg(args, "--num-noisy-variants", :num_noisy_variants)
    |> parse_flag(args, "--prefetch", :prefetch)
    |> parse_flag(args, "--no-prefetch", :no_prefetch)
    |> parse_flag(args, "--gradient-checkpoint", :gradient_checkpoint)
    |> parse_int_arg(args, "--checkpoint-every", :checkpoint_every)
    |> then(fn opts ->
      if opts[:no_prefetch], do: Keyword.put(opts, :prefetch, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_precompute], do: Keyword.put(opts, :precompute, false), else: opts
    end)
    |> parse_int_arg(args, "--prefetch-buffer", :prefetch_buffer)
    |> parse_flag(args, "--layer-norm", :layer_norm)
    |> parse_flag(args, "--no-layer-norm", :no_layer_norm)
    |> then(fn opts ->
      if opts[:no_layer_norm], do: Keyword.put(opts, :layer_norm, false), else: opts
    end)
    |> parse_flag(args, "--residual", :residual)
    |> parse_flag(args, "--no-residual", :no_residual)
    |> then(fn opts ->
      if opts[:no_residual], do: Keyword.put(opts, :residual, false), else: opts
    end)
    |> parse_atom_arg(args, "--optimizer", :optimizer, ctx[:valid_optimizers] || [])
    |> parse_flag(args, "--dry-run", :dry_run)
    |> parse_atom_list_arg(args, "--character", :characters, ctx[:valid_characters] || [])
    |> parse_atom_list_arg(args, "--characters", :characters, ctx[:valid_characters] || [])
    |> parse_atom_list_arg(args, "--stage", :stages, ctx[:valid_stages] || [])
    |> parse_atom_list_arg(args, "--stages", :stages, ctx[:valid_stages] || [])
    |> parse_string_arg(args, "--kmeans-centers", :kmeans_centers)
    |> parse_optional_int_arg(args, "--stream-chunk-size", :stream_chunk_size)
    |> parse_stage_mode_arg(args)
    |> parse_action_mode_arg(args)
    |> parse_character_mode_arg(args)
    |> parse_nana_mode_arg(args)
    |> parse_jumps_normalized_arg(args)
    |> parse_optional_int_arg(args, "--num-player-names", :num_player_names)
    # Player style learning
    |> parse_flag(args, "--learn-player-styles", :learn_player_styles)
    |> parse_flag(args, "--no-learn-player-styles", :no_learn_player_styles)
    |> parse_string_arg(args, "--player-registry", :player_registry)
    |> parse_optional_int_arg(args, "--min-player-games", :min_player_games)
    # Verbosity control
    |> parse_verbosity_flags(args)
    |> parse_optional_int_arg(args, "--log-interval", :log_interval)
    # Reproducibility
    |> parse_optional_int_arg(args, "--seed", :seed)
    # Checkpoint safety
    |> parse_flag(args, "--overwrite", :overwrite)
    |> parse_flag(args, "--no-overwrite", :no_overwrite)
    |> parse_flag(args, "--backup", :backup)
    |> parse_flag(args, "--no-backup", :no_backup)
    |> parse_optional_int_arg(args, "--backup-count", :backup_count)
    # Duplicate detection
    |> parse_flag(args, "--skip-duplicates", :skip_duplicates)
    |> parse_flag(args, "--no-skip-duplicates", :no_skip_duplicates)
    # Replay quality filtering
    |> parse_optional_int_arg(args, "--min-quality", :min_quality)
    |> parse_flag(args, "--show-quality-stats", :show_quality_stats)
    # Memory management
    |> parse_optional_int_arg(args, "--gc-every", :gc_every)
    # Profiling
    |> parse_flag(args, "--profile", :profile)
    # Parallel validation
    |> parse_optional_int_arg(args, "--val-concurrency", :val_concurrency)
    # Memory-mapped embeddings
    |> parse_flag_or_string(args, "--mmap-embeddings", :mmap_embeddings)
    |> parse_string_arg(args, "--mmap-path", :mmap_path)
    # Batch size auto-tuning
    |> parse_flag(args, "--auto-batch-size", :auto_batch_size)
    |> parse_optional_int_arg(args, "--auto-batch-min", :auto_batch_min)
    |> parse_optional_int_arg(args, "--auto-batch-max", :auto_batch_max)
    |> parse_float_arg(args, "--auto-batch-backoff", :auto_batch_backoff)
    |> then(fn opts ->
      if opts[:no_overwrite], do: Keyword.put(opts, :overwrite, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_backup], do: Keyword.put(opts, :backup, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_skip_duplicates], do: Keyword.put(opts, :skip_duplicates, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_learn_player_styles],
        do: Keyword.put(opts, :learn_player_styles, false),
        else: opts
    end)
  end

  # ============================================================================
  # Argument Parsing Helpers
  # ============================================================================

  @doc false
  def get_arg_value(args, flag) do
    case Enum.find_index(args, &(&1 == flag)) do
      nil -> nil
      idx -> Enum.at(args, idx + 1)
    end
  end

  @doc false
  def has_flag?(args, flag) do
    Enum.member?(args, flag)
  end

  @doc false
  def has_flag_value?(args, flag) do
    case Enum.find_index(args, &(&1 == flag)) do
      nil -> false
      idx -> Enum.at(args, idx + 1) != nil
    end
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
      nil ->
        opts

      value ->
        case Float.parse(value) do
          {float, ""} -> Keyword.put(opts, key, float)
          _ -> raise ArgumentError, "Invalid float for #{flag}: #{value}"
        end
    end
  end

  defp parse_atom_arg(opts, args, flag, key, allowed) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, AtomSafety.safe_to_atom!(value, allowed))
    end
  end

  defp parse_flag(opts, args, flag, key) do
    if has_flag?(args, flag) do
      Keyword.put(opts, key, true)
    else
      opts
    end
  end

  defp parse_flag_or_string(opts, args, flag, key) do
    if has_flag?(args, flag) do
      case get_arg_value(args, flag) do
        nil ->
          Keyword.put(opts, key, true)

        value when is_binary(value) ->
          if String.starts_with?(value, "--") do
            Keyword.put(opts, key, true)
          else
            Keyword.put(opts, key, value)
          end
      end
    else
      opts
    end
  end

  defp parse_atom_list_arg(opts, args, flag, key, allowed) do
    case get_arg_value(args, flag) do
      nil ->
        opts

      value ->
        atoms =
          value
          |> String.split(",")
          |> Enum.map(&String.trim/1)
          |> Enum.map(&AtomSafety.safe_to_atom!(&1, allowed))

        Keyword.put(opts, key, atoms)
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

  defp parse_online_robust_flag(opts, args) do
    if has_flag?(args, "--online-robust") do
      Keyword.put(opts, :frame_delay_augment, true)
    else
      opts
    end
  end

  defp parse_verbosity_flags(opts, args) do
    cond do
      "--quiet" in args -> Keyword.put(opts, :verbosity, 0)
      "--verbose" in args -> Keyword.put(opts, :verbosity, 2)
      true -> opts
    end
  end

  # ============================================================================
  # Mode Parsing Helpers
  # ============================================================================

  defp parse_stage_mode_arg(opts, args) do
    cond do
      has_flag?(args, "--stage-mode-full") ->
        Keyword.put(opts, :stage_mode, :one_hot_full)

      has_flag?(args, "--stage-mode-compact") ->
        Keyword.put(opts, :stage_mode, :one_hot_compact)

      has_flag?(args, "--stage-mode-learned") ->
        Keyword.put(opts, :stage_mode, :learned)

      has_flag_value?(args, "--stage-mode") ->
        mode_str = get_arg_value(args, "--stage-mode")

        mode =
          case mode_str do
            "full" -> :one_hot_full
            "one_hot_full" -> :one_hot_full
            "compact" -> :one_hot_compact
            "one_hot_compact" -> :one_hot_compact
            "learned" -> :learned
            other -> raise "Unknown stage mode: #{other}"
          end

        Keyword.put(opts, :stage_mode, mode)

      true ->
        opts
    end
  end

  defp parse_action_mode_arg(opts, args) do
    cond do
      has_flag?(args, "--action-mode-one-hot") ->
        Keyword.put(opts, :action_mode, :one_hot)

      has_flag?(args, "--action-mode-learned") ->
        Keyword.put(opts, :action_mode, :learned)

      has_flag_value?(args, "--action-mode") ->
        mode_str = get_arg_value(args, "--action-mode")

        mode =
          case mode_str do
            "one_hot" -> :one_hot
            "learned" -> :learned
            other -> raise "Unknown action mode: #{other}"
          end

        Keyword.put(opts, :action_mode, mode)

      true ->
        opts
    end
  end

  defp parse_character_mode_arg(opts, args) do
    cond do
      has_flag?(args, "--character-mode-one-hot") ->
        Keyword.put(opts, :character_mode, :one_hot)

      has_flag?(args, "--character-mode-learned") ->
        Keyword.put(opts, :character_mode, :learned)

      has_flag_value?(args, "--character-mode") ->
        mode_str = get_arg_value(args, "--character-mode")

        mode =
          case mode_str do
            "one_hot" -> :one_hot
            "learned" -> :learned
            other -> raise "Unknown character mode: #{other}"
          end

        Keyword.put(opts, :character_mode, mode)

      true ->
        opts
    end
  end

  defp parse_nana_mode_arg(opts, args) do
    cond do
      has_flag_value?(args, "--nana-mode") ->
        mode_str = get_arg_value(args, "--nana-mode")

        mode =
          case mode_str do
            "compact" -> :compact
            "enhanced" -> :enhanced
            "full" -> :full
            other -> raise "Unknown nana mode: #{other}"
          end

        Keyword.put(opts, :nana_mode, mode)

      true ->
        opts
    end
  end

  defp parse_jumps_normalized_arg(opts, args) do
    cond do
      has_flag?(args, "--jumps-normalized") ->
        Keyword.put(opts, :jumps_normalized, true)

      has_flag?(args, "--no-jumps-normalized") ->
        Keyword.put(opts, :jumps_normalized, false)

      true ->
        opts
    end
  end

  # ============================================================================
  # Flag Suggestion (Levenshtein Distance)
  # ============================================================================

  defp suggest_flag(typo, valid_flags) do
    valid_flags
    |> Enum.map(fn flag -> {flag, levenshtein_distance(typo, flag)} end)
    |> Enum.min_by(fn {_flag, distance} -> distance end)
    |> case do
      {flag, distance} when distance <= 3 -> flag
      _ -> nil
    end
  end

  defp levenshtein_distance(s1, s2) do
    s1_chars = String.graphemes(s1)
    s2_chars = String.graphemes(s2)
    s2_len = length(s2_chars)

    initial_row = Enum.to_list(0..s2_len)

    {final_row, _} =
      Enum.reduce(Enum.with_index(s1_chars), {initial_row, 0}, fn {c1, i}, {prev_row, _} ->
        first = i + 1

        {new_row_reversed, _} =
          Enum.reduce(Enum.with_index(s2_chars), {[first], first}, fn {c2, j},
                                                                      {row_acc, diagonal} ->
            above = Enum.at(prev_row, j + 1)
            left = hd(row_acc)

            cost = if c1 == c2, do: 0, else: 1
            min_val = min(min(above + 1, left + 1), diagonal + cost)

            {[min_val | row_acc], above}
          end)

        {Enum.reverse(new_row_reversed), i + 1}
      end)

    List.last(final_row)
  end
end
