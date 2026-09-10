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
    |> apply_flag_table(args, ctx)
    |> parse_hidden_sizes_arg(args)
    |> then(fn opts ->
      if opts[:fail_fast], do: Keyword.put(opts, :skip_errors, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:hide_errors], do: Keyword.put(opts, :show_errors, false), else: opts
    end)
    # Policy type options
    # Jamba stability options
    |> then(fn opts ->
      if opts[:no_pre_norm], do: Keyword.put(opts, :pre_norm, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_qk_layernorm], do: Keyword.put(opts, :qk_layernorm, false), else: opts
    end)
    # Chunked attention
    |> then(fn opts ->
      if opts[:no_chunked_attention], do: Keyword.put(opts, :chunked_attention, false), else: opts
    end)
    # Memory-efficient attention
    |> then(fn opts ->
      if opts[:no_memory_efficient_attention],
        do: Keyword.put(opts, :memory_efficient_attention, false),
        else: opts
    end)
    # FlashAttention NIF
    |> then(fn opts ->
      if opts[:no_flash_attention_nif],
        do: Keyword.put(opts, :flash_attention_nif, false),
        else: opts
    end)
    |> parse_precision_arg(args)
    |> parse_online_robust_flag(args)
    |> parse_button_pos_weight(args)
    |> then(fn opts ->
      if opts[:no_prefetch], do: Keyword.put(opts, :prefetch, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_precompute], do: Keyword.put(opts, :precompute, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_layer_norm], do: Keyword.put(opts, :layer_norm, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_residual], do: Keyword.put(opts, :residual, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_pipeline_chunks], do: Keyword.put(opts, :pipeline_chunks, false), else: opts
    end)
    |> then(fn opts ->
      if opts[:no_cache_streaming], do: Keyword.put(opts, :cache_streaming, false), else: opts
    end)
    |> parse_stage_mode_arg(args)
    |> parse_action_mode_arg(args)
    |> parse_character_mode_arg(args)
    |> parse_nana_mode_arg(args)
    |> parse_jumps_normalized_arg(args)
    # Player style learning
    # Verbosity control
    |> parse_verbosity_flags(args)
    # Reproducibility
    # Checkpoint safety
    # Duplicate detection
    # Replay quality filtering
    # Memory management
    # Profiling
    # Parallel validation
    # Memory-mapped embeddings
    # Batch size auto-tuning
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


  # ---------------------------------------------------------------------------
  # INVARIANTS.md item 2 (phase B, 2026-09-09): THE flag table. Every simple
  # flag is a row here; `apply_flag_table/3` is the only code that parses
  # them, and `Config.@valid_flags` is DERIVED from `flags/0` — so a flag
  # that is accepted but not parsed (the --num-heads bug) or parsed but
  # rejected (the mode aliases) cannot be written. Multi-key steps
  # (hidden sizes, precision, modes, button pos-weight, verbosity, preset,
  # config) stay explicit below and are listed in @special_flags.
  # Types: :int :float :string :optional_int :flag :neg_flag
  #        :flag_or_string {:atom, allowed | {:ctx, key, default}}
  #        {:atom_list, {:ctx, key, default}}
  # ---------------------------------------------------------------------------
  @flag_table [
    {"--replays", :replays, :string},
    {"--replay-dir", :replays, :string},
    {"--corpus", :corpus, :string},
    {"--epochs", :epochs, :int},
    {"--batch-size", :batch_size, :int},
    {"--max-files", :max_files, :optional_int},
    {"--skip-errors", :skip_errors, :flag},
    {"--fail-fast", :fail_fast, :flag},
    {"--show-errors", :show_errors, :flag},
    {"--hide-errors", :hide_errors, :flag},
    {"--error-log", :error_log, :string},
    {"--checkpoint", :checkpoint, :string},
    {"--player", :player_port, :int},
    {"--train-character", :train_character, {:atom, {:ctx, :valid_characters, []}}},
    {"--select-character-port", :select_character_port, :flag},
    {"--dual-port", :dual_port, :flag},
    {"--balance-characters", :balance_characters, :flag},
    {"--wandb", :wandb, :flag},
    {"--wandb-project", :wandb_project, :string},
    {"--wandb-name", :wandb_name, :string},
    {"--temporal", :temporal, :flag},
    {"--backbone", :backbone, {:atom, {:ctx_plus, :valid_backbones, [:mlp]}}},
    {"--policy-type", :policy_type, {:atom, {:ctx, :valid_policy_types, [:autoregressive]}}},
    {"--head", :head, {:atom, {:ctx, :valid_heads, [:independent, :autoregressive]}}},
    {"--action-horizon", :action_horizon, :int},
    {"--num-inference-steps", :num_inference_steps, :int},
    {"--kl-weight", :kl_weight, :float},
    {"--window-size", :window_size, :int},
    {"--stride", :stride, :int},
    {"--num-layers", :num_layers, :int},
    {"--attention-every", :attention_every, :int},
    {"--pre-norm", :pre_norm, :flag},
    {"--no-pre-norm", :no_pre_norm, :flag},
    {"--qk-layernorm", :qk_layernorm, :flag},
    {"--no-qk-layernorm", :no_qk_layernorm, :flag},
    {"--chunked-attention", :chunked_attention, :flag},
    {"--no-chunked-attention", :no_chunked_attention, :flag},
    {"--chunk-size", :chunk_size, :int},
    {"--memory-efficient-attention", :memory_efficient_attention, :flag},
    {"--no-memory-efficient-attention", :no_memory_efficient_attention, :flag},
    {"--flash-attention-nif", :flash_attention_nif, :flag},
    {"--no-flash-attention-nif", :no_flash_attention_nif, :flag},
    {"--state-size", :state_size, :int},
    {"--expand-factor", :expand_factor, :int},
    {"--conv-size", :conv_size, :int},
    {"--truncate-bptt", :truncate_bptt, :optional_int},
    {"--bptt", :bptt, :flag},
    {"--unroll", :unroll, :int},
    {"--bptt-overlap", :bptt_overlap, :int},
    {"--bptt-val-files", :bptt_val_files, :int},
    {"--mixed-precision", :mixed_precision, :flag},
    {"--frame-delay", :frame_delay, :int},
    {"--num-heads", :num_heads, :int},
    {"--head-dim", :head_dim, :int},
    {"--log-file", :log_file, :string},
    {"--frame-delay-augment", :frame_delay_augment, :flag},
    {"--frame-delay-min", :frame_delay_min, :int},
    {"--frame-delay-max", :frame_delay_max, :int},
    {"--stage-internals", :stage_internals, :flag},
    {"--early-stopping", :early_stopping, :flag},
    {"--patience", :patience, :int},
    {"--min-delta", :min_delta, :float},
    {"--save-best", :save_best, :flag},
    {"--save-every", :save_every, :optional_int},
    {"--save-every-batches", :save_every_batches, :optional_int},
    {"--lr", :learning_rate, :float},
    {"--learning-rate", :learning_rate, :float},
    {"--lr-schedule", :lr_schedule, {:atom, {:ctx, :valid_lr_schedules, []}}},
    {"--warmup-steps", :warmup_steps, :optional_int},
    {"--decay-steps", :decay_steps, :optional_int},
    {"--restart-period", :restart_period, :int},
    {"--restart-mult", :restart_mult, :float},
    {"--max-grad-norm", :max_grad_norm, :float},
    {"--resume", :resume, :string},
    {"--reinit-head", :reinit_head, :flag},
    {"--name", :name, :string},
    {"--accumulation-steps", :accumulation_steps, :int},
    {"--val-split", :val_split, :float},
    {"--augment", :augment, :flag},
    {"--mirror-prob", :mirror_prob, :float},
    {"--noise-prob", :noise_prob, :float},
    {"--noise-scale", :noise_scale, :float},
    {"--label-smoothing", :label_smoothing, :float},
    {"--dropout", :dropout, :float},
    {"--focal-loss", :focal_loss, :flag},
    {"--prev-action", :use_prev_action, :flag},
    {"--no-prev-action", :use_prev_action, :neg_flag},
    {"--prev-action-dropout", :prev_action_dropout, :float},
    {"--scheduled-sampling", :scheduled_sampling, :float},
    {"--ss-ramp", :ss_ramp, :int},
    {"--mix-frames", :mix_frames, :string},
    {"--mix-corpus", :mix_corpus, :string},
    {"--mix-oversample", :mix_oversample, :int},
    {"--per-stage-ledge", :per_stage_ledge, :flag},
    {"--action-delay", :action_delay, :int},
    {"--no-focal-loss", :focal_loss, :neg_flag},
    {"--focal-gamma", :focal_gamma, :float},
    {"--button-weight", :button_weight, :float},
    {"--stick-edge-weight", :stick_edge_weight, :float},
    {"--entropy-weight", :entropy_weight, :float},
    {"--neutral-weight", :neutral_weight, :float},
    {"--transition-weight", :transition_weight, :float},
    {"--offstage-weight", :offstage_weight, :float},
    {"--awbc", :awbc, :flag},
    {"--awbc-reward", :awbc_reward, {:atom, [:shine, :standard]}},
    {"--awbc-beta", :awbc_beta, :float},
    {"--awbc-shuffle", :awbc_shuffle, :flag},
    {"--head-normalize", :head_normalize, :flag},
    {"--no-head-normalize", :head_normalize, :neg_flag},
    {"--action-oversample", :action_oversample, :float},
    {"--lazy-sequences", :lazy_sequences, :flag},
    {"--use-batch", :use_batch, :flag},
    {"--no-register", :no_register, :flag},
    {"--keep-best", :keep_best, :optional_int},
    {"--ema", :ema, :flag},
    {"--ema-decay", :ema_decay, :float},
    {"--precompute", :precompute, :flag},
    {"--no-precompute", :no_precompute, :flag},
    {"--cache-embeddings", :cache_embeddings, :flag},
    {"--no-cache", :no_cache, :flag},
    {"--cache-dir", :cache_dir, :string},
    {"--cache-augmented", :cache_augmented, :flag},
    {"--num-noisy-variants", :num_noisy_variants, :int},
    {"--prefetch", :prefetch, :flag},
    {"--no-prefetch", :no_prefetch, :flag},
    {"--gradient-checkpoint", :gradient_checkpoint, :flag},
    {"--checkpoint-every", :checkpoint_every, :int},
    {"--prefetch-buffer", :prefetch_buffer, :int},
    {"--layer-norm", :layer_norm, :flag},
    {"--no-layer-norm", :no_layer_norm, :flag},
    {"--residual", :residual, :flag},
    {"--no-residual", :no_residual, :flag},
    {"--optimizer", :optimizer, {:atom, {:ctx, :valid_optimizers, []}}},
    {"--dry-run", :dry_run, :flag},
    {"--character", :characters, {:atom_list, {:ctx, :valid_characters, []}}},
    {"--characters", :characters, {:atom_list, {:ctx, :valid_characters, []}}},
    {"--stage", :stages, {:atom_list, {:ctx, :valid_stages, []}}},
    {"--stages", :stages, {:atom_list, {:ctx, :valid_stages, []}}},
    {"--kmeans-centers", :kmeans_centers, :string},
    {"--stream-chunk-size", :stream_chunk_size, :optional_int},
    {"--pipeline-chunks", :pipeline_chunks, :flag},
    {"--no-pipeline-chunks", :no_pipeline_chunks, :flag},
    {"--cache-streaming", :cache_streaming, :flag},
    {"--no-cache-streaming", :no_cache_streaming, :flag},
    {"--num-player-names", :num_player_names, :optional_int},
    {"--learn-player-styles", :learn_player_styles, :flag},
    {"--no-learn-player-styles", :no_learn_player_styles, :flag},
    {"--player-registry", :player_registry, :string},
    {"--min-player-games", :min_player_games, :optional_int},
    {"--log-interval", :log_interval, :optional_int},
    {"--seed", :seed, :optional_int},
    {"--overwrite", :overwrite, :flag},
    {"--no-overwrite", :no_overwrite, :flag},
    {"--backup", :backup, :flag},
    {"--no-backup", :no_backup, :flag},
    {"--backup-count", :backup_count, :optional_int},
    {"--skip-duplicates", :skip_duplicates, :flag},
    {"--no-skip-duplicates", :no_skip_duplicates, :flag},
    {"--min-quality", :min_quality, :optional_int},
    {"--show-quality-stats", :show_quality_stats, :flag},
    {"--gc-every", :gc_every, :optional_int},
    {"--profile", :profile, :flag},
    {"--val-concurrency", :val_concurrency, :optional_int},
    {"--mmap-embeddings", :mmap_embeddings, :flag_or_string},
    {"--mmap-path", :mmap_path, :string},
    {"--auto-batch-size", :auto_batch_size, :flag},
    {"--auto-batch-min", :auto_batch_min, :optional_int},
    {"--auto-batch-max", :auto_batch_max, :optional_int},
    {"--auto-batch-backoff", :auto_batch_backoff, :float},
  ]

  @special_flags ~w(
    --preset --config --verbose --quiet
    --hidden-sizes --precision --online-robust --button-pos-weight
    --stage-mode --stage-mode-full --stage-mode-compact --stage-mode-learned
    --action-mode --action-mode-one-hot --action-mode-learned
    --character-mode --character-mode-one-hot --character-mode-learned
    --nana-mode --jumps-normalized --no-jumps-normalized
  )

  @doc "Every flag the parser handles: the table plus the explicit multi-key steps."
  @spec flags() :: [String.t()]
  def flags, do: Enum.map(@flag_table, &elem(&1, 0)) ++ @special_flags

  @doc "Keys the table writes (for defaults parity)."
  @spec table_keys() :: [atom()]
  def table_keys, do: Enum.map(@flag_table, &elem(&1, 1))

  @doc false
  def flag_table, do: @flag_table

  defp apply_flag_table(opts, args, ctx) do
    Enum.reduce(@flag_table, opts, fn
      {flag, key, :int}, acc -> parse_int_arg(acc, args, flag, key)
      {flag, key, :float}, acc -> parse_float_arg(acc, args, flag, key)
      {flag, key, :string}, acc -> parse_string_arg(acc, args, flag, key)
      {flag, key, :optional_int}, acc -> parse_optional_int_arg(acc, args, flag, key)
      {flag, key, :flag}, acc -> parse_flag(acc, args, flag, key)
      {flag, key, :neg_flag}, acc -> parse_neg_flag(acc, args, flag, key)
      {flag, key, :flag_or_string}, acc -> parse_flag_or_string(acc, args, flag, key)
      {flag, key, {:atom, allowed}}, acc -> parse_atom_arg(acc, args, flag, key, resolve_allowed(allowed, ctx))
      {flag, key, {:atom_list, allowed}}, acc -> parse_atom_list_arg(acc, args, flag, key, resolve_allowed(allowed, ctx))
    end)
  end

  defp resolve_allowed({:ctx, key, default}, ctx), do: ctx[key] || default
  defp resolve_allowed({:ctx_plus, key, extra}, ctx), do: (ctx[key] || []) ++ extra
  defp resolve_allowed(list, _ctx) when is_list(list), do: list

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

  @doc "Parse a string to float, raising on failure."
  def parse_float!(value) do
    case Float.parse(value) do
      {float, ""} -> float
      _ -> raise ArgumentError, "Invalid float: #{value}"
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

  defp parse_neg_flag(opts, args, flag, key) do
    if has_flag?(args, flag) do
      Keyword.put(opts, key, false)
    else
      opts
    end
  end

  defp parse_button_pos_weight(opts, args) do
    case get_arg_value(args, "--button-pos-weight") do
      nil ->
        opts

      "auto" ->
        Keyword.put(opts, :button_pos_weight, :auto)

      value ->
        weights =
          value
          |> String.split(",")
          |> Enum.map(fn s ->
            case Float.parse(String.trim(s)) do
              {f, ""} -> f
              _ -> raise ArgumentError, "Invalid float in --button-pos-weight: #{s}"
            end
          end)

        if length(weights) != 8 do
          raise ArgumentError,
                "--button-pos-weight requires exactly 8 values (got #{length(weights)}). " <>
                  "Order: A, B, X, Y, Z, L, R, D-Up"
        end

        Keyword.put(opts, :button_pos_weight, weights)
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
