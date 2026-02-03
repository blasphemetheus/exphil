defmodule ExPhil.Training.Config do
  @moduledoc """
  Training configuration parsing and generation.

  Extracts the configuration logic from training scripts into a testable module.
  Handles:
  - Command-line argument parsing
  - Timestamped checkpoint name generation
  - Training config JSON structure

  ## See Also

  - `ExPhil.Training.Imitation` - The main training module that uses this config
  - `ExPhil.Training.Data` - Data loading and batching
  - `ExPhil.Training.Help` - CLI help text generation
  """

  alias ExPhil.Constants
  alias ExPhil.Training.Config.AtomSafety
  alias ExPhil.Training.Config.Checkpoint
  alias ExPhil.Training.Config.Diff
  alias ExPhil.Training.Config.Inference
  alias ExPhil.Training.Config.Presets
  alias ExPhil.Training.Config.Validator

  # Default replays directory - relative path for portability
  # Can be overridden with --replays or --replay-dir
  @default_replays_dir "./replays"
  @default_hidden_sizes [512, 256]

  # Mode allowlists for safe atom conversion
  @valid_action_modes [:one_hot, :learned]
  @valid_character_modes [:one_hot, :learned]
  @valid_stage_modes [:one_hot_full, :one_hot_compact, :learned]
  @valid_nana_modes [:compact, :enhanced, :full]
  @valid_precision_modes [:f32, :bf16, :f16]

  # Training option allowlists (used in convert_value and parse_atom_arg)
  # Note: :hybrid is an alias for :lstm_hybrid (kept for backwards compatibility)
  @valid_backbones [:lstm, :gru, :mamba, :attention, :sliding_window, :lstm_hybrid, :hybrid, :jamba]
  @valid_optimizers [:adam, :adamw, :lamb, :radam, :sgd, :rmsprop]
  @valid_lr_schedules [:constant, :cosine, :cosine_restarts, :exponential, :linear]

  # Presets are now defined in ExPhil.Training.Config.Presets
  # Use Presets.valid_presets() to get the list

  # All valid CLI flags for argument validation
  # This list is used to detect typos and suggest corrections
  @valid_flags [
    "--replays",
    "--replay-dir",
    "--epochs",
    "--batch-size",
    "--hidden-sizes",
    "--max-files",
    "--skip-errors",
    "--fail-fast",
    "--show-errors",
    "--hide-errors",
    "--error-log",
    "--checkpoint",
    "--player",
    "--wandb",
    "--wandb-project",
    "--wandb-name",
    "--temporal",
    "--backbone",
    "--window-size",
    "--stride",
    "--num-layers",
    "--attention-every",
    "--num-heads",
    # Jamba stability options (prevent NaN)
    "--pre-norm",
    "--no-pre-norm",
    "--qk-layernorm",
    "--no-qk-layernorm",
    # Chunked attention for reduced memory
    "--chunked-attention",
    "--no-chunked-attention",
    "--chunk-size",
    # Memory-efficient attention (true O(n) memory via online softmax)
    "--memory-efficient-attention",
    "--no-memory-efficient-attention",
    # FlashAttention NIF for inference (forward-only, requires Ampere+ GPU)
    "--flash-attention-nif",
    "--no-flash-attention-nif",
    "--state-size",
    "--expand-factor",
    "--conv-size",
    "--truncate-bptt",
    "--precision",
    "--mixed-precision",
    "--frame-delay",
    "--frame-delay-augment",
    "--frame-delay-min",
    "--frame-delay-max",
    "--online-robust",
    "--early-stopping",
    "--patience",
    "--min-delta",
    "--save-best",
    "--save-every",
    "--save-every-batches",
    "--lr",
    "--learning-rate",
    "--lr-schedule",
    "--warmup-steps",
    "--decay-steps",
    "--restart-period",
    "--restart-mult",
    "--max-grad-norm",
    "--resume",
    "--name",
    "--accumulation-steps",
    "--val-split",
    "--augment",
    "--mirror-prob",
    "--noise-prob",
    "--noise-scale",
    "--label-smoothing",
    "--dropout",
    "--focal-loss",
    "--focal-gamma",
    "--button-weight",
    "--stick-edge-weight",
    "--no-register",
    "--keep-best",
    "--ema",
    "--ema-decay",
    "--precompute",
    "--no-precompute",
    "--cache-embeddings",
    "--no-cache",
    "--cache-dir",
    # Augmented embedding cache (precompute all variants for ~100x speedup with --augment)
    "--cache-augmented",
    # Number of noisy variants to precompute (default: 2)
    "--num-noisy-variants",
    "--prefetch",
    "--no-prefetch",
    "--gradient-checkpoint",
    "--checkpoint-every",
    "--prefetch-buffer",
    "--layer-norm",
    "--no-layer-norm",
    "--residual",
    "--no-residual",
    "--optimizer",
    "--preset",
    "--dry-run",
    "--character",
    "--characters",
    "--stage",
    "--stages",
    # YAML config file path
    "--config",
    # K-means cluster centers file for stick discretization
    "--kmeans-centers",
    # Process files in chunks for memory efficiency
    "--stream-chunk-size",
    # Auto-select port based on character
    "--train-character",
    # Train on both players per replay
    "--dual-port",
    # Weight sampling by inverse character frequency
    "--balance-characters",
    # Stage embedding mode: full, compact, learned
    "--stage-mode",
    # Action embedding mode: one_hot (399 dims) or learned (64-dim trainable)
    "--action-mode",
    # Character embedding mode: one_hot (33 dims) or learned (64-dim trainable)
    "--character-mode",
    # Nana (Ice Climbers) embedding mode: compact (39 dims), enhanced (14 + ID), full (449 dims)
    "--nana-mode",
    # Jumps remaining representation: normalized (1 dim) or one_hot (7 dims)
    "--jumps-normalized",
    "--no-jumps-normalized",
    # Number of player name embedding dims (0 to disable, default: 112)
    "--num-player-names",
    # Enable style-conditional training (build player registry)
    "--learn-player-styles",
    # Disable style-conditional training
    "--no-learn-player-styles",
    # Path to save/load player registry JSON
    "--player-registry",
    # Minimum games for player to be included in registry (default: 1)
    "--min-player-games",
    # Verbosity control
    # Extra debug output (level 2)
    "--verbose",
    # Minimal output, errors only (level 0)
    "--quiet",
    # Progress bar update interval (batches between updates, default: 1)
    "--log-interval",
    # Reproducibility
    # Random seed for reproducibility
    "--seed",
    # Checkpoint safety
    # Allow overwriting existing checkpoints
    "--overwrite",
    # Fail if checkpoint exists
    "--no-overwrite",
    # Create .bak before overwrite (default)
    "--backup",
    # Skip backup creation
    "--no-backup",
    # Number of backup versions to keep (default: 3)
    "--backup-count",
    # Duplicate detection
    # Skip duplicate replay files by hash (default)
    "--skip-duplicates",
    # Include all files even if duplicates
    "--no-skip-duplicates",
    # Replay quality filtering
    # Minimum quality score (0-100) for replays
    "--min-quality",
    # Show quality distribution stats
    "--show-quality-stats",
    # Memory management
    # Run garbage collection every N batches (0 = disabled)
    "--gc-every",
    # Profiling
    # Enable detailed timing profiler
    "--profile",
    # Parallel validation concurrency (number of concurrent batches)
    "--val-concurrency",
    # Memory-mapped embeddings (for datasets larger than RAM)
    "--mmap-embeddings",
    "--mmap-path",
    # Batch size auto-tuning
    "--auto-batch-size",
    "--auto-batch-min",
    "--auto-batch-max",
    "--auto-batch-backoff"
  ]

  @doc """
  List of available preset names.

  ## Examples

      iex> presets = ExPhil.Training.Config.available_presets()
      iex> :quick in presets
      true
      iex> :production in presets
      true
      iex> :mewtwo in presets
      true

  """
  @spec available_presets() :: [atom()]
  def available_presets, do: Presets.valid_presets()

  @doc """
  Default training options.

  ## Examples

      iex> opts = ExPhil.Training.Config.defaults()
      iex> opts[:epochs]
      10
      iex> opts[:batch_size]
      64
      iex> opts[:temporal]
      false

  """
  @spec defaults() :: keyword()
  def defaults do
    [
      replays: @default_replays_dir,
      epochs: 10,
      batch_size: 64,
      hidden_sizes: @default_hidden_sizes,
      max_files: nil,
      # Error handling for bad replay files
      # Continue past bad files (default: true for convenience)
      skip_errors: true,
      # Show individual file errors (default: true)
      show_errors: true,
      # Optional file path to log errors
      error_log: nil,
      checkpoint: nil,
      player_port: 1,
      # Auto-select port based on character (e.g., :mewtwo)
      train_character: nil,
      # Train on both ports (doubles training data)
      dual_port: false,
      # Weight sampling by inverse character frequency
      balance_characters: false,
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
      # Jamba stability options (Pre-LN + QK LayerNorm prevent NaN)
      pre_norm: true,
      qk_layernorm: true,
      # Chunked attention for reduced memory (20-30% savings)
      # Processes queries in chunks against all keys - same results, lower peak memory
      chunked_attention: false,
      chunk_size: 32,
      # Memory-efficient attention (true O(n) memory via online softmax)
      # Slower than chunked but uses less memory for very long sequences
      memory_efficient_attention: false,
      # FlashAttention NIF for inference (forward-only, no gradients)
      # Requires Ampere+ GPU (RTX 30xx/40xx, A100, H100) for CUDA acceleration
      # Falls back to CPU if CUDA unavailable (slower than Pure Nx due to copy overhead)
      flash_attention_nif: false,
      truncate_bptt: nil,
      # FP32 is default - benchmarks show BF16 is 2x SLOWER on RTX 4090 due to
      # XLA issues: dimension misalignment (287 dims not divisible by 16),
      # type casting overhead, and fallback to FP32 kernels internally.
      # See: https://github.com/openxla/xla/issues/12429
      precision: :f32,
      # Mixed precision training (FP32 master weights + BF16 compute)
      # Not recommended - adds overhead without tensor core benefits on current XLA
      mixed_precision: false,
      frame_delay: 0,
      # Frame delay augmentation for online robustness
      # Enable with --frame-delay-augment or --online-robust
      frame_delay_augment: false,
      # Local play - no delay
      frame_delay_min: 0,
      # Online play - typical Slippi delay (uses Constants.online_frame_delay())
      frame_delay_max: Constants.online_frame_delay(),
      preset: nil,
      character: nil,
      # Early stopping
      early_stopping: false,
      patience: 5,
      min_delta: 0.01,
      # Checkpointing
      save_best: true,
      save_every: nil,
      # Save checkpoint every N batches (useful for streaming mode)
      save_every_batches: nil,
      # Learning rate
      learning_rate: 1.0e-4,
      lr_schedule: :constant,
      # 1 instead of 0 to avoid Polaris/Nx 0.10 compatibility bug
      warmup_steps: 1,
      decay_steps: nil,
      # Cosine restarts (SGDR)
      # Initial period before first restart (T_0)
      restart_period: 1000,
      # Multiply period by this after each restart (T_mult)
      restart_mult: 2,
      # Gradient clipping
      # Clip gradients by global norm (0 = disabled)
      max_grad_norm: 1.0,
      # Resumption
      resume: nil,
      # Model naming
      name: nil,
      # Gradient accumulation
      accumulation_steps: 1,
      # Validation split
      val_split: 0.0,
      # Data augmentation
      augment: false,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      # Label smoothing
      # 0.0 = no smoothing, 0.1 = typical value
      label_smoothing: 0.0,
      # Dropout rate (0.0 = no dropout, 0.1 = typical)
      # Note: MLP backbone defaults to 0.1 if not specified
      dropout: nil,
      # Focal loss for rare actions (Z, L, R buttons)
      focal_loss: false,
      # Higher = more focus on hard examples
      focal_gamma: 2.0,
      # Button loss weight: multiply button loss to balance vs 5 stick/shoulder losses
      # Default 1.0 = equal weight; try 3.0-5.0 to boost button learning
      button_weight: 1.0,
      # Stick edge bucket weight: weight edge buckets (0, 16) higher than center (8)
      # Addresses neutral↔far confusion by penalizing edge mistakes more
      # nil = disabled, 2.0 = edges weighted 2x, linearly interpolated to center
      stick_edge_weight: nil,
      # Registry
      no_register: false,
      # Checkpoint pruning
      # nil = no pruning, N = keep best N epoch checkpoints
      keep_best: nil,
      # Model EMA
      ema: false,
      ema_decay: 0.999,
      # Embedding precomputation (2-3x speedup for MLP training)
      # Precompute embeddings for 2-3x speedup (auto-disabled with augmentation)
      precompute: true,
      # Override for explicitly disabling precomputation
      no_precompute: false,
      # Embedding disk caching (save precomputed embeddings to disk for reuse)
      # Default: true - caches embeddings to disk for faster subsequent runs
      cache_embeddings: true,
      # Force recompute even if cache exists
      no_cache: false,
      # Directory for embedding cache files
      cache_dir: "cache/embeddings",
      # Augmented embedding cache (precompute original + mirrored + noisy variants)
      # When true, enables ~100x speedup for --augment training
      cache_augmented: false,
      # Number of noisy variants to precompute (only used with cache_augmented)
      num_noisy_variants: 2,
      # Data prefetching (load next batch while GPU trains)
      # Only effective with --stream-chunk-size (streaming mode)
      prefetch: false,
      # Number of batches to prefetch
      prefetch_buffer: 2,
      # Layer normalization for MLP backbone
      layer_norm: false,
      # Residual connections for MLP backbone (enables deeper networks, +5-15% accuracy)
      residual: false,
      # Optimizer selection
      # :adam, :adamw, :lamb, :radam
      optimizer: :adam,
      # Gradient checkpointing (memory vs compute trade-off)
      gradient_checkpoint: false,
      # Checkpoint every N layers (1 = every layer, 2 = every other)
      checkpoint_every: 1,
      # Dry run mode - validate config without training
      dry_run: false,
      # Replay filtering
      # Filter replays by character (e.g., [:mewtwo, :fox])
      characters: [],
      # Filter replays by stage (e.g., [:battlefield, :fd])
      stages: [],
      # K-means stick discretization
      # Path to K-means cluster centers file (.nx)
      kmeans_centers: nil,
      # Streaming data loading (process files in chunks to bound memory)
      # nil = load all at once, N = process N files per chunk
      stream_chunk_size: nil,
      # Embedding options
      # Stage: :one_hot_full (64 dims), :one_hot_compact (7 dims), :learned (1 ID)
      stage_mode: :one_hot_compact,
      # Action: :one_hot (399 dims per player) or :learned (64-dim trainable, 2 IDs)
      action_mode: :learned,
      # Character: :one_hot (33 dims per player) or :learned (64-dim trainable, 2 IDs)
      character_mode: :learned,
      # Nana (Ice Climbers): :compact (39 dims), :enhanced (14 + ID), :full (449 dims)
      nana_mode: :compact,
      # Jumps: true = normalized (1 dim), false = one_hot (7 dims)
      jumps_normalized: true,
      # Player name embedding dims (0 = disable, 112 = slippi-ai compatible)
      num_player_names: 112,
      # Enable style-conditional training
      learn_player_styles: false,
      # Path to save/load player registry JSON
      player_registry: nil,
      # Minimum games for player to be in registry
      min_player_games: 1,
      # Verbosity control
      # 0 = quiet (errors only), 1 = normal, 2 = verbose (debug)
      verbosity: 1,
      # Progress bar update interval (batches between updates)
      # Higher = less log spam, faster training (less IO)
      # Default 100 keeps logs readable while still showing progress
      log_interval: 100,
      # Reproducibility
      # Random seed (nil = generate from entropy)
      seed: nil,
      # Checkpoint safety
      # Allow overwriting existing checkpoints
      overwrite: false,
      # Create .bak before overwrite
      backup: true,
      # Number of backup versions to keep
      backup_count: 3,
      # Duplicate detection
      # Skip duplicate replay files by hash
      skip_duplicates: true,
      # Replay quality filtering
      # nil = no quality filtering, N = minimum score (0-100)
      min_quality: nil,
      # Show quality distribution after filtering
      show_quality_stats: false,
      # Memory management
      # Run garbage collection every N batches (0 = disabled)
      gc_every: 100,
      # Profiling
      # Enable detailed timing profiler
      profile: false,
      # Parallel validation
      # Number of concurrent batches during validation (1 = sequential)
      val_concurrency: 4,
      # Memory-mapped embeddings for datasets larger than RAM
      # false = disabled, true = auto path, string = custom path
      mmap_embeddings: false,
      # Explicit path for mmap file (overrides auto-generated path)
      mmap_path: nil,
      # Batch size auto-tuning (find optimal batch size for GPU)
      auto_batch_size: false,
      # Minimum batch size to test
      auto_batch_min: 32,
      # Maximum batch size to test
      auto_batch_max: 4096,
      # Safety factor after finding largest working size (0.8 = 20% headroom)
      auto_batch_backoff: 0.8
    ]
    |> apply_env_defaults()
  end

  # Apply environment variable defaults (lower priority than CLI args)
  defp apply_env_defaults(opts) do
    opts
    |> maybe_env(:replays, "EXPHIL_REPLAYS_DIR")
    |> maybe_env(:wandb_project, "EXPHIL_WANDB_PROJECT")
    |> maybe_env_preset("EXPHIL_DEFAULT_PRESET")
  end

  defp maybe_env(opts, key, env_var) do
    case System.get_env(env_var) do
      nil -> opts
      # Override default with env var
      value -> Keyword.put(opts, key, value)
    end
  end

  defp maybe_env_preset(opts, env_var) do
    case System.get_env(env_var) do
      nil ->
        opts

      value ->
        case AtomSafety.safe_to_atom(value, Presets.valid_presets()) do
          {:ok, preset_atom} -> Keyword.put_new(opts, :preset, preset_atom)
          {:error, _} -> opts
        end
    end
  end

  # Character name mappings (atom -> display name, also accepts aliases)
  @character_map %{
    captain_falcon: "Captain Falcon",
    falcon: "Captain Falcon",
    donkey_kong: "Donkey Kong",
    dk: "Donkey Kong",
    fox: "Fox",
    game_and_watch: "Game & Watch",
    gnw: "Game & Watch",
    gameandwatch: "Game & Watch",
    kirby: "Kirby",
    bowser: "Bowser",
    link: "Link",
    luigi: "Luigi",
    mario: "Mario",
    marth: "Marth",
    mewtwo: "Mewtwo",
    ness: "Ness",
    peach: "Peach",
    pikachu: "Pikachu",
    pika: "Pikachu",
    ice_climbers: "Ice Climbers",
    ics: "Ice Climbers",
    icies: "Ice Climbers",
    jigglypuff: "Jigglypuff",
    puff: "Jigglypuff",
    jiggs: "Jigglypuff",
    samus: "Samus",
    yoshi: "Yoshi",
    zelda: "Zelda",
    sheik: "Sheik",
    falco: "Falco",
    young_link: "Young Link",
    ylink: "Young Link",
    dr_mario: "Dr. Mario",
    doc: "Dr. Mario",
    roy: "Roy",
    pichu: "Pichu",
    ganondorf: "Ganondorf",
    ganon: "Ganondorf"
  }

  # Stage name mappings (atom -> {display name, stage ID})
  @stage_map %{
    fountain_of_dreams: {"Fountain of Dreams", 2},
    fod: {"Fountain of Dreams", 2},
    fountain: {"Fountain of Dreams", 2},
    pokemon_stadium: {"Pokemon Stadium", 3},
    ps: {"Pokemon Stadium", 3},
    stadium: {"Pokemon Stadium", 3},
    yoshis_story: {"Yoshi's Story", 8},
    yoshis: {"Yoshi's Story", 8},
    ys: {"Yoshi's Story", 8},
    dream_land: {"Dream Land", 28},
    dreamland: {"Dream Land", 28},
    dl: {"Dream Land", 28},
    battlefield: {"Battlefield", 31},
    bf: {"Battlefield", 31},
    final_destination: {"Final Destination", 32},
    fd: {"Final Destination", 32}
  }

  @doc "Get display name for a character atom"
  def character_name(char) when is_atom(char), do: Map.get(@character_map, char, to_string(char))

  @doc "Get display name and ID for a stage atom"
  def stage_info(stage) when is_atom(stage), do: Map.get(@stage_map, stage)

  @doc "Get stage ID for a stage atom"
  def stage_id(stage) when is_atom(stage) do
    case Map.get(@stage_map, stage) do
      {_name, id} -> id
      nil -> nil
    end
  end

  @doc "List of valid character atoms"
  def valid_characters, do: Map.keys(@character_map)

  @doc "List of valid stage atoms"
  def valid_stages, do: Map.keys(@stage_map)

  # ============================================================================
  # Config File Loading (YAML)
  # ============================================================================

  @doc """
  Load training configuration from a YAML file.

  Returns `{:ok, opts}` on success, or `{:error, reason}` on failure.

  ## File Format

  The YAML file should contain training options as key-value pairs.
  Keys can use either snake_case or kebab-case.

  ## Example config.yaml

      # Basic training settings
      epochs: 20
      batch_size: 128
      hidden_sizes: [256, 256]

      # Model architecture
      temporal: true
      backbone: mamba
      window_size: 60
      num_layers: 2

      # Regularization
      augment: true
      label_smoothing: 0.1

      # LR schedule
      learning_rate: 0.0001
      lr_schedule: cosine
      warmup_steps: 500

  ## Examples

      iex> ExPhil.Training.Config.load_yaml("missing.yaml")
      {:error, :file_not_found}

  For existing YAML files:

      {:ok, opts} = Config.load_yaml("config/training.yaml")
      opts[:epochs]  # => value from YAML

  """
  @spec load_yaml(String.t()) :: {:ok, keyword()} | {:error, atom() | String.t()}
  def load_yaml(path) do
    case File.read(path) do
      {:ok, content} ->
        parse_yaml(content)

      {:error, :enoent} ->
        {:error, :file_not_found}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Load and merge YAML config with CLI args.

  CLI args take precedence over YAML config, which takes precedence over defaults.

  ## Example

      # With a YAML file containing `batch_size: 128`:
      {:ok, opts} = Config.load_with_yaml("config.yaml", ["--epochs", "5"])
      opts[:epochs]     # => 5 (from CLI, overrides YAML)
      opts[:batch_size] # => 128 (from YAML)

  """
  @spec load_with_yaml(String.t(), [String.t()]) :: {:ok, keyword()} | {:error, any()}
  def load_with_yaml(yaml_path, cli_args) do
    with {:ok, yaml_opts} <- load_yaml(yaml_path),
         cli_opts <- parse_args(cli_args) do
      # CLI args override YAML config
      merged = Keyword.merge(yaml_opts, cli_opts)
      {:ok, merged}
    end
  end

  @doc """
  Parse YAML content into training options.
  """
  @spec parse_yaml(String.t()) :: {:ok, keyword()} | {:error, any()}
  def parse_yaml(content) do
    case YamlElixir.read_from_string(content) do
      {:ok, map} when is_map(map) ->
        opts = convert_yaml_map(map)
        {:ok, opts}

      {:ok, _other} ->
        {:error, :invalid_yaml_format}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Save current configuration to a YAML file.

  Useful for saving the effective configuration after a training run.
  """
  @spec save_yaml(keyword(), String.t()) :: :ok | {:error, any()}
  def save_yaml(opts, path) do
    yaml = opts_to_yaml(opts)
    File.write(path, yaml)
  end

  # Convert YAML map to keyword list with proper atom/type conversion
  defp convert_yaml_map(map) do
    map
    |> Enum.map(fn {key, value} ->
      atom_key = normalize_key(key)
      converted_value = convert_value(atom_key, value)
      {atom_key, converted_value}
    end)
    |> Keyword.new()
  end

  # Normalize key from string (handles kebab-case and snake_case)
  # Uses safe_to_existing_atom to prevent atom table exhaustion from untrusted YAML
  defp normalize_key(key) when is_binary(key) do
    normalized = String.replace(key, "-", "_")

    case AtomSafety.safe_to_existing_atom(normalized) do
      {:ok, atom} -> atom
      # Fall back to known config keys or raise for unknown keys
      {:error, :not_existing} ->
        raise ArgumentError, "Unknown config key: #{inspect(key)}"
    end
  end

  defp normalize_key(key) when is_atom(key), do: key

  # Convert values based on expected types using safe atom conversion
  defp convert_value(:backbone, value) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, @valid_backbones ++ [:mlp])
  end

  defp convert_value(:lr_schedule, value) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, @valid_lr_schedules)
  end

  defp convert_value(:optimizer, value) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, @valid_optimizers)
  end

  defp convert_value(:precision, value) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, @valid_precision_modes)
  end

  defp convert_value(:preset, value) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, Presets.valid_presets())
  end

  defp convert_value(:character, value) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, Map.keys(@character_map))
  end

  defp convert_value(:characters, values) when is_list(values) do
    valid_chars = Map.keys(@character_map)
    Enum.map(values, &AtomSafety.safe_to_atom!(&1, valid_chars))
  end

  defp convert_value(:stages, values) when is_list(values) do
    valid_stages = Map.keys(@stage_map)
    Enum.map(values, &AtomSafety.safe_to_atom!(&1, valid_stages))
  end

  defp convert_value(:hidden_sizes, values) when is_list(values), do: values
  defp convert_value(_key, value), do: value

  # Convert opts to YAML string
  defp opts_to_yaml(opts) do
    opts
    |> Enum.map(fn {key, value} ->
      yaml_key = key |> to_string() |> String.replace("_", "-")
      yaml_value = format_yaml_value(value)
      "#{yaml_key}: #{yaml_value}"
    end)
    |> Enum.join("\n")
  end

  defp format_yaml_value(value) when is_list(value) do
    items = Enum.map(value, &to_string/1) |> Enum.join(", ")
    "[#{items}]"
  end

  defp format_yaml_value(value) when is_atom(value), do: to_string(value)
  defp format_yaml_value(value) when is_binary(value), do: "\"#{value}\""
  defp format_yaml_value(value), do: to_string(value)

  # ============================================================================
  # Training Presets
  # ============================================================================

  @doc """
  Get training options for a preset.

  ## Available Presets

  ### CPU Presets (No GPU Required)
  - `:quick` - Fast iteration for testing (1 epoch, 5 files, small MLP)
  - `:standard` - Balanced CPU training (10 epochs, 50 files, augmentation)
  - `:full_cpu` - Maximum CPU quality (20 epochs, 100 files, all regularization)

  ### GPU Presets (Requires CUDA/ROCm)
  - `:gpu_quick` - Fast GPU test (3 epochs, 20 files, Mamba temporal)
  - `:gpu_mlp_quick` - Fastest GPU test (5 epochs, 50 files, MLP + precompute, 2-3x faster)
  - `:gpu_lstm_quick` - LSTM backbone test (3 epochs, 30 files)
  - `:gpu_gru_quick` - GRU backbone test (3 epochs, 30 files)
  - `:gpu_attention_quick` - Attention backbone test (3 epochs, 30 files)
  - `:gpu_standard` - Standard GPU training (20 epochs, Mamba, all features)
  - `:full` - High quality GPU (50 epochs, Mamba, temporal, full regularization)
  - `:production` - Maximum quality (100 epochs, Mamba, all optimizations, EMA)

  ### Character Presets (Built on :production)
  - `:mewtwo` - Longer context (90 frames) for teleport recovery tracking
  - `:ganondorf` - Standard context (60 frames) for spacing-focused play
  - `:link` - Extended context (75 frames) for projectile tracking
  - `:gameandwatch` - Shorter context (45 frames) since no L-cancel
  - `:zelda` - Standard context (60 frames) for transform mechanics

  ## Best Practices Applied

  | Feature | quick | standard | full | production |
  |---------|-------|----------|------|------------|
  | Augmentation | ✗ | ✓ | ✓ | ✓ |
  | Label Smoothing | ✗ | 0.05 | 0.1 | 0.1 |
  | EMA | ✗ | ✗ | ✓ | ✓ |
  | LR Schedule | constant | cosine | cosine | cosine_restarts |
  | Val Split | ✗ | 0.1 | 0.1 | 0.15 |
  | Early Stopping | ✗ | ✓ | ✓ | ✓ |

  ## Examples

      iex> opts = ExPhil.Training.Config.preset(:quick)
      iex> opts[:epochs]
      1
      iex> opts[:max_files]
      5
      iex> opts[:temporal]
      false

      iex> opts = ExPhil.Training.Config.preset(:production)
      iex> opts[:epochs]
      100
      iex> opts[:ema]
      true
      iex> opts[:lr_schedule]
      :cosine_restarts

      iex> opts = ExPhil.Training.Config.preset(:mewtwo)
      iex> opts[:character]
      :mewtwo
      iex> opts[:window_size]
      90

  """
  @spec preset(atom() | String.t()) :: keyword() | no_return()
  defdelegate preset(name), to: Presets, as: :get

  # ============================================================================
  # Config Diff Display
  # ============================================================================

  # Delegated to ExPhil.Training.Config.Diff
  # See that module for implementation details

  @doc """
  Get a list of options that differ from defaults.
  Delegates to `ExPhil.Training.Config.Diff.from_defaults/3`.
  """
  @spec diff_from_defaults(keyword(), keyword()) :: [{atom(), any(), any()}]
  def diff_from_defaults(opts, diff_opts \\ []) do
    Diff.from_defaults(opts, &defaults/0, diff_opts)
  end

  @doc """
  Format config diff as a human-readable string.
  Delegates to `ExPhil.Training.Config.Diff.format/2`.
  """
  @spec format_diff(keyword()) :: String.t() | nil
  def format_diff(opts) do
    Diff.format(opts, &defaults/0)
  end

  # ============================================================================
  # Validation
  # ============================================================================
  # Validation logic is in ExPhil.Training.Config.Validator

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
  @spec validate(keyword()) :: {:ok, keyword()} | {:error, [String.t()]}
  def validate(opts) do
    Validator.validate(opts, validation_context())
  end

  @doc """
  Validate training options, raising on errors.

  Returns opts if valid, raises `ArgumentError` if invalid.
  Warnings are logged but don't cause validation to fail.

  ## Examples

      iex> ExPhil.Training.Config.validate!(epochs: 10, batch_size: 64)
      [epochs: 10, batch_size: 64]

  Invalid configurations raise an ArgumentError:

      Config.validate!(epochs: -1)
      # => raises ArgumentError with "Invalid training configuration..."

  """
  @spec validate!(keyword()) :: keyword()
  def validate!(opts) do
    Validator.validate!(opts, validation_context())
  end

  # Build the validation context with allowlists
  defp validation_context do
    %{
      valid_backbones: @valid_backbones,
      valid_optimizers: @valid_optimizers,
      valid_lr_schedules: @valid_lr_schedules
    }
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
    valid_backbones = @valid_backbones ++ [:mlp]

    []
    |> maybe_add_override(args, "--epochs", :epochs, &String.to_integer/1)
    |> maybe_add_override(args, "--batch-size", :batch_size, &String.to_integer/1)
    |> maybe_add_override(args, "--max-files", :max_files, &String.to_integer/1)
    |> maybe_add_override(args, "--hidden-sizes", :hidden_sizes, &parse_hidden_sizes/1)
    |> maybe_add_override(args, "--window-size", :window_size, &String.to_integer/1)
    |> maybe_add_override(args, "--backbone", :backbone, &AtomSafety.safe_to_atom!(&1, valid_backbones))
    |> maybe_add_override(args, "--num-layers", :num_layers, &String.to_integer/1)
    |> maybe_add_override(args, "--attention-every", :attention_every, &String.to_integer/1)
    |> maybe_add_override(args, "--frame-delay", :frame_delay, &String.to_integer/1)
    |> maybe_add_override(args, "--replays", :replays, & &1)
    |> maybe_add_override(args, "--replay-dir", :replays, & &1)
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

      iex> opts = ExPhil.Training.Config.parse_args(["--epochs", "5", "--temporal"])
      iex> opts[:epochs]
      5
      iex> opts[:temporal]
      true

      iex> opts = ExPhil.Training.Config.parse_args(["--preset", "quick"])
      iex> opts[:epochs]
      1
      iex> opts[:max_files]
      5

      iex> opts = ExPhil.Training.Config.parse_args(["--preset", "quick", "--epochs", "3"])
      iex> opts[:epochs]
      3
      iex> opts[:max_files]
      5

  """
  @spec parse_args([String.t()]) :: keyword()
  def parse_args(args) when is_list(args) do
    # Check if config file is specified first
    base_opts =
      if has_flag_value?(args, "--config") do
        config_path = get_arg_value(args, "--config")

        case load_yaml(config_path) do
          {:ok, yaml_opts} ->
            # Merge YAML opts on top of defaults
            Keyword.merge(defaults(), yaml_opts)

          {:error, reason} ->
            IO.puts(:stderr, "Error loading config file: #{inspect(reason)}")
            System.halt(1)
        end
      else
        defaults()
      end

    # Check if preset is specified - if so, use apply_preset flow
    if has_flag_value?(args, "--preset") do
      apply_preset(base_opts, args)
    else
      # No preset - standard parsing flow
      parse_args_standard(args, base_opts)
    end
  end

  defp parse_args_standard(args, base_opts) do
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
      # --fail-fast is opposite of skip-errors
      if opts[:fail_fast], do: Keyword.put(opts, :skip_errors, false), else: opts
    end)
    |> then(fn opts ->
      # --hide-errors disables show_errors
      if opts[:hide_errors], do: Keyword.put(opts, :show_errors, false), else: opts
    end)
    |> parse_string_arg(args, "--checkpoint", :checkpoint)
    |> parse_int_arg(args, "--player", :player_port)
    |> parse_atom_arg(args, "--train-character", :train_character, Map.keys(@character_map))
    |> parse_flag(args, "--dual-port", :dual_port)
    |> parse_flag(args, "--balance-characters", :balance_characters)
    |> parse_flag(args, "--wandb", :wandb)
    |> parse_string_arg(args, "--wandb-project", :wandb_project)
    |> parse_string_arg(args, "--wandb-name", :wandb_name)
    |> parse_flag(args, "--temporal", :temporal)
    |> parse_atom_arg(args, "--backbone", :backbone, @valid_backbones ++ [:mlp])
    |> parse_int_arg(args, "--window-size", :window_size)
    |> parse_int_arg(args, "--stride", :stride)
    |> parse_int_arg(args, "--num-layers", :num_layers)
    |> parse_int_arg(args, "--attention-every", :attention_every)
    # Jamba stability options
    |> parse_flag(args, "--pre-norm", :pre_norm)
    |> parse_flag(args, "--no-pre-norm", :no_pre_norm)
    |> then(fn opts ->
      # --no-pre-norm disables Pre-LayerNorm
      if opts[:no_pre_norm], do: Keyword.put(opts, :pre_norm, false), else: opts
    end)
    |> parse_flag(args, "--qk-layernorm", :qk_layernorm)
    |> parse_flag(args, "--no-qk-layernorm", :no_qk_layernorm)
    |> then(fn opts ->
      # --no-qk-layernorm disables QK LayerNorm
      if opts[:no_qk_layernorm], do: Keyword.put(opts, :qk_layernorm, false), else: opts
    end)
    # Chunked attention for reduced memory
    |> parse_flag(args, "--chunked-attention", :chunked_attention)
    |> parse_flag(args, "--no-chunked-attention", :no_chunked_attention)
    |> then(fn opts ->
      # --no-chunked-attention disables chunked attention
      if opts[:no_chunked_attention], do: Keyword.put(opts, :chunked_attention, false), else: opts
    end)
    |> parse_int_arg(args, "--chunk-size", :chunk_size)
    # Memory-efficient attention (true O(n) memory)
    |> parse_flag(args, "--memory-efficient-attention", :memory_efficient_attention)
    |> parse_flag(args, "--no-memory-efficient-attention", :no_memory_efficient_attention)
    |> then(fn opts ->
      if opts[:no_memory_efficient_attention], do: Keyword.put(opts, :memory_efficient_attention, false), else: opts
    end)
    # FlashAttention NIF for inference (forward-only, requires Ampere+ GPU)
    |> parse_flag(args, "--flash-attention-nif", :flash_attention_nif)
    |> parse_flag(args, "--no-flash-attention-nif", :no_flash_attention_nif)
    |> then(fn opts ->
      if opts[:no_flash_attention_nif], do: Keyword.put(opts, :flash_attention_nif, false), else: opts
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
    |> parse_atom_arg(args, "--lr-schedule", :lr_schedule, @valid_lr_schedules)
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
      # --no-prefetch disables prefetching
      if opts[:no_prefetch], do: Keyword.put(opts, :prefetch, false), else: opts
    end)
    |> then(fn opts ->
      # --no-precompute disables embedding precomputation
      if opts[:no_precompute], do: Keyword.put(opts, :precompute, false), else: opts
    end)
    |> parse_int_arg(args, "--prefetch-buffer", :prefetch_buffer)
    |> parse_flag(args, "--layer-norm", :layer_norm)
    |> parse_flag(args, "--no-layer-norm", :no_layer_norm)
    |> then(fn opts ->
      # --no-layer-norm disables layer normalization
      if opts[:no_layer_norm], do: Keyword.put(opts, :layer_norm, false), else: opts
    end)
    |> parse_flag(args, "--residual", :residual)
    |> parse_flag(args, "--no-residual", :no_residual)
    |> then(fn opts ->
      # --no-residual disables residual connections
      if opts[:no_residual], do: Keyword.put(opts, :residual, false), else: opts
    end)
    |> parse_atom_arg(args, "--optimizer", :optimizer, @valid_optimizers)
    |> parse_flag(args, "--dry-run", :dry_run)
    |> parse_atom_list_arg(args, "--character", :characters, Map.keys(@character_map))
    |> parse_atom_list_arg(args, "--characters", :characters, Map.keys(@character_map))
    |> parse_atom_list_arg(args, "--stage", :stages, Map.keys(@stage_map))
    |> parse_atom_list_arg(args, "--stages", :stages, Map.keys(@stage_map))
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
      # --no-overwrite makes overwrite explicitly false
      if opts[:no_overwrite], do: Keyword.put(opts, :overwrite, false), else: opts
    end)
    |> then(fn opts ->
      # --no-backup disables backup
      if opts[:no_backup], do: Keyword.put(opts, :backup, false), else: opts
    end)
    |> then(fn opts ->
      # --no-skip-duplicates disables duplicate detection
      if opts[:no_skip_duplicates], do: Keyword.put(opts, :skip_duplicates, false), else: opts
    end)
    |> then(fn opts ->
      # --no-learn-player-styles disables style-conditional training
      if opts[:no_learn_player_styles],
        do: Keyword.put(opts, :learn_player_styles, false),
        else: opts
    end)
  end

  # Parse verbosity flags: --quiet (0), default (1), --verbose (2)
  defp parse_verbosity_flags(opts, args) do
    cond do
      "--quiet" in args -> Keyword.put(opts, :verbosity, 0)
      "--verbose" in args -> Keyword.put(opts, :verbosity, 2)
      true -> opts
    end
  end

  # Parse stage mode with alias support (full, compact, learned -> atoms)
  # Stage mode aliases for user convenience
  @stage_mode_aliases %{
    "full" => :one_hot_full,
    "one_hot_full" => :one_hot_full,
    "compact" => :one_hot_compact,
    "one_hot_compact" => :one_hot_compact,
    "learned" => :learned
  }

  defp parse_stage_mode_arg(opts, args) do
    case get_arg_value(args, "--stage-mode") do
      nil ->
        opts

      value ->
        downcased = String.downcase(value)

        mode =
          case Map.fetch(@stage_mode_aliases, downcased) do
            {:ok, atom} -> atom
            :error -> AtomSafety.safe_to_atom!(downcased, @valid_stage_modes)
          end

        Keyword.put(opts, :stage_mode, mode)
    end
  end

  # Action mode aliases for user convenience
  @action_mode_aliases %{
    "one_hot" => :one_hot,
    "onehot" => :one_hot,
    "learned" => :learned
  }

  # Parse action mode: one_hot or learned
  defp parse_action_mode_arg(opts, args) do
    case get_arg_value(args, "--action-mode") do
      nil ->
        opts

      value ->
        downcased = String.downcase(value)

        mode =
          case Map.fetch(@action_mode_aliases, downcased) do
            {:ok, atom} -> atom
            :error -> AtomSafety.safe_to_atom!(downcased, @valid_action_modes)
          end

        Keyword.put(opts, :action_mode, mode)
    end
  end

  # Character mode aliases for user convenience
  @character_mode_aliases %{
    "one_hot" => :one_hot,
    "onehot" => :one_hot,
    "learned" => :learned
  }

  # Parse character mode: one_hot or learned
  defp parse_character_mode_arg(opts, args) do
    case get_arg_value(args, "--character-mode") do
      nil ->
        opts

      value ->
        downcased = String.downcase(value)

        mode =
          case Map.fetch(@character_mode_aliases, downcased) do
            {:ok, atom} -> atom
            :error -> AtomSafety.safe_to_atom!(downcased, @valid_character_modes)
          end

        Keyword.put(opts, :character_mode, mode)
    end
  end

  # Parse nana mode: compact, enhanced, or full
  defp parse_nana_mode_arg(opts, args) do
    case get_arg_value(args, "--nana-mode") do
      nil ->
        opts

      value ->
        # All nana modes are direct matches, no aliases needed
        mode = AtomSafety.safe_to_atom!(String.downcase(value), @valid_nana_modes)

        Keyword.put(opts, :nana_mode, mode)
    end
  end

  # Parse jumps normalized flag
  defp parse_jumps_normalized_arg(opts, args) do
    cond do
      "--no-jumps-normalized" in args -> Keyword.put(opts, :jumps_normalized, false)
      "--jumps-normalized" in args -> Keyword.put(opts, :jumps_normalized, true)
      true -> opts
    end
  end

  # Parse comma-separated list of atoms (e.g., "mewtwo,fox,falco" -> [:mewtwo, :fox, :falco])
  # Uses safe atom conversion with an allowlist to prevent atom table exhaustion
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

        # Merge with existing list (allows both --character and --characters)
        existing = Keyword.get(opts, key, [])
        Keyword.put(opts, key, Enum.uniq(existing ++ atoms))
    end
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

  Auto-generated names use random words and timestamps:

      iex> opts = [checkpoint: nil, temporal: false, character: nil, name: nil]
      iex> path = ExPhil.Training.Config.ensure_checkpoint_name(opts) |> Keyword.get(:checkpoint)
      iex> String.starts_with?(path, "checkpoints/") and String.ends_with?(path, ".axon")
      true

  Character and backbone are included in auto-generated names:

      iex> opts = [checkpoint: nil, temporal: true, backbone: :mamba, character: :mewtwo, name: nil]
      iex> path = ExPhil.Training.Config.ensure_checkpoint_name(opts) |> Keyword.get(:checkpoint)
      iex> String.contains?(path, "mewtwo_mamba_") and String.ends_with?(path, ".axon")
      true

  Explicit checkpoints are preserved:

      iex> opts = [checkpoint: "my_model.axon"]
      iex> ExPhil.Training.Config.ensure_checkpoint_name(opts) |> Keyword.get(:checkpoint)
      "my_model.axon"

  """
  def ensure_checkpoint_name(opts) do
    if opts[:checkpoint] do
      opts
    else
      alias ExPhil.Training.Naming

      timestamp = generate_timestamp()
      backbone = if opts[:temporal], do: opts[:backbone], else: :mlp
      auto_name = Naming.generate()
      user_name = opts[:name]
      character = opts[:character]

      # If user provided a name, use it directly without backbone prefix
      # If auto-generated, include backbone for clarity
      checkpoint_name =
        cond do
          user_name && character ->
            "checkpoints/#{character}_#{user_name}_#{timestamp}.axon"

          user_name ->
            "checkpoints/#{user_name}_#{timestamp}.axon"

          character ->
            "checkpoints/#{character}_#{backbone}_#{auto_name}_#{timestamp}.axon"

          true ->
            "checkpoints/#{backbone}_#{auto_name}_#{timestamp}.axon"
        end

      # Store the effective name in opts for display
      display_name = user_name || auto_name

      opts
      |> Keyword.put(:checkpoint, checkpoint_name)
      |> Keyword.put(:name, display_name)
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

      # Data filtering (for provenance)
      characters: format_atom_list(opts[:characters]),
      stages: format_atom_list(opts[:stages]),

      # Replay manifest (for provenance)
      replay_count: results[:replay_count],
      replay_files: results[:replay_files],
      replay_manifest_hash: results[:replay_manifest_hash],
      character_distribution: results[:character_distribution],

      # Model architecture
      temporal: opts[:temporal],
      backbone: if(opts[:temporal], do: to_string(opts[:backbone]), else: "mlp"),
      hidden_sizes: opts[:hidden_sizes],
      embed_size: results[:embed_size],
      layer_norm: opts[:layer_norm],
      residual: opts[:residual],
      kmeans_centers: opts[:kmeans_centers],

      # Embedding options (always capture effective value, not just explicit CLI args)
      stage_mode: get_embedding_mode(opts, :stage_mode),
      action_mode: get_embedding_mode(opts, :action_mode),
      character_mode: get_embedding_mode(opts, :character_mode),
      nana_mode: get_embedding_mode(opts, :nana_mode),
      jumps_normalized: Keyword.get(opts, :jumps_normalized, defaults()[:jumps_normalized]),

      # Temporal options
      window_size: opts[:window_size],
      stride: opts[:stride],
      num_layers: opts[:num_layers],
      truncate_bptt: opts[:truncate_bptt],

      # Mamba options
      state_size: opts[:state_size],
      expand_factor: opts[:expand_factor],
      conv_size: opts[:conv_size],

      # Attention/Jamba options
      attention_every: opts[:attention_every],
      num_heads: opts[:num_heads],
      head_dim: opts[:head_dim],

      # Training parameters
      epochs: opts[:epochs],
      batch_size: opts[:batch_size],
      precision: to_string(opts[:precision]),
      frame_delay: opts[:frame_delay],

      # Optimizer settings
      learning_rate: opts[:lr],
      lr_schedule: opts[:lr_schedule] && to_string(opts[:lr_schedule]),
      warmup_steps: opts[:warmup_steps],
      optimizer: opts[:optimizer] && to_string(opts[:optimizer]),
      max_grad_norm: opts[:max_grad_norm],
      accumulation_steps: opts[:accumulation_steps],

      # Regularization
      label_smoothing: opts[:label_smoothing],
      dropout: opts[:dropout],
      focal_loss: opts[:focal_loss],
      focal_gamma: opts[:focal_gamma],
      button_weight: opts[:button_weight],
      stick_edge_weight: opts[:stick_edge_weight],
      ema: opts[:ema],
      ema_decay: opts[:ema_decay],

      # Data options
      train_character: opts[:train_character] && to_string(opts[:train_character]),
      augment: opts[:augment],
      val_split: opts[:val_split],
      seed: opts[:seed],

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
  Compute a SHA256 hash of a list of file paths for replay manifest.

  Sorts paths alphabetically before hashing for determinism.
  """
  @spec compute_manifest_hash([String.t()]) :: String.t()
  def compute_manifest_hash([]), do: nil

  def compute_manifest_hash(paths) when is_list(paths) do
    paths
    |> Enum.sort()
    |> Enum.join("\n")
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> then(&"sha256:#{&1}")
  end

  # Format a list of atoms as strings for JSON serialization
  defp format_atom_list(nil), do: nil
  defp format_atom_list([]), do: nil

  defp format_atom_list(atoms) when is_list(atoms) do
    Enum.map(atoms, &to_string/1)
  end

  # Get embedding mode with default fallback, converting to string for JSON
  defp get_embedding_mode(opts, key) do
    value = Keyword.get(opts, key, defaults()[key])

    case value do
      nil -> nil
      atom when is_atom(atom) -> to_string(atom)
      other -> other
    end
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
      nil ->
        opts

      value ->
        # Use Float.parse to handle scientific notation (e.g., "1e-4")
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

  # Parse a flag that can be either a boolean (--flag) or a string (--flag value)
  # If next arg starts with --, treat as boolean flag
  defp parse_flag_or_string(opts, args, flag, key) do
    if has_flag?(args, flag) do
      case get_arg_value(args, flag) do
        nil ->
          # No next arg, treat as boolean
          Keyword.put(opts, key, true)

        value when is_binary(value) ->
          # If next arg is another flag, treat as boolean
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

  # --online-robust is a convenience flag that enables frame delay augmentation
  # with sensible defaults for training models that work well on Slippi online
  defp parse_online_robust_flag(opts, args) do
    if has_flag?(args, "--online-robust") do
      opts
      |> Keyword.put(:frame_delay_augment, true)

      # Use defaults: 0-18 frame range (local to online play)
      # Can be overridden with explicit --frame-delay-min/max
    else
      opts
    end
  end

  @doc """
  List of valid CLI flags.

  ## Examples

      iex> flags = ExPhil.Training.Config.valid_flags()
      iex> "--epochs" in flags
      true
      iex> "--batch-size" in flags
      true
      iex> "--preset" in flags
      true

  """
  @spec valid_flags() :: [String.t()]
  def valid_flags, do: @valid_flags

  @doc """
  Validate command-line arguments for unrecognized flags.

  Returns `{:ok, []}` if all flags are valid, or `{:ok, warnings}` with a list
  of warning messages for unrecognized flags with suggestions.

  ## Examples

      iex> ExPhil.Training.Config.validate_args(["--epochs", "10", "--batch-size", "32"])
      {:ok, []}

      iex> {:ok, [warning]} = ExPhil.Training.Config.validate_args(["--ephocs", "10"])
      iex> warning =~ "Did you mean '--epochs'"
      true

  """
  @spec validate_args(list(String.t())) :: {:ok, list(String.t())}
  def validate_args(args) when is_list(args) do
    # Extract all flags (args starting with --)
    input_flags =
      args
      |> Enum.filter(&String.starts_with?(&1, "--"))
      |> Enum.uniq()

    # Find unrecognized flags
    unrecognized = input_flags -- @valid_flags

    warnings =
      Enum.map(unrecognized, fn flag ->
        case suggest_flag(flag) do
          nil -> "Unknown flag '#{flag}'. Run with --help to see available options."
          suggestion -> "Unknown flag '#{flag}'. Did you mean '#{suggestion}'?"
        end
      end)

    {:ok, warnings}
  end

  @doc """
  Validate args and print warnings if any.

  This is a convenience function that calls validate_args/1 and prints
  any warnings to stderr, returning {:ok, opts} or {:error, reason}.

  ## Examples

      iex> ExPhil.Training.Config.validate_args!(["--epochs", "10"])
      :ok

  Note: Invalid flags print warnings to stderr but still return `:ok`:

      ExPhil.Training.Config.validate_args!(["--ephocs", "10"])
      # Prints: "⚠️  Unknown flag '--ephocs'. Did you mean '--epochs'?"
      # Returns: :ok

  """
  @spec validate_args!(list(String.t())) :: :ok
  def validate_args!(args) do
    {:ok, warnings} = validate_args(args)

    if warnings != [] do
      IO.puts(:stderr, "")

      Enum.each(warnings, fn warning ->
        IO.puts(:stderr, "⚠️  #{warning}")
      end)

      IO.puts(:stderr, "")
    end

    :ok
  end

  # Suggest a valid flag for a typo using Levenshtein distance
  # Returns nil if no close match found (distance > 3)
  defp suggest_flag(typo) do
    @valid_flags
    |> Enum.map(fn flag -> {flag, levenshtein_distance(typo, flag)} end)
    |> Enum.min_by(fn {_flag, distance} -> distance end)
    |> case do
      {flag, distance} when distance <= 3 -> flag
      _ -> nil
    end
  end

  # =============================================================================
  # Smart Flag Inference
  # =============================================================================

  # Delegated to ExPhil.Training.Config.Inference
  # See that module for implementation details

  @doc """
  Apply smart defaults based on flag combinations.

  Delegates to `ExPhil.Training.Config.Inference.infer_smart_defaults/1`.
  See that module for details on what inferences are applied.
  """
  @spec infer_smart_defaults(keyword()) :: {keyword(), list(String.t())}
  defdelegate infer_smart_defaults(opts), to: Inference

  # Calculate Levenshtein distance between two strings
  # This is the minimum number of single-character edits (insertions,
  # deletions, substitutions) needed to transform one string into another
  defp levenshtein_distance(s1, s2) do
    s1_chars = String.graphemes(s1)
    s2_chars = String.graphemes(s2)
    s2_len = length(s2_chars)

    # Use dynamic programming with a 2-row approach for memory efficiency
    # Initialize first row: distances from empty string to s2 prefixes
    initial_row = Enum.to_list(0..s2_len)

    # Process each character of s1
    {final_row, _} =
      Enum.reduce(Enum.with_index(s1_chars), {initial_row, 0}, fn {c1, i}, {prev_row, _} ->
        # Start new row with distance from s1 prefix to empty string
        first = i + 1

        # Process each character of s2
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

  # =============================================================================
  # Checkpoint Safety Functions
  # =============================================================================

  # Delegated to ExPhil.Training.Config.Checkpoint
  # See that module for implementation details

  @doc """
  Check if a checkpoint path would overwrite an existing file.
  Delegates to `ExPhil.Training.Config.Checkpoint.check_checkpoint_path/2`.
  """
  defdelegate check_checkpoint_path(path, opts \\ []), to: Checkpoint

  @doc """
  Format file info for display in collision warnings.
  Delegates to `ExPhil.Training.Config.Checkpoint.format_file_info/1`.
  """
  defdelegate format_file_info(info), to: Checkpoint

  @doc """
  Backup an existing checkpoint before overwriting.
  Delegates to `ExPhil.Training.Config.Checkpoint.backup_checkpoint/2`.
  """
  defdelegate backup_checkpoint(path, opts \\ []), to: Checkpoint

  # =============================================================================
  # Reproducibility Functions
  # =============================================================================

  @doc """
  Initialize random seed for reproducibility.

  If seed is provided, uses it directly. Otherwise generates a seed from system entropy.
  Returns the seed used (for logging).

  Sets seeds for:
  - Erlang's :rand module
  - Nx global default seed (for parameter initialization, dropout)
  """
  @spec init_seed(integer() | nil) :: integer()
  def init_seed(nil) do
    # Generate seed from system entropy
    seed = :rand.uniform(2_147_483_647)
    init_seed(seed)
  end

  def init_seed(seed) when is_integer(seed) do
    # Seed Erlang's random module
    :rand.seed(:exsss, {seed, seed, seed})

    # Seed Nx's global key (affects Nx.Random operations)
    # Note: Nx uses a PRNG key system, this sets the default
    Nx.default_backend(EXLA.Backend)
    Application.put_env(:nx, :default_defn_options, seed: seed)

    seed
  end

  @doc """
  Get verbosity level description.
  """
  @spec verbosity_name(integer()) :: String.t()
  def verbosity_name(0), do: "quiet"
  def verbosity_name(1), do: "normal"
  def verbosity_name(2), do: "verbose"
  def verbosity_name(_), do: "unknown"
end
