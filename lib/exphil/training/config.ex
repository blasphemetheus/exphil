defmodule ExPhil.Training.Config do
  @moduledoc """
  Training configuration parsing and generation.

  Extracts the configuration logic from training scripts into a testable module.
  Handles:
  - Command-line argument parsing
  - Timestamped checkpoint name generation
  - Training config JSON structure
  """

  alias ExPhil.Training.Help

  # Default replays directory - relative path for portability
  # Can be overridden with --replays or --replay-dir
  @default_replays_dir "./replays"
  @default_hidden_sizes [512, 256]

  @valid_presets [
    :quick,
    :standard,
    :full,
    :full_cpu,
    :gpu_quick,
    :gpu_mlp_quick,
    :gpu_lstm_quick,
    :gpu_gru_quick,
    :gpu_attention_quick,
    :gpu_standard,
    :production,
    # RTX 4090 optimized (24GB VRAM)
    :rtx4090_quick,
    :rtx4090_standard,
    :rtx4090_full,
    # Character presets
    :mewtwo,
    :ganondorf,
    :link,
    :gameandwatch,
    :zelda
  ]

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
    "--focal-loss",
    "--focal-gamma",
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
    "--gc-every"
  ]

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
      # Online play - typical Slippi delay
      frame_delay_max: 18,
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
      # Focal loss for rare actions (Z, L, R buttons)
      focal_loss: false,
      # Higher = more focus on hard examples
      focal_gamma: 2.0,
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
      prefetch: true,
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
      gc_every: 100
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
        preset = String.to_atom(value)

        if preset in @valid_presets do
          Keyword.put_new(opts, :preset, preset)
        else
          opts
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

      iex> Config.load_yaml("config/training.yaml")
      {:ok, [epochs: 20, batch_size: 128, ...]}

      iex> Config.load_yaml("missing.yaml")
      {:error, :file_not_found}
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

  ## Examples

      iex> Config.load_with_yaml("config.yaml", ["--epochs", "5"])
      {:ok, [epochs: 5, batch_size: 128, ...]}  # CLI overrides YAML
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
  defp normalize_key(key) when is_binary(key) do
    key
    |> String.replace("-", "_")
    |> String.to_atom()
  end

  defp normalize_key(key) when is_atom(key), do: key

  # Convert values based on expected types
  defp convert_value(:backbone, value) when is_binary(value), do: String.to_atom(value)
  defp convert_value(:lr_schedule, value) when is_binary(value), do: String.to_atom(value)
  defp convert_value(:optimizer, value) when is_binary(value), do: String.to_atom(value)
  defp convert_value(:precision, value) when is_binary(value), do: String.to_atom(value)
  defp convert_value(:preset, value) when is_binary(value), do: String.to_atom(value)
  defp convert_value(:character, value) when is_binary(value), do: String.to_atom(value)

  defp convert_value(:characters, values) when is_list(values) do
    Enum.map(values, &String.to_atom/1)
  end

  defp convert_value(:stages, values) when is_list(values) do
    Enum.map(values, &String.to_atom/1)
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

      iex> Config.preset(:quick)
      [epochs: 1, max_files: 5, hidden_sizes: [32, 32], ...]

      iex> Config.preset(:production)
      [epochs: 100, hidden_sizes: [256, 256], ema: true, lr_schedule: :cosine_restarts, ...]

      iex> Config.preset(:mewtwo)
      [character: :mewtwo, epochs: 100, window_size: 90, ...]

  """

  # ============================================================================
  # CPU Presets (No GPU Required)
  # ============================================================================

  def preset(:quick) do
    # Fast iteration for testing code changes
    # Use: mix run scripts/train_from_replays.exs --preset quick
    [
      epochs: 1,
      max_files: 5,
      batch_size: 32,
      hidden_sizes: [32, 32],
      temporal: false,
      # No regularization - just testing
      preset: :quick
    ]
  end

  def preset(:standard) do
    # Balanced CPU training with proper regularization
    # Use: mix run scripts/train_from_replays.exs --preset standard
    [
      epochs: 10,
      max_files: 50,
      batch_size: 64,
      hidden_sizes: [64, 64],
      temporal: false,
      # Regularization
      augment: true,
      mirror_prob: 0.5,
      noise_prob: 0.2,
      noise_scale: 0.01,
      label_smoothing: 0.05,
      # Validation & early stopping
      val_split: 0.1,
      early_stopping: true,
      patience: 5,
      # LR schedule
      lr_schedule: :cosine,
      learning_rate: 3.0e-4,
      save_best: true,
      # Precompute embeddings for 2-3x speedup (MLP only)
      precompute: true,
      preset: :standard
    ]
  end

  def preset(:full_cpu) do
    # Maximum quality on CPU (no temporal/Mamba for speed)
    # Use: mix run scripts/train_from_replays.exs --preset full_cpu
    [
      epochs: 30,
      max_files: 200,
      batch_size: 64,
      hidden_sizes: [128, 128],
      temporal: false,
      # Full regularization
      augment: true,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      label_smoothing: 0.1,
      # Validation & early stopping
      val_split: 0.1,
      early_stopping: true,
      patience: 7,
      min_delta: 0.005,
      # LR schedule with warmup
      lr_schedule: :cosine,
      learning_rate: 1.0e-4,
      warmup_steps: 500,
      # EMA for better generalization
      ema: true,
      ema_decay: 0.999,
      save_best: true,
      keep_best: 3,
      # Precompute embeddings for 2-3x speedup (MLP only)
      precompute: true,
      preset: :full_cpu
    ]
  end

  # ============================================================================
  # GPU Presets (Requires CUDA/ROCm)
  # ============================================================================

  def preset(:gpu_quick) do
    # Fast GPU test - verify everything works
    # Use: mix run scripts/train_from_replays.exs --preset gpu_quick
    [
      epochs: 3,
      max_files: 20,
      batch_size: 256,
      hidden_sizes: [64, 64],
      temporal: true,
      backbone: :mamba,
      window_size: 30,
      num_layers: 1,
      # Light regularization
      augment: true,
      val_split: 0.1,
      preset: :gpu_quick
    ]
  end

  def preset(:gpu_mlp_quick) do
    # Fast GPU test with MLP + precomputed embeddings (fastest iteration)
    # Use: mix run scripts/train_from_replays.exs --preset gpu_mlp_quick
    # ~2-3x faster than temporal training, good for rapid iteration
    [
      epochs: 5,
      max_files: 50,
      batch_size: 256,
      hidden_sizes: [128, 128],
      temporal: false,
      # Precompute embeddings for maximum speed
      precompute: true,
      # Light regularization
      augment: true,
      mirror_prob: 0.5,
      val_split: 0.1,
      # LR schedule
      lr_schedule: :cosine,
      learning_rate: 3.0e-4,
      warmup_steps: 100,
      preset: :gpu_mlp_quick
    ]
  end

  def preset(:gpu_lstm_quick) do
    # GPU test with LSTM backbone
    # Use: mix run scripts/train_from_replays.exs --preset gpu_lstm_quick
    [
      epochs: 3,
      max_files: 30,
      batch_size: 256,
      hidden_sizes: [64, 64],
      temporal: true,
      backbone: :lstm,
      window_size: 30,
      num_layers: 1,
      augment: true,
      val_split: 0.1,
      preset: :gpu_lstm_quick
    ]
  end

  def preset(:gpu_gru_quick) do
    # GPU test with GRU backbone
    # Use: mix run scripts/train_from_replays.exs --preset gpu_gru_quick
    [
      epochs: 3,
      max_files: 30,
      batch_size: 256,
      hidden_sizes: [64, 64],
      temporal: true,
      backbone: :gru,
      window_size: 30,
      num_layers: 1,
      augment: true,
      val_split: 0.1,
      preset: :gpu_gru_quick
    ]
  end

  def preset(:gpu_attention_quick) do
    # GPU test with attention backbone
    # Use: mix run scripts/train_from_replays.exs --preset gpu_attention_quick
    [
      epochs: 3,
      max_files: 30,
      batch_size: 256,
      hidden_sizes: [64, 64],
      temporal: true,
      backbone: :attention,
      window_size: 30,
      num_layers: 1,
      num_heads: 4,
      augment: true,
      val_split: 0.1,
      preset: :gpu_attention_quick
    ]
  end

  def preset(:gpu_standard) do
    # Standard GPU training with all features
    # Use: mix run scripts/train_from_replays.exs --preset gpu_standard
    [
      epochs: 20,
      max_files: 100,
      batch_size: 256,
      hidden_sizes: [128, 128],
      temporal: true,
      backbone: :mamba,
      window_size: 60,
      num_layers: 2,
      # Full regularization
      augment: true,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      label_smoothing: 0.1,
      # Validation & early stopping
      val_split: 0.1,
      early_stopping: true,
      patience: 5,
      # LR schedule
      lr_schedule: :cosine,
      learning_rate: 1.0e-4,
      warmup_steps: 500,
      # EMA
      ema: true,
      ema_decay: 0.999,
      save_best: true,
      keep_best: 5,
      preset: :gpu_standard
    ]
  end

  def preset(:full) do
    # High quality GPU training
    # Use: mix run scripts/train_from_replays.exs --preset full
    [
      epochs: 50,
      max_files: nil,
      batch_size: 256,
      hidden_sizes: [256, 256],
      temporal: true,
      backbone: :mamba,
      window_size: 60,
      num_layers: 2,
      # Full regularization
      augment: true,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      label_smoothing: 0.1,
      # Validation & early stopping
      val_split: 0.1,
      early_stopping: true,
      patience: 7,
      min_delta: 0.005,
      # LR schedule with warmup
      lr_schedule: :cosine,
      learning_rate: 1.0e-4,
      warmup_steps: 1000,
      # EMA for better test-time performance
      ema: true,
      ema_decay: 0.999,
      # Gradient accumulation for larger effective batch
      # effective batch = 512
      accumulation_steps: 2,
      save_best: true,
      keep_best: 5,
      preset: :full
    ]
  end

  def preset(:production) do
    # Maximum quality for deployment
    # Use: mix run scripts/train_from_replays.exs --preset production
    [
      epochs: 100,
      max_files: nil,
      batch_size: 256,
      hidden_sizes: [256, 256],
      temporal: true,
      backbone: :mamba,
      window_size: 60,
      num_layers: 3,
      state_size: 32,
      expand_factor: 2,
      # Full regularization
      augment: true,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      label_smoothing: 0.1,
      # Larger validation for reliable metrics
      val_split: 0.15,
      early_stopping: true,
      patience: 10,
      min_delta: 0.001,
      # Cosine restarts - helps escape local minima
      lr_schedule: :cosine_restarts,
      learning_rate: 1.0e-4,
      warmup_steps: 2000,
      restart_period: 5000,
      restart_mult: 2,
      # EMA with slower decay for more stability
      ema: true,
      ema_decay: 0.9995,
      # Gradient accumulation
      # effective batch = 1024
      accumulation_steps: 4,
      save_best: true,
      keep_best: 10,
      preset: :production
    ]
  end

  # ============================================================================
  # RTX 4090 Optimized Presets (24GB VRAM)
  # ============================================================================
  # These presets are tuned for NVIDIA RTX 4090 with 24GB VRAM.
  # Usage: EXLA_TARGET=cuda mix run scripts/train_from_replays.exs --preset rtx4090_quick

  def preset(:rtx4090_quick) do
    # Fast test on 4090 - larger batches than generic gpu_quick
    # ~5 minutes, good for verifying GPU setup works
    [
      epochs: 3,
      max_files: 30,
      # 4090 can handle larger batches
      batch_size: 512,
      hidden_sizes: [128, 128],
      temporal: true,
      backbone: :mamba,
      window_size: 30,
      num_layers: 1,
      # Light regularization
      augment: true,
      val_split: 0.1,
      preset: :rtx4090_quick
    ]
  end

  def preset(:rtx4090_standard) do
    # Standard training on 4090 - ~30 minutes to 1 hour
    [
      epochs: 20,
      max_files: 200,
      # Larger batch for faster training
      batch_size: 512,
      hidden_sizes: [256, 256],
      temporal: true,
      backbone: :mamba,
      window_size: 60,
      num_layers: 2,
      state_size: 16,
      # Full regularization
      augment: true,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      label_smoothing: 0.1,
      # Validation & early stopping
      val_split: 0.1,
      early_stopping: true,
      patience: 5,
      # LR schedule
      lr_schedule: :cosine,
      learning_rate: 1.0e-4,
      warmup_steps: 500,
      # EMA
      ema: true,
      ema_decay: 0.999,
      save_best: true,
      keep_best: 5,
      preset: :rtx4090_standard
    ]
  end

  def preset(:rtx4090_full) do
    # Maximum quality on 4090 - several hours
    # Uses gradient accumulation for effective batch size of 2048
    [
      epochs: 50,
      # All available files
      max_files: nil,
      batch_size: 512,
      hidden_sizes: [256, 256],
      temporal: true,
      backbone: :mamba,
      window_size: 60,
      num_layers: 3,
      state_size: 32,
      expand_factor: 2,
      # Full regularization
      augment: true,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      label_smoothing: 0.1,
      # Larger validation
      val_split: 0.15,
      early_stopping: true,
      patience: 10,
      min_delta: 0.001,
      # Cosine restarts
      lr_schedule: :cosine_restarts,
      learning_rate: 1.0e-4,
      warmup_steps: 1000,
      restart_period: 5000,
      restart_mult: 2,
      # EMA with slower decay
      ema: true,
      ema_decay: 0.9995,
      # Gradient accumulation for larger effective batch
      # effective batch = 2048
      accumulation_steps: 4,
      save_best: true,
      keep_best: 10,
      preset: :rtx4090_full
    ]
  end

  # ============================================================================
  # Character-Specific Presets (Built on :production)
  # ============================================================================

  def preset(:mewtwo) do
    # Mewtwo: Long context for teleport recovery timing
    # Teleport takes ~40 frames, need to track full recovery sequences
    Keyword.merge(preset(:production),
      character: :mewtwo,
      window_size: 90,
      preset: :mewtwo
    )
  end

  def preset(:ganondorf) do
    # Ganondorf: Standard context, spacing-focused
    # Slower character benefits from prediction over reaction
    Keyword.merge(preset(:production),
      character: :ganondorf,
      window_size: 60,
      preset: :ganondorf
    )
  end

  def preset(:link) do
    # Link: Extended context for projectile tracking
    # Boomerang return timing, bomb trajectories
    Keyword.merge(preset(:production),
      character: :link,
      window_size: 75,
      preset: :link
    )
  end

  def preset(:gameandwatch) do
    # Game & Watch: Shorter context for unique timing
    # Only fair/dair have L-cancel, bucket/hammer RNG
    Keyword.merge(preset(:production),
      character: :gameandwatch,
      window_size: 45,
      preset: :gameandwatch
    )
  end

  def preset(:zelda) do
    # Zelda: Standard context for transform mechanics
    # Focus on spacing with kicks, transform rarely needed in training
    Keyword.merge(preset(:production),
      character: :zelda,
      window_size: 60,
      preset: :zelda
    )
  end

  def preset(name) when is_binary(name) do
    preset(String.to_atom(name))
  end

  def preset(invalid) do
    raise ArgumentError, """
    Unknown preset: #{inspect(invalid)}

    Available presets:
      CPU:       quick, standard, full_cpu
      GPU:       gpu_quick, gpu_mlp_quick, gpu_standard, full, production
      RTX 4090:  rtx4090_quick, rtx4090_standard, rtx4090_full
      Character: mewtwo, ganondorf, link, gameandwatch, zelda

    Recommended progression:
      1. Test code changes:  --preset quick
      2. Validate on GPU:    --preset gpu_quick (or rtx4090_quick for 4090)
      3. Standard training:  --preset gpu_standard (or rtx4090_standard)
      4. Full quality:       --preset full (or rtx4090_full)
      5. Production deploy:  --preset production (or character preset)

    GPU training requires EXLA_TARGET=cuda:
      EXLA_TARGET=cuda mix run scripts/train_from_replays.exs --preset rtx4090_quick

    Or use the convenience script:
      ./scripts/gpu_train.sh --preset rtx4090_quick --replays ./replays
    """
  end

  # ============================================================================
  # Config Diff Display
  # ============================================================================

  @doc """
  Get a list of options that differ from defaults.

  Useful for displaying at training start to verify configuration.
  Returns a list of `{key, current_value, default_value}` tuples.

  ## Options

  - `:skip` - List of keys to skip (default: [:replays, :checkpoint, :name, :wandb_name])
  - `:include_nil` - Include keys where current is nil but default is not (default: false)

  ## Examples

      iex> Config.diff_from_defaults(epochs: 20, batch_size: 64)
      [{:epochs, 20, 10}]

      iex> Config.diff_from_defaults(defaults())
      []
  """
  @spec diff_from_defaults(keyword(), keyword()) :: [{atom(), any(), any()}]
  def diff_from_defaults(opts, diff_opts \\ []) do
    defaults = defaults()

    skip_keys =
      Keyword.get(diff_opts, :skip, [:replays, :checkpoint, :name, :wandb_name, :wandb_project])

    include_nil = Keyword.get(diff_opts, :include_nil, false)

    opts
    |> Enum.filter(fn {key, value} ->
      key not in skip_keys and
        Keyword.has_key?(defaults, key) and
        value != Keyword.get(defaults, key) and
        (include_nil or value != nil)
    end)
    |> Enum.map(fn {key, value} ->
      {key, value, Keyword.get(defaults, key)}
    end)
    |> Enum.sort_by(fn {key, _, _} -> key end)
  end

  @doc """
  Format config diff as a human-readable string.

  Returns a string showing changed settings, or nil if no changes.

  ## Examples

      iex> Config.format_diff(epochs: 20, batch_size: 128)
      "  epochs: 20 (default: 10)\\n  batch_size: 128 (default: 64)"
  """
  @spec format_diff(keyword()) :: String.t() | nil
  def format_diff(opts) do
    diff = diff_from_defaults(opts)

    if Enum.empty?(diff) do
      nil
    else
      diff
      |> Enum.map(fn {key, current, default} ->
        "  #{key}: #{format_value(current)} (default: #{format_value(default)})"
      end)
      |> Enum.join("\n")
    end
  end

  defp format_value(value) when is_list(value), do: inspect(value, charlists: :as_lists)
  defp format_value(value) when is_atom(value), do: ":#{value}"

  defp format_value(value) when is_float(value),
    do: :erlang.float_to_binary(value, [:compact, decimals: 6])

  defp format_value(nil), do: "nil"
  defp format_value(value), do: "#{value}"

  # ============================================================================
  # Validation
  # ============================================================================

  @valid_backbones [:lstm, :gru, :mamba, :attention, :sliding_window, :lstm_hybrid, :jamba]
  @valid_optimizers [:adam, :adamw, :lamb, :radam, :sgd, :rmsprop]

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
    |> validate_non_negative(opts, :frame_delay_min)
    |> validate_non_negative(opts, :frame_delay_max)
    |> validate_frame_delay_range(opts)
    |> validate_hidden_sizes(opts)
    |> validate_temporal_backbone(opts)
    |> validate_precision(opts)
    |> validate_optimizer(opts)
    |> validate_replays_dir(opts)
    |> validate_positive(opts, :patience)
    |> validate_positive_float(opts, :min_delta)
    |> validate_positive_float(opts, :learning_rate)
    |> validate_lr_schedule(opts)
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

  defp validate_non_negative_float(errors, opts, key) do
    value = opts[key]

    cond do
      is_nil(value) -> errors
      is_number(value) and value >= 0 -> errors
      is_number(value) -> ["#{key} must be non-negative, got: #{value}" | errors]
      true -> ["#{key} must be a non-negative number, got: #{inspect(value)}" | errors]
    end
  end

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
    max_delay = opts[:frame_delay_max] || 18

    cond do
      min_delay > max_delay ->
        [
          "frame_delay_min (#{min_delay}) cannot be greater than frame_delay_max (#{max_delay})"
          | errors
        ]

      max_delay > 60 ->
        ["frame_delay_max > 60 is unusually high (online play is typically 18 frames)" | errors]

      true ->
        errors
    end
  end

  defp validate_temporal_backbone(errors, opts) do
    if opts[:temporal] do
      backbone = opts[:backbone]

      if backbone not in @valid_backbones do
        msg =
          "temporal training requires backbone in #{inspect(@valid_backbones)}, got: #{inspect(backbone)}"

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

  defp validate_optimizer(errors, opts) do
    case opts[:optimizer] do
      o when o in @valid_optimizers ->
        errors

      nil ->
        errors

      other ->
        [
          "optimizer must be one of #{inspect(@valid_optimizers)}, got: #{inspect(other)}"
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

  @valid_lr_schedules [:constant, :cosine, :cosine_restarts, :exponential, :linear]

  defp validate_lr_schedule(errors, opts) do
    schedule = opts[:lr_schedule]

    if schedule && schedule not in @valid_lr_schedules do
      msg =
        "lr_schedule must be one of #{inspect(@valid_lr_schedules)}, got: #{inspect(schedule)}"

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

  # Warning collectors
  defp warn_large_window_size(warnings, opts) do
    if opts[:window_size] && opts[:window_size] > 120 do
      msg = "window_size #{opts[:window_size]} > 120 may cause memory issues"
      [Help.warning_with_help(msg, :window_size) | warnings]
    else
      warnings
    end
  end

  defp warn_large_batch_size(warnings, opts) do
    if opts[:batch_size] && opts[:batch_size] > 256 do
      msg = "batch_size #{opts[:batch_size]} > 256 may cause memory issues on CPU"
      [Help.warning_with_help(msg, :batch_size) | warnings]
    else
      warnings
    end
  end

  defp warn_many_epochs_without_wandb(warnings, opts) do
    epochs = opts[:epochs] || 0
    wandb = opts[:wandb] || false

    if epochs >= 20 and not wandb do
      msg = "training #{epochs} epochs without --wandb; consider enabling for metrics tracking"
      [Help.warning_with_help(msg, :wandb) | warnings]
    else
      warnings
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

      iex> Config.parse_args(["--epochs", "5", "--temporal"])
      [epochs: 5, temporal: true, ...]

      iex> Config.parse_args(["--preset", "quick"])
      [epochs: 1, max_files: 5, hidden_sizes: [32, 32], ...]

      iex> Config.parse_args(["--preset", "quick", "--epochs", "3"])
      [epochs: 3, max_files: 5, hidden_sizes: [32, 32], ...]  # epochs overridden

  """
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
    |> parse_atom_arg(args, "--train-character", :train_character)
    |> parse_flag(args, "--dual-port", :dual_port)
    |> parse_flag(args, "--balance-characters", :balance_characters)
    |> parse_flag(args, "--wandb", :wandb)
    |> parse_string_arg(args, "--wandb-project", :wandb_project)
    |> parse_string_arg(args, "--wandb-name", :wandb_name)
    |> parse_flag(args, "--temporal", :temporal)
    |> parse_atom_arg(args, "--backbone", :backbone)
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
    |> parse_atom_arg(args, "--lr-schedule", :lr_schedule)
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
    |> parse_flag(args, "--focal-loss", :focal_loss)
    |> parse_float_arg(args, "--focal-gamma", :focal_gamma)
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
    |> parse_atom_arg(args, "--optimizer", :optimizer)
    |> parse_flag(args, "--dry-run", :dry_run)
    |> parse_atom_list_arg(args, "--character", :characters)
    |> parse_atom_list_arg(args, "--characters", :characters)
    |> parse_atom_list_arg(args, "--stage", :stages)
    |> parse_atom_list_arg(args, "--stages", :stages)
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
  defp parse_stage_mode_arg(opts, args) do
    case get_arg_value(args, "--stage-mode") do
      nil ->
        opts

      value ->
        mode =
          case String.downcase(value) do
            "full" -> :one_hot_full
            "one_hot_full" -> :one_hot_full
            "compact" -> :one_hot_compact
            "one_hot_compact" -> :one_hot_compact
            "learned" -> :learned
            other -> String.to_atom(other)
          end

        Keyword.put(opts, :stage_mode, mode)
    end
  end

  # Parse action mode: one_hot or learned
  defp parse_action_mode_arg(opts, args) do
    case get_arg_value(args, "--action-mode") do
      nil ->
        opts

      value ->
        mode =
          case String.downcase(value) do
            "one_hot" -> :one_hot
            "onehot" -> :one_hot
            "learned" -> :learned
            other -> String.to_atom(other)
          end

        Keyword.put(opts, :action_mode, mode)
    end
  end

  # Parse character mode: one_hot or learned
  defp parse_character_mode_arg(opts, args) do
    case get_arg_value(args, "--character-mode") do
      nil ->
        opts

      value ->
        mode =
          case String.downcase(value) do
            "one_hot" -> :one_hot
            "onehot" -> :one_hot
            "learned" -> :learned
            other -> String.to_atom(other)
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
        mode =
          case String.downcase(value) do
            "compact" -> :compact
            "enhanced" -> :enhanced
            "full" -> :full
            other -> String.to_atom(other)
          end

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
  defp parse_atom_list_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil ->
        opts

      value ->
        atoms =
          value
          |> String.split(",")
          |> Enum.map(&String.trim/1)
          |> Enum.map(&String.to_atom/1)

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

      checkpoint_name =
        if character do
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
      dropout: opts[:dropout],

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
      focal_loss: opts[:focal_loss],
      focal_gamma: opts[:focal_gamma],
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
  """
  def valid_flags, do: @valid_flags

  @doc """
  Validate command-line arguments for unrecognized flags.

  Returns `{:ok, []}` if all flags are valid, or `{:ok, warnings}` with a list
  of warning messages for unrecognized flags with suggestions.

  ## Examples

      iex> Config.validate_args(["--epochs", "10", "--batch-size", "32"])
      {:ok, []}

      iex> Config.validate_args(["--ephocs", "10"])
      {:ok, ["Unknown flag '--ephocs'. Did you mean '--epochs'?"]}

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

      iex> Config.validate_args!(["--epochs", "10"])
      :ok

      iex> Config.validate_args!(["--ephocs", "10"])
      # Prints: "⚠️  Unknown flag '--ephocs'. Did you mean '--epochs'?"
      :ok

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

  @doc """
  Check if a checkpoint path would overwrite an existing file.

  Returns `{:ok, :new}` if path doesn't exist,
  `{:ok, :overwrite, info}` if exists and overwrite allowed,
  `{:error, :exists, info}` if exists and overwrite not allowed.

  The `info` map contains file metadata for warning display.
  """
  @spec check_checkpoint_path(Path.t(), keyword()) ::
          {:ok, :new} | {:ok, :overwrite, map()} | {:error, :exists, map()}
  def check_checkpoint_path(path, opts \\ []) do
    overwrite = Keyword.get(opts, :overwrite, false)

    case File.stat(path) do
      {:error, :enoent} ->
        {:ok, :new}

      {:ok, stat} ->
        info = %{
          path: path,
          size: stat.size,
          modified: stat.mtime
        }

        if overwrite do
          {:ok, :overwrite, info}
        else
          {:error, :exists, info}
        end
    end
  end

  @doc """
  Format file info for display in collision warnings.

  ## Example

      iex> format_file_info(%{path: "model.axon", size: 45_200_000, modified: {{2026, 1, 23}, {14, 30, 0}}})
      "Size: 45.2 MB, Modified: 2026-01-23 14:30:00"
  """
  @spec format_file_info(map()) :: String.t()
  def format_file_info(info) do
    size_str = format_bytes(info.size)
    time_str = format_datetime(info.modified)
    "Size: #{size_str}, Modified: #{time_str}"
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"

  defp format_bytes(bytes) when bytes < 1024 * 1024 * 1024,
    do: "#{Float.round(bytes / (1024 * 1024), 1)} MB"

  defp format_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024 * 1024), 2)} GB"

  defp format_datetime({{y, m, d}, {h, min, s}}) do
    "#{y}-#{pad(m)}-#{pad(d)} #{pad(h)}:#{pad(min)}:#{pad(s)}"
  end

  defp pad(n) when n < 10, do: "0#{n}"
  defp pad(n), do: "#{n}"

  @doc """
  Backup an existing checkpoint before overwriting.

  Creates backups with rotation: file.bak, file.bak.1, file.bak.2, etc.
  Keeps at most `backup_count` versions.

  ## Options
    - `:backup_count` - Maximum number of backups to keep (default: 3)

  Returns `{:ok, backup_path}` on success, `{:error, reason}` on failure.
  """
  @spec backup_checkpoint(Path.t(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  def backup_checkpoint(path, opts \\ []) do
    backup_count = Keyword.get(opts, :backup_count, 3)

    if File.exists?(path) do
      # Rotate existing backups
      rotate_backups(path, backup_count)

      # Create new backup
      backup_path = "#{path}.bak"

      case File.copy(path, backup_path) do
        {:ok, _} -> {:ok, backup_path}
        {:error, reason} -> {:error, reason}
      end
    else
      # Nothing to backup
      {:ok, nil}
    end
  end

  defp rotate_backups(path, count) when count > 0 do
    # Delete the oldest backup if it exists
    oldest = "#{path}.bak.#{count - 1}"
    File.rm(oldest)

    # Rotate .bak.N -> .bak.N+1 (from highest to lowest)
    Enum.each((count - 2)..0, fn n ->
      src = if n == 0, do: "#{path}.bak", else: "#{path}.bak.#{n}"
      dst = "#{path}.bak.#{n + 1}"
      if File.exists?(src), do: File.rename(src, dst)
    end)
  end

  defp rotate_backups(_path, _count), do: :ok

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
