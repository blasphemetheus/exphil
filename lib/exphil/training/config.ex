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
  alias ExPhil.Training.Config.Parser
  alias ExPhil.Training.Config.Presets
  alias ExPhil.Training.Config.Validator
  alias ExPhil.Training.Config.Yaml

  # Default replays directory - relative path for portability
  # Can be overridden with --replays or --replay-dir
  @default_replays_dir "./replays"
  @default_hidden_sizes [512, 256]

  # Mode allowlists for safe atom conversion (used by YAML module)
  @valid_precision_modes [:f32, :bf16, :f16]

  # Training option allowlists (used by Parser and YAML modules)
  # Note: :hybrid is an alias for :lstm_hybrid (kept for backwards compatibility)
  @valid_backbones [:lstm, :gru, :mamba, :attention, :sliding_window, :lstm_hybrid, :hybrid, :jamba]
  @valid_optimizers [:adam, :adamw, :lamb, :radam, :sgd, :rmsprop]
  @valid_lr_schedules [:constant, :cosine, :cosine_restarts, :exponential, :linear]
  # Policy types: how actions are predicted
  # - :autoregressive - Standard 6-head sequential prediction (current default)
  # - :diffusion - DDPM-based iterative denoising (slow but high quality)
  # - :act - Action Chunking with Transformers (fast, predicts sequences)
  # - :flow_matching - ODE-based continuous normalizing flow (fast, simpler than diffusion)
  @valid_policy_types [:autoregressive, :diffusion, :act, :flow_matching]

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
    # Policy type options
    "--policy-type",
    # Action horizon for ACT and generative policies
    "--action-horizon",
    # Number of diffusion/flow steps for inference
    "--num-inference-steps",
    # KL weight for ACT (CVAE regularization)
    "--kl-weight",
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
      # Policy type: :autoregressive, :diffusion, :act, :flow_matching
      policy_type: :autoregressive,
      # Action horizon for ACT/generative policies (frames to predict at once)
      action_horizon: 8,
      # Number of steps for diffusion/flow inference (more = higher quality, slower)
      num_inference_steps: 20,
      # KL weight (β) for ACT CVAE regularization
      kl_weight: 10.0,
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

  # Delegated to ExPhil.Training.Config.Yaml
  # See that module for implementation details

  @doc """
  Load training configuration from a YAML file.
  Delegates to `ExPhil.Training.Config.Yaml.load/2`.
  """
  @spec load_yaml(String.t()) :: {:ok, keyword()} | {:error, atom() | String.t()}
  def load_yaml(path) do
    Yaml.load(path, yaml_context())
  end

  @doc """
  Load and merge YAML config with CLI args.
  CLI args take precedence over YAML config.
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
  Delegates to `ExPhil.Training.Config.Yaml.parse/2`.
  """
  @spec parse_yaml(String.t()) :: {:ok, keyword()} | {:error, any()}
  def parse_yaml(content) do
    Yaml.parse(content, yaml_context())
  end

  @doc """
  Save current configuration to a YAML file.
  Delegates to `ExPhil.Training.Config.Yaml.save/2`.
  """
  @spec save_yaml(keyword(), String.t()) :: :ok | {:error, any()}
  def save_yaml(opts, path) do
    Yaml.save(opts, path)
  end

  # Build context for YAML parsing with allowlists
  defp yaml_context do
    %{
      valid_backbones: @valid_backbones,
      valid_optimizers: @valid_optimizers,
      valid_lr_schedules: @valid_lr_schedules,
      valid_precision_modes: @valid_precision_modes,
      valid_presets: Presets.valid_presets(),
      valid_characters: Map.keys(@character_map),
      valid_stages: Map.keys(@stage_map)
    }
  end

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
      valid_lr_schedules: @valid_lr_schedules,
      valid_policy_types: @valid_policy_types
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
    case Parser.get_arg_value(args, "--preset") do
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
    |> maybe_add_override(args, "--hidden-sizes", :hidden_sizes, &Parser.parse_hidden_sizes/1)
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
    case Parser.get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, parser.(value))
    end
  end

  defp maybe_add_flag_override(opts, args, flag, key) do
    if Parser.has_flag?(args, flag) do
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
      if Parser.has_flag_value?(args, "--config") do
        config_path = Parser.get_arg_value(args, "--config")

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
    if Parser.has_flag_value?(args, "--preset") do
      apply_preset(base_opts, args)
    else
      # No preset - standard parsing flow
      Parser.parse(args, base_opts, parser_context())
    end
  end

  # Build context for argument parsing with allowlists
  defp parser_context do
    %{
      valid_backbones: @valid_backbones,
      valid_optimizers: @valid_optimizers,
      valid_lr_schedules: @valid_lr_schedules,
      valid_characters: Map.keys(@character_map),
      valid_stages: Map.keys(@stage_map),
      valid_flags: @valid_flags
    }
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
  List of valid policy types.

  ## Examples

      iex> types = ExPhil.Training.Config.valid_policy_types()
      iex> :autoregressive in types
      true
      iex> :diffusion in types
      true

  """
  @spec valid_policy_types() :: [atom()]
  def valid_policy_types, do: @valid_policy_types

  @doc """
  Validate command-line arguments for unrecognized flags.
  Delegates to `ExPhil.Training.Config.Parser.validate_args/2`.
  """
  @spec validate_args(list(String.t())) :: {:ok, list(String.t())}
  def validate_args(args) when is_list(args) do
    Parser.validate_args(args, @valid_flags)
  end

  @doc """
  Validate args and print warnings if any.
  Delegates to `ExPhil.Training.Config.Parser.validate_args!/2`.
  """
  @spec validate_args!(list(String.t())) :: :ok
  def validate_args!(args) do
    Parser.validate_args!(args, @valid_flags)
  end

  @doc """
  Parse hidden sizes string into list of integers.
  Delegates to `ExPhil.Training.Config.Parser.parse_hidden_sizes/1`.
  """
  @spec parse_hidden_sizes(String.t()) :: [integer()]
  defdelegate parse_hidden_sizes(str), to: Parser

  # =============================================================================
  # Checkpoint Naming and Path Utilities
  # =============================================================================

  @doc """
  Generate a checkpoint name with memorable naming if not already specified.

  Format: `checkpoints/{character_}{backbone}_{name}_{timestamp}.axon`
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
  Derive the policy path from a checkpoint path.
  """
  def derive_policy_path(nil), do: nil

  def derive_policy_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_policy.bin")
  end

  @doc """
  Derive the config JSON path from a checkpoint path.
  """
  def derive_config_path(nil), do: nil

  def derive_config_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_config.json")
  end

  @doc """
  Derive the best checkpoint path from a checkpoint path.
  """
  def derive_best_checkpoint_path(nil), do: nil

  def derive_best_checkpoint_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_best.axon")
  end

  @doc """
  Derive the best policy path from a checkpoint path.
  """
  def derive_best_policy_path(nil), do: nil

  def derive_best_policy_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_best_policy.bin")
  end

  @doc """
  Compute a SHA256 hash of a list of file paths for replay manifest.
  """
  @spec compute_manifest_hash([String.t()]) :: String.t() | nil
  def compute_manifest_hash([]), do: nil

  def compute_manifest_hash(paths) when is_list(paths) do
    paths
    |> Enum.sort()
    |> Enum.join("\n")
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> then(&"sha256:#{&1}")
  end

  @doc """
  Build the training config map that gets saved as JSON alongside the model.
  """
  def build_config_json(opts, results \\ %{}) do
    %{
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      replays_dir: opts[:replays],
      max_files: opts[:max_files],
      player_port: opts[:player_port],
      characters: format_atom_list(opts[:characters]),
      stages: format_atom_list(opts[:stages]),
      replay_count: results[:replay_count],
      replay_files: results[:replay_files],
      replay_manifest_hash: results[:replay_manifest_hash],
      character_distribution: results[:character_distribution],
      temporal: opts[:temporal],
      backbone: if(opts[:temporal], do: to_string(opts[:backbone]), else: "mlp"),
      policy_type: to_string(opts[:policy_type] || :autoregressive),
      action_horizon: opts[:action_horizon],
      num_inference_steps: opts[:num_inference_steps],
      kl_weight: opts[:kl_weight],
      hidden_sizes: opts[:hidden_sizes],
      embed_size: results[:embed_size],
      layer_norm: opts[:layer_norm],
      residual: opts[:residual],
      kmeans_centers: opts[:kmeans_centers],
      stage_mode: get_embedding_mode(opts, :stage_mode),
      action_mode: get_embedding_mode(opts, :action_mode),
      character_mode: get_embedding_mode(opts, :character_mode),
      nana_mode: get_embedding_mode(opts, :nana_mode),
      jumps_normalized: Keyword.get(opts, :jumps_normalized, defaults()[:jumps_normalized]),
      window_size: opts[:window_size],
      stride: opts[:stride],
      num_layers: opts[:num_layers],
      truncate_bptt: opts[:truncate_bptt],
      state_size: opts[:state_size],
      expand_factor: opts[:expand_factor],
      conv_size: opts[:conv_size],
      attention_every: opts[:attention_every],
      num_heads: opts[:num_heads],
      head_dim: opts[:head_dim],
      epochs: opts[:epochs],
      batch_size: opts[:batch_size],
      precision: to_string(opts[:precision]),
      frame_delay: opts[:frame_delay],
      learning_rate: opts[:lr],
      lr_schedule: opts[:lr_schedule] && to_string(opts[:lr_schedule]),
      warmup_steps: opts[:warmup_steps],
      optimizer: opts[:optimizer] && to_string(opts[:optimizer]),
      max_grad_norm: opts[:max_grad_norm],
      accumulation_steps: opts[:accumulation_steps],
      label_smoothing: opts[:label_smoothing],
      dropout: opts[:dropout],
      focal_loss: opts[:focal_loss],
      focal_gamma: opts[:focal_gamma],
      button_weight: opts[:button_weight],
      stick_edge_weight: opts[:stick_edge_weight],
      ema: opts[:ema],
      ema_decay: opts[:ema_decay],
      train_character: opts[:train_character] && to_string(opts[:train_character]),
      augment: opts[:augment],
      val_split: opts[:val_split],
      seed: opts[:seed],
      early_stopping: opts[:early_stopping],
      patience: opts[:patience],
      min_delta: opts[:min_delta],
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

  defp format_atom_list(nil), do: nil
  defp format_atom_list([]), do: nil

  defp format_atom_list(atoms) when is_list(atoms) do
    Enum.map(atoms, &to_string/1)
  end

  defp get_embedding_mode(opts, key) do
    value = Keyword.get(opts, key, defaults()[key])

    case value do
      nil -> nil
      atom when is_atom(atom) -> to_string(atom)
      other -> other
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
