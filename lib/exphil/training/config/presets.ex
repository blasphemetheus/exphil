defmodule ExPhil.Training.Config.Presets do
  @moduledoc """
  Training presets for different use cases and hardware configurations.

  Presets provide pre-configured training options optimized for specific
  scenarios, from quick iteration during development to production-quality
  training on high-end GPUs.

  ## Preset Categories

  ### CPU Presets (No GPU Required)
  - `:quick` - Fast iteration for testing (1 epoch, 5 files, small MLP)
  - `:standard` - Balanced CPU training (10 epochs, 50 files, augmentation)
  - `:full_cpu` - Maximum CPU quality (30 epochs, 200 files, all regularization)

  ### GPU Presets (Requires CUDA/ROCm)
  - `:gpu_quick` - Fast GPU test (3 epochs, 20 files, Mamba temporal)
  - `:gpu_mlp_quick` - Fastest GPU test (5 epochs, 50 files, MLP + precompute)
  - `:gpu_lstm_quick` - LSTM backbone test (3 epochs, 30 files)
  - `:gpu_gru_quick` - GRU backbone test (3 epochs, 30 files)
  - `:gpu_attention_quick` - Attention backbone test (3 epochs, 30 files)
  - `:gpu_standard` - Standard GPU training (20 epochs, Mamba, all features)
  - `:full` - High quality GPU (50 epochs, Mamba, temporal, full regularization)
  - `:production` - Maximum quality (100 epochs, Mamba, all optimizations, EMA)

  ### RTX 4090 Optimized (24GB VRAM)
  - `:rtx4090_quick` - Fast test on 4090 with larger batches
  - `:rtx4090_standard` - Standard training optimized for 4090
  - `:rtx4090_full` - Maximum quality on 4090

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

  ## See Also

  - `ExPhil.Training.Config` - Main configuration module
  - `ExPhil.Training.Imitation` - Training implementation
  """

  alias ExPhil.Training.Config.AtomSafety

  # All valid preset names
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

  @doc """
  Get the list of all valid preset names.

  ## Examples

      iex> presets = ExPhil.Training.Config.Presets.valid_presets()
      iex> :quick in presets
      true
      iex> :production in presets
      true

  """
  @spec valid_presets() :: [atom()]
  def valid_presets, do: @valid_presets

  @doc """
  Get training options for a preset.

  Returns a keyword list of training options optimized for the specified
  preset. Can be merged with custom options using `Keyword.merge/2`.

  ## Examples

      iex> opts = ExPhil.Training.Config.Presets.get(:quick)
      iex> opts[:epochs]
      1
      iex> opts[:max_files]
      5

      iex> opts = ExPhil.Training.Config.Presets.get(:production)
      iex> opts[:epochs]
      100
      iex> opts[:ema]
      true

      iex> opts = ExPhil.Training.Config.Presets.get(:mewtwo)
      iex> opts[:character]
      :mewtwo
      iex> opts[:window_size]
      90

  """
  @spec get(atom() | String.t()) :: keyword() | no_return()

  # ============================================================================
  # CPU Presets (No GPU Required)
  # ============================================================================

  def get(:quick) do
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

  def get(:standard) do
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

  def get(:full_cpu) do
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

  def get(:gpu_quick) do
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

  def get(:gpu_mlp_quick) do
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

  def get(:gpu_lstm_quick) do
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

  def get(:gpu_gru_quick) do
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

  def get(:gpu_attention_quick) do
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

  def get(:gpu_standard) do
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

  def get(:full) do
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

  def get(:production) do
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

  def get(:rtx4090_quick) do
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

  def get(:rtx4090_standard) do
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

  def get(:rtx4090_full) do
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

  def get(:mewtwo) do
    # Mewtwo: Long context for teleport recovery timing
    # Teleport takes ~40 frames, need to track full recovery sequences
    Keyword.merge(get(:production),
      character: :mewtwo,
      window_size: 90,
      preset: :mewtwo
    )
  end

  def get(:ganondorf) do
    # Ganondorf: Standard context, spacing-focused
    # Slower character benefits from prediction over reaction
    Keyword.merge(get(:production),
      character: :ganondorf,
      window_size: 60,
      preset: :ganondorf
    )
  end

  def get(:link) do
    # Link: Extended context for projectile tracking
    # Boomerang return timing, bomb trajectories
    Keyword.merge(get(:production),
      character: :link,
      window_size: 75,
      preset: :link
    )
  end

  def get(:gameandwatch) do
    # Game & Watch: Shorter context for unique timing
    # Only fair/dair have L-cancel, bucket/hammer RNG
    Keyword.merge(get(:production),
      character: :gameandwatch,
      window_size: 45,
      preset: :gameandwatch
    )
  end

  def get(:zelda) do
    # Zelda: Standard context for transform mechanics
    # Focus on spacing with kicks, transform rarely needed in training
    Keyword.merge(get(:production),
      character: :zelda,
      window_size: 60,
      preset: :zelda
    )
  end

  # String conversion - delegate to atom version with safe conversion
  def get(name) when is_binary(name) do
    get(AtomSafety.safe_to_atom!(name, @valid_presets))
  end

  # Unknown preset - raise helpful error
  def get(invalid) do
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
end
