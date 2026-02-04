defmodule ExPhil.Embeddings.Game.Config do
  @moduledoc """
  Configuration for game state embedding.

  Defines the config struct and helper functions for querying embedding modes
  (learned vs one-hot), computing ID counts, and calculating embedding sizes.

  ## Configuration Options

  - `:player` - Player embedding config (see `ExPhil.Embeddings.Player`)
  - `:controller` - Controller embedding config
  - `:with_projectiles` - Enable projectile tracking (default: true)
  - `:max_projectiles` - Max projectiles to embed (default: 5)
  - `:with_items` - Enable item tracking (default: false)
  - `:max_items` - Max items to embed (default: 5)
  - `:with_distance` - Include player distance feature (default: true)
  - `:with_relative_pos` - Include dx/dy to opponent (default: true)
  - `:with_frame_count` - Include game timer (default: true)
  - `:stage_mode` - `:one_hot_full`, `:one_hot_compact`, or `:learned`

  ## See Also

  - `ExPhil.Embeddings.Game` - Main embedding module
  - `ExPhil.Embeddings.Game.Stage` - Stage embedding details
  """

  alias ExPhil.Embeddings.Player, as: PlayerEmbed
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Embeddings.Game.Stage
  alias ExPhil.Constants

  # Tensor core alignment for GPU efficiency
  @tensor_core_alignment Constants.tensor_alignment()

  # ============================================================================
  # Configuration Struct
  # ============================================================================

  defstruct player: %PlayerEmbed{},
            controller: %ControllerEmbed{},
            # Enabled for Link/Samus/Falco/etc projectile tracking
            with_projectiles: true,
            max_projectiles: 5,
            with_items: false,
            max_items: 5,
            # For learning player styles (reduced for 2048 alignment)
            num_player_names: 112,
            # Game-level spatial features
            # Distance between players (+1 dim)
            with_distance: true,
            # Relative position dx, dy (+2 dims)
            with_relative_pos: true,
            # Game frame/timer (+1 dim)
            with_frame_count: true,
            # Stage embedding mode
            # :one_hot_full (64), :one_hot_compact (7), :learned (ID in network)
            stage_mode: :one_hot_compact

  @type stage_mode :: :one_hot_full | :one_hot_compact | :learned

  @type t :: %__MODULE__{
          player: PlayerEmbed.config(),
          controller: ControllerEmbed.config(),
          with_projectiles: boolean(),
          max_projectiles: non_neg_integer(),
          with_items: boolean(),
          max_items: non_neg_integer(),
          num_player_names: non_neg_integer(),
          with_distance: boolean(),
          with_relative_pos: boolean(),
          with_frame_count: boolean(),
          stage_mode: stage_mode()
        }

  @doc """
  Returns the default game embedding configuration.
  """
  @spec default() :: t()
  def default do
    %__MODULE__{
      player: PlayerEmbed.default_config(),
      controller: ControllerEmbed.default_config()
    }
  end

  # ============================================================================
  # Learned Embedding Predicates
  # ============================================================================

  @doc """
  Check if the config uses learned action embeddings.
  """
  @spec uses_learned_actions?(t()) :: boolean()
  def uses_learned_actions?(config) do
    config.player.action_mode == :learned
  end

  @doc """
  Check if the config uses enhanced Nana mode with learned embeddings.
  """
  @spec uses_enhanced_nana?(t()) :: boolean()
  def uses_enhanced_nana?(config) do
    config.player.nana_mode == :enhanced and config.player.with_nana
  end

  @doc """
  Check if the config uses learned character embeddings.
  """
  @spec uses_learned_characters?(t()) :: boolean()
  def uses_learned_characters?(config) do
    config.player.character_mode == :learned
  end

  @doc """
  Check if the config uses learned stage embeddings.
  """
  @spec uses_learned_stages?(t()) :: boolean()
  def uses_learned_stages?(config) do
    config.stage_mode == :learned
  end

  @doc """
  Check if the config uses learned stage embeddings (alias).
  """
  @spec uses_learned_stage?(t()) :: boolean()
  def uses_learned_stage?(config) do
    config.stage_mode == :learned
  end

  # ============================================================================
  # ID Count Functions
  # ============================================================================

  @doc """
  Number of action IDs appended when using learned embeddings.

  - 2 IDs when using learned player actions only (own + opponent)
  - 4 IDs when also using enhanced Nana mode (own + opponent + own_nana + opponent_nana)
  """
  @spec num_action_ids(t()) :: non_neg_integer()
  def num_action_ids(config) do
    base_ids = if config.player.action_mode == :learned, do: 2, else: 0
    nana_ids = if config.player.nana_mode == :enhanced and config.player.with_nana, do: 2, else: 0
    base_ids + nana_ids
  end

  @doc """
  Number of character IDs appended when using learned character embeddings.

  - 2 IDs when using learned character embeddings (own + opponent)
  """
  @spec num_character_ids(t()) :: non_neg_integer()
  def num_character_ids(config) do
    if config.player.character_mode == :learned, do: 2, else: 0
  end

  @doc """
  Number of stage IDs appended when using learned stage embeddings.

  - 1 ID when using learned stage embedding
  - 0 when using one-hot (full or compact)
  """
  @spec num_stage_ids(t()) :: non_neg_integer()
  def num_stage_ids(config) do
    if config.stage_mode == :learned, do: 1, else: 0
  end

  @doc """
  Total number of IDs appended (action IDs + character IDs + stage IDs).
  """
  @spec num_total_ids(t()) :: non_neg_integer()
  def num_total_ids(config) do
    num_action_ids(config) + num_character_ids(config) + num_stage_ids(config)
  end

  # ============================================================================
  # Embedding Size Calculations
  # ============================================================================

  @doc """
  Calculate the total embedding size for a given configuration.

  Includes padding for tensor core alignment (multiples of 8).
  """
  @spec embedding_size(t()) :: non_neg_integer()
  def embedding_size(config) do
    raw_size = raw_embedding_size(config)
    padding_size = padding_for_alignment(raw_size)
    raw_size + padding_size
  end

  @doc """
  Get the raw embedding size before tensor core alignment padding.

  Useful for debugging or when you need to know the "semantic" size.
  """
  @spec raw_embedding_size(t()) :: non_neg_integer()
  def raw_embedding_size(config) do
    player_size = PlayerEmbed.embedding_size(config.player)

    # Two players
    players_size = 2 * player_size

    # Stage (depends on stage_mode)
    stage_size = Stage.embedding_size(config.stage_mode)

    # Previous action (continuous)
    prev_action_size = ControllerEmbed.continuous_embedding_size()

    # Player name (one-hot)
    name_size = config.num_player_names

    # Projectiles
    projectile_size =
      if config.with_projectiles do
        config.max_projectiles * projectile_embedding_size()
      else
        0
      end

    # Items (Link bombs, etc.)
    item_size =
      if config.with_items do
        config.max_items * item_embedding_size()
      else
        0
      end

    # Game-level spatial features
    distance_size = if config.with_distance, do: 1, else: 0
    relative_pos_size = if config.with_relative_pos, do: 2, else: 0
    frame_count_size = if config.with_frame_count, do: 1, else: 0

    # IDs appended at end when using learned embeddings
    action_ids_size = num_action_ids(config)
    character_ids_size = num_character_ids(config)
    stage_ids_size = num_stage_ids(config)

    players_size + stage_size + prev_action_size + name_size +
      projectile_size + item_size +
      distance_size + relative_pos_size + frame_count_size +
      action_ids_size + character_ids_size + stage_ids_size
  end

  @doc """
  Get the size of the continuous features (everything except IDs).

  When using learned embeddings, the network needs to know where to split:
  - continuous_features = input[:, :-n]
  - ids = input[:, -n:]

  where n = num_total_ids(config)
  """
  @spec continuous_embedding_size(t()) :: non_neg_integer()
  def continuous_embedding_size(config) do
    embedding_size(config) - num_total_ids(config)
  end

  @doc """
  Calculate padding needed to align to tensor core boundary.

  Returns 0 if already aligned, otherwise the number of dims to add.
  """
  @spec padding_for_alignment(non_neg_integer()) :: non_neg_integer()
  def padding_for_alignment(size) do
    remainder = rem(size, @tensor_core_alignment)

    if remainder == 0 do
      0
    else
      @tensor_core_alignment - remainder
    end
  end

  # ============================================================================
  # Component Embedding Sizes (for size calculation)
  # ============================================================================

  @doc false
  def projectile_embedding_size do
    # exists (1) + owner (1) + x,y (2) + type (1) + speed_x,y (2)
    1 + 1 + 2 + 1 + 2
  end

  @doc false
  def item_embedding_size do
    # exists (1) + x,y (2) + category (6) + is_held (1) + held_by_self (1) + timer (1)
    1 + 2 + 6 + 1 + 1 + 1
  end
end
