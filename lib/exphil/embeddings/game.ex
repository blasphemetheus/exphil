defmodule ExPhil.Embeddings.Game do
  @moduledoc """
  Full game state embedding - the complete input to the neural network.

  Combines:
  - Player 0 (self) embedding
  - Player 1 (opponent) embedding
  - Stage embedding
  - Previous action embedding
  - Optional: Projectiles, items, stage-specific data

  ## Input Format

  The network receives a "StateAction" tuple containing:
  - Current game state (players, stage)
  - Previous action taken (for autoregressive conditioning)
  - Player name/tag (for learning player-specific styles)

  ## Ego-Centric View

  The embedding always puts the agent's own player first, regardless
  of which port they're actually on. This simplifies learning since
  the network always sees a consistent "self vs opponent" structure.
  """

  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Embeddings.Player, as: PlayerEmbed
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Bridge.{GameState, Projectile, ControllerState}

  # ============================================================================
  # Configuration
  # ============================================================================

  @doc """
  Configuration for game state embedding.
  """
  defstruct [
    player: %PlayerEmbed{},
    controller: %ControllerEmbed{},
    with_projectiles: false,
    max_projectiles: 5,
    with_items: false,
    max_items: 5,
    num_player_names: 128  # For learning player styles
  ]

  @type config :: %__MODULE__{
    player: PlayerEmbed.config(),
    controller: ControllerEmbed.config(),
    with_projectiles: boolean(),
    max_projectiles: non_neg_integer(),
    with_items: boolean(),
    max_items: non_neg_integer(),
    num_player_names: non_neg_integer()
  }

  @spec default_config() :: config()
  def default_config do
    %__MODULE__{
      player: PlayerEmbed.default_config(),
      controller: ControllerEmbed.default_config()
    }
  end

  # ============================================================================
  # Embedding Size Calculation
  # ============================================================================

  @doc """
  Calculate the total embedding size for a given configuration.
  """
  @spec embedding_size(config()) :: non_neg_integer()
  def embedding_size(config \\ default_config()) do
    player_size = PlayerEmbed.embedding_size(config.player)

    # Two players
    players_size = 2 * player_size

    # Stage
    stage_size = Primitives.embedding_size(:stage)

    # Previous action (continuous)
    prev_action_size = ControllerEmbed.continuous_embedding_size()

    # Player name (one-hot)
    name_size = config.num_player_names

    # Projectiles
    projectile_size = if config.with_projectiles do
      config.max_projectiles * projectile_embedding_size()
    else
      0
    end

    players_size + stage_size + prev_action_size + name_size + projectile_size
  end

  defp projectile_embedding_size do
    1 +  # exists
    1 +  # owner
    2 +  # x, y
    1 +  # type (simplified)
    2    # speed_x, speed_y
  end

  # ============================================================================
  # Main Embedding Functions
  # ============================================================================

  @doc """
  Embed a complete game state for the neural network.

  ## Parameters
    - `game_state` - Current game state from the bridge
    - `prev_action` - Previous controller state (for autoregressive)
    - `own_port` - Which port the agent is playing on (1 or 2)
    - `opts` - Options including `:config` and `:name_id`

  ## Returns
    A single Nx tensor containing the full embedded state.
  """
  @spec embed(GameState.t(), ControllerState.t() | nil, integer(), keyword()) :: Nx.Tensor.t()
  def embed(game_state, prev_action, own_port \\ 1, opts \\ []) do
    config = Keyword.get(opts, :config, default_config())
    name_id = Keyword.get(opts, :name_id, 0)

    # Get players in ego-centric order (own first, opponent second)
    {own, opponent} = get_players_ego(game_state, own_port)

    embeddings = [
      # Players (ego-centric)
      PlayerEmbed.embed(own, config.player),
      PlayerEmbed.embed(opponent, config.player),

      # Stage
      Primitives.stage_embed(game_state.stage || 0),

      # Previous action
      ControllerEmbed.embed_continuous(prev_action),

      # Player name/tag ID (for style learning)
      Primitives.one_hot(name_id, size: config.num_player_names, clamp: true)
    ]

    # Optional: projectiles
    embeddings = if config.with_projectiles do
      [embed_projectiles(game_state.projectiles, config) | embeddings]
    else
      embeddings
    end

    Nx.concatenate(Enum.reverse(embeddings))
  end

  @doc """
  Embed just the state portion (no previous action).

  Useful for value function that doesn't need action conditioning.
  """
  @spec embed_state(GameState.t(), integer(), keyword()) :: Nx.Tensor.t()
  def embed_state(game_state, own_port \\ 1, opts \\ []) do
    config = Keyword.get(opts, :config, default_config())

    {own, opponent} = get_players_ego(game_state, own_port)

    Nx.concatenate([
      PlayerEmbed.embed(own, config.player),
      PlayerEmbed.embed(opponent, config.player),
      Primitives.stage_embed(game_state.stage || 0)
    ])
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp get_players_ego(game_state, own_port) do
    own = GameState.get_player(game_state, own_port)

    opponent_port = if own_port == 1, do: 2, else: 1
    opponent = GameState.get_player(game_state, opponent_port)

    {own, opponent}
  end

  defp embed_projectiles(nil, config) do
    Nx.broadcast(0.0, {config.max_projectiles * projectile_embedding_size()})
  end

  defp embed_projectiles(projectiles, config) when is_list(projectiles) do
    # Pad or truncate to max_projectiles
    projectiles = Enum.take(projectiles, config.max_projectiles)
    num_existing = length(projectiles)
    padding_count = config.max_projectiles - num_existing

    embedded = Enum.map(projectiles, &embed_single_projectile/1)

    padding = if padding_count > 0 do
      [Nx.broadcast(0.0, {padding_count * projectile_embedding_size()})]
    else
      []
    end

    Nx.concatenate(embedded ++ padding)
  end

  defp embed_single_projectile(%Projectile{} = proj) do
    Nx.concatenate([
      Primitives.bool_embed(true),  # exists
      Primitives.float_embed(proj.owner, scale: 0.5),  # owner (1 or 2)
      Primitives.xy_embed(proj.x),
      Primitives.xy_embed(proj.y),
      Primitives.float_embed(proj.type, scale: 0.01),  # simplified type
      Primitives.speed_embed(proj.speed_x),
      Primitives.speed_embed(proj.speed_y)
    ])
  end

  # ============================================================================
  # Batch Embedding (for training)
  # ============================================================================

  @doc """
  Embed a batch of game states.

  Takes a list of {game_state, prev_action, own_port} tuples and returns
  a batched tensor of shape [batch_size, embedding_size].
  """
  @spec embed_batch([{GameState.t(), ControllerState.t() | nil, integer()}], keyword()) ::
    Nx.Tensor.t()
  def embed_batch(states, opts \\ []) do
    states
    |> Enum.map(fn {gs, prev_action, own_port} ->
      embed(gs, prev_action, own_port, opts)
    end)
    |> Nx.stack()
  end

  @doc """
  Create a dummy embedding for model initialization.

  Returns a tensor of the correct shape filled with zeros.
  """
  @spec dummy(keyword()) :: Nx.Tensor.t()
  def dummy(opts \\ []) do
    config = Keyword.get(opts, :config, default_config())
    Nx.broadcast(0.0, {embedding_size(config)})
  end

  @doc """
  Create a batched dummy embedding.
  """
  @spec dummy_batch(non_neg_integer(), keyword()) :: Nx.Tensor.t()
  def dummy_batch(batch_size, opts \\ []) do
    config = Keyword.get(opts, :config, default_config())
    Nx.broadcast(0.0, {batch_size, embedding_size(config)})
  end
end
