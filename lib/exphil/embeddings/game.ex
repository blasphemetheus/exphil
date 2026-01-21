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
  alias ExPhil.Bridge.{GameState, Projectile, Item, ControllerState}

  # ============================================================================
  # Configuration
  # ============================================================================

  @doc """
  Configuration for game state embedding.
  """
  defstruct [
    player: %PlayerEmbed{},
    controller: %ControllerEmbed{},
    with_projectiles: true,  # Enabled for Link/Samus/Falco/etc projectile tracking
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

    # Items (Link bombs, etc.)
    item_size = if config.with_items do
      config.max_items * item_embedding_size()
    else
      0
    end

    players_size + stage_size + prev_action_size + name_size + projectile_size + item_size
  end

  defp projectile_embedding_size do
    1 +  # exists
    1 +  # owner
    2 +  # x, y
    1 +  # type (simplified)
    2    # speed_x, speed_y
  end

  defp item_embedding_size do
    1 +  # exists
    2 +  # x, y position
    6 +  # category one-hot (6 categories: none, bomb, melee, ranged, container, other)
    1 +  # is_held (boolean)
    1 +  # held_by_self (boolean - important for Link self-damage)
    1    # timer (normalized)
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

    # Optional: items (Link bombs, etc.)
    embeddings = if config.with_items do
      [embed_items(game_state.items, own_port, config) | embeddings]
    else
      embeddings
    end

    Nx.concatenate(Enum.reverse(embeddings))
  end

  @doc """
  Batch embed multiple game states at once - MUCH faster than calling embed/4 in a loop.

  This is the key optimization for temporal training. Instead of:
  - 415K calls to embed() with 20+ Nx ops each = 8M+ Nx operations

  We do:
  - 1 batch extraction + ~20 batch Nx ops = ~20 operations

  ## Returns
    Tensor of shape [batch_size, embedding_size]
  """
  @spec embed_states_fast([GameState.t()], integer(), keyword()) :: Nx.Tensor.t()
  def embed_states_fast(game_states, own_port \\ 1, opts \\ []) when is_list(game_states) do
    if Enum.empty?(game_states) do
      Nx.broadcast(0.0, {0, embedding_size(opts)})
    else
      config = Keyword.get(opts, :config, default_config())
      name_id = Keyword.get(opts, :name_id, 0)

      # Extract own players and opponent players
      {own_players, opponent_players} = game_states
      |> Enum.map(fn gs -> get_players_ego(gs, own_port) end)
      |> Enum.unzip()

      # Extract stages
      stages = Enum.map(game_states, fn gs -> gs.stage || 0 end)

      # Batch embed players
      own_emb = PlayerEmbed.embed_batch(own_players, config.player)
      opp_emb = PlayerEmbed.embed_batch(opponent_players, config.player)

      # Batch embed stages
      stage_emb = Primitives.batch_one_hot(Nx.tensor(stages, type: :s32), size: 64, clamp: true)

      # Batch embed name_id (same for all in batch)
      batch_size = length(game_states)
      name_ids = List.duplicate(name_id, batch_size)
      name_emb = Primitives.batch_one_hot(Nx.tensor(name_ids, type: :s32), size: config.num_player_names, clamp: true)

      # Previous action (zeros since we don't have it in temporal sequences)
      # continuous_embedding_size = 8 + 2 + 2 + 1 = 13
      prev_action_emb = Nx.broadcast(0.0, {batch_size, ControllerEmbed.continuous_embedding_size()})

      # Concatenate all: [batch, total_embed_size]
      # Order must match Game.embed: players, stage, prev_action, name_id
      Nx.concatenate([own_emb, opp_emb, stage_emb, prev_action_emb, name_emb], axis: 1)
    end
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
  # Item Embedding (Link bombs, etc.)
  # ============================================================================

  defp embed_items(nil, _own_port, config) do
    Nx.broadcast(0.0, {config.max_items * item_embedding_size()})
  end

  defp embed_items(items, own_port, config) when is_list(items) do
    # Pad or truncate to max_items
    items = Enum.take(items, config.max_items)
    num_existing = length(items)
    padding_count = config.max_items - num_existing

    embedded = Enum.map(items, &embed_single_item(&1, own_port))

    padding = if padding_count > 0 do
      [Nx.broadcast(0.0, {padding_count * item_embedding_size()})]
    else
      []
    end

    Nx.concatenate(embedded ++ padding)
  end

  defp embed_single_item(%Item{} = item, own_port) do
    category = Item.item_category(item)
    is_held = Item.held?(item)
    held_by_self = is_held and item.held_by == own_port

    Nx.concatenate([
      Primitives.bool_embed(true),  # exists
      Primitives.xy_embed(item.x),
      Primitives.xy_embed(item.y),
      Primitives.one_hot(category, size: 6, clamp: true),  # category
      Primitives.bool_embed(is_held),
      Primitives.bool_embed(held_by_self),
      Primitives.float_embed(normalize_timer(item.timer))  # normalized timer
    ])
  end

  # Normalize item timer to [0, 1] range
  # Link's bomb timer is typically 0-180 frames (3 seconds)
  defp normalize_timer(nil), do: 0.0
  defp normalize_timer(timer) when is_integer(timer) do
    min(1.0, timer / 180.0)
  end
  defp normalize_timer(_), do: 0.0

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
