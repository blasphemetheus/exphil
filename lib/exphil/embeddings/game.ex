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
    num_player_names: 112,  # For learning player styles (reduced for 2048 alignment)
    # Game-level spatial features
    with_distance: true,  # Distance between players (+1 dim)
    with_relative_pos: true,  # Relative position dx, dy (+2 dims)
    with_frame_count: true  # Game frame/timer (+1 dim)
  ]

  @type config :: %__MODULE__{
    player: PlayerEmbed.config(),
    controller: ControllerEmbed.config(),
    with_projectiles: boolean(),
    max_projectiles: non_neg_integer(),
    with_items: boolean(),
    max_items: non_neg_integer(),
    num_player_names: non_neg_integer(),
    with_distance: boolean(),
    with_relative_pos: boolean(),
    with_frame_count: boolean()
  }

  @doc """
  Number of action IDs appended when using learned embeddings.

  - 2 IDs when using learned player actions only (own + opponent)
  - 4 IDs when also using enhanced Nana mode (own + opponent + own_nana + opponent_nana)
  """
  @spec num_action_ids(config()) :: non_neg_integer()
  def num_action_ids(config \\ default_config()) do
    base_ids = if config.player.action_mode == :learned, do: 2, else: 0
    nana_ids = if config.player.nana_mode == :enhanced and config.player.with_nana, do: 2, else: 0
    base_ids + nana_ids
  end

  @doc """
  Get the size of the continuous features (everything except action IDs).

  When using learned embeddings, the network needs to know where to split:
  - continuous_features = input[:, :-n]
  - action_ids = input[:, -n:]

  where n = num_action_ids(config)

  This function returns the size of continuous_features.
  """
  @spec continuous_embedding_size(config()) :: non_neg_integer()
  def continuous_embedding_size(config \\ default_config()) do
    total = embedding_size(config)
    total - num_action_ids(config)
  end

  @doc """
  Check if the config uses learned action embeddings.
  """
  @spec uses_learned_actions?(config()) :: boolean()
  def uses_learned_actions?(config) do
    config.player.action_mode == :learned
  end

  @doc """
  Check if the config uses enhanced Nana mode with learned embeddings.
  """
  @spec uses_enhanced_nana?(config()) :: boolean()
  def uses_enhanced_nana?(config) do
    config.player.nana_mode == :enhanced and config.player.with_nana
  end

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

    # Game-level spatial features
    distance_size = if config.with_distance, do: 1, else: 0
    relative_pos_size = if config.with_relative_pos, do: 2, else: 0
    frame_count_size = if config.with_frame_count, do: 1, else: 0

    # Action IDs appended at end when using learned embeddings
    # Includes: player actions (2) + Nana actions if enhanced mode (2)
    action_ids_size = num_action_ids(config)

    players_size + stage_size + prev_action_size + name_size +
      projectile_size + item_size +
      distance_size + relative_pos_size + frame_count_size + action_ids_size
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

    # Optional: distance between players
    embeddings = if config.with_distance do
      [embed_distance(game_state, own, opponent) | embeddings]
    else
      embeddings
    end

    # Optional: relative position (dx, dy from own to opponent)
    embeddings = if config.with_relative_pos do
      [embed_relative_pos(own, opponent) | embeddings]
    else
      embeddings
    end

    # Optional: frame count (game timer)
    embeddings = if config.with_frame_count do
      [embed_frame_count(game_state) | embeddings]
    else
      embeddings
    end

    # Append player action IDs at end when using learned embeddings
    # The network extracts these, embeds them, and concatenates with the rest
    embeddings = if config.player.action_mode == :learned do
      own_action = PlayerEmbed.get_action_id(own)
      opp_action = PlayerEmbed.get_action_id(opponent)
      action_ids = Nx.tensor([own_action, opp_action], type: :f32)
      [action_ids | embeddings]
    else
      embeddings
    end

    # Append Nana action IDs when using enhanced Nana mode
    # Order: [player_own, player_opp, nana_own, nana_opp]
    embeddings = if config.player.nana_mode == :enhanced and config.player.with_nana do
      own_nana_action = PlayerEmbed.get_nana_action_id(own)
      opp_nana_action = PlayerEmbed.get_nana_action_id(opponent)
      nana_action_ids = Nx.tensor([own_nana_action, opp_nana_action], type: :f32)
      [nana_action_ids | embeddings]
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

      # Base embeddings
      base_embs = [own_emb, opp_emb, stage_emb, prev_action_emb, name_emb]

      # Add distance if configured
      embs_with_distance = if config.with_distance do
        distances = Enum.zip([game_states, own_players, opponent_players])
        |> Enum.map(fn {gs, own, opp} ->
          cond do
            gs.distance && gs.distance > 0 -> gs.distance
            own && opp ->
              dx = (opp.x || 0.0) - (own.x || 0.0)
              dy = (opp.y || 0.0) - (own.y || 0.0)
              :math.sqrt(dx * dx + dy * dy)
            true -> 0.0
          end
        end)
        distance_emb = Primitives.batch_float_embed(distances, scale: 1/200.0, lower: 0.0, upper: 1.5)
        base_embs ++ [distance_emb]
      else
        base_embs
      end

      # Add relative position if configured
      embs_with_rel_pos = if config.with_relative_pos do
        rel_positions = Enum.zip(own_players, opponent_players)
        |> Enum.map(fn {own, opp} ->
          dx = if own && opp, do: ((opp.x || 0.0) - (own.x || 0.0)) / 200.0, else: 0.0
          dy = if own && opp, do: ((opp.y || 0.0) - (own.y || 0.0)) / 100.0, else: 0.0
          {dx, dy}
        end)
        dxs = Enum.map(rel_positions, fn {dx, _} -> dx end)
        dys = Enum.map(rel_positions, fn {_, dy} -> dy end)
        dx_emb = Primitives.batch_float_embed(dxs, scale: 1.0)
        dy_emb = Primitives.batch_float_embed(dys, scale: 1.0)
        embs_with_distance ++ [dx_emb, dy_emb]
      else
        embs_with_distance
      end

      # Add frame count if configured
      embs_with_frame = if config.with_frame_count do
        frames = Enum.map(game_states, fn gs -> (gs.frame || 0) / 28800.0 end)
        frame_emb = Primitives.batch_float_embed(frames, scale: 1.0, lower: 0.0, upper: 1.0)
        embs_with_rel_pos ++ [frame_emb]
      else
        embs_with_rel_pos
      end

      # Append player action IDs at end when using learned embeddings
      # Shape: [batch, 2] with own_action and opponent_action as floats
      embs_with_actions = if config.player.action_mode == :learned do
        own_actions = Enum.map(own_players, &PlayerEmbed.get_action_id/1)
        opp_actions = Enum.map(opponent_players, &PlayerEmbed.get_action_id/1)
        # Stack as [batch, 2] tensor
        action_ids = Nx.stack([
          Nx.tensor(own_actions, type: :f32),
          Nx.tensor(opp_actions, type: :f32)
        ], axis: 1)
        embs_with_frame ++ [action_ids]
      else
        embs_with_frame
      end

      # Append Nana action IDs when using enhanced Nana mode
      # Order: [player_own, player_opp, nana_own, nana_opp]
      all_embs = if config.player.nana_mode == :enhanced and config.player.with_nana do
        own_nana_actions = Enum.map(own_players, &PlayerEmbed.get_nana_action_id/1)
        opp_nana_actions = Enum.map(opponent_players, &PlayerEmbed.get_nana_action_id/1)
        # Stack as [batch, 2] tensor
        nana_action_ids = Nx.stack([
          Nx.tensor(own_nana_actions, type: :f32),
          Nx.tensor(opp_nana_actions, type: :f32)
        ], axis: 1)
        embs_with_actions ++ [nana_action_ids]
      else
        embs_with_actions
      end

      # Concatenate all: [batch, total_embed_size]
      Nx.concatenate(all_embs, axis: 1)
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

  # Embed distance between players
  # Uses GameState.distance if available, otherwise calculates from positions
  defp embed_distance(game_state, own, opponent) do
    distance = cond do
      game_state.distance && game_state.distance > 0 ->
        game_state.distance
      own && opponent ->
        dx = (opponent.x || 0.0) - (own.x || 0.0)
        dy = (opponent.y || 0.0) - (own.y || 0.0)
        :math.sqrt(dx * dx + dy * dy)
      true ->
        0.0
    end
    # Normalize: typical stage width ~200, max distance ~300
    normalized = min(distance / 200.0, 1.5)
    Nx.tensor([normalized], type: :f32)
  end

  # Embed relative position from own player to opponent (ego-centric)
  # Positive dx = opponent is to the right, positive dy = opponent is above
  defp embed_relative_pos(own, opponent) do
    dx = if own && opponent do
      ((opponent.x || 0.0) - (own.x || 0.0)) / 200.0  # Normalize by stage width
    else
      0.0
    end

    dy = if own && opponent do
      ((opponent.y || 0.0) - (own.y || 0.0)) / 100.0  # Normalize by typical height range
    else
      0.0
    end

    Nx.tensor([dx, dy], type: :f32)
  end

  # Embed game frame count (useful for time pressure awareness)
  # Melee runs at 60 FPS, typical match is 8 minutes = 28800 frames
  defp embed_frame_count(game_state) do
    frame = game_state.frame || 0
    # Normalize: 8 minutes = 28800 frames, cap at 1.0
    normalized = min(frame / 28800.0, 1.0)
    Nx.tensor([normalized], type: :f32)
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
