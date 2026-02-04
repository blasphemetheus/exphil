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

  ## Submodules

  This module delegates to specialized submodules:
  - `Game.Config` - Configuration and size calculations
  - `Game.Stage` - Stage embedding (full, compact, learned modes)
  - `Game.Spatial` - Distance, relative position, frame count
  - `Game.Projectiles` - Projectile embedding
  - `Game.Items` - Item embedding (Link bombs, etc.)

  ## See Also

  - `ExPhil.Embeddings.Player` - Per-player state embedding
  - `ExPhil.Embeddings.Controller` - Controller input embedding
  - `ExPhil.Embeddings.Primitives` - Low-level encoding utilities
  - `ExPhil.Networks.Policy` - The network that consumes these embeddings
  """

  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Embeddings.Player, as: PlayerEmbed
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Embeddings.Game.{Config, Stage, Spatial, Projectiles, Items}
  alias ExPhil.Bridge.{GameState, ControllerState}

  # ============================================================================
  # Configuration Struct (mirrors Config module for backwards compatibility)
  # ============================================================================

  # Keep struct in Game for backwards compatibility with existing code
  # that uses %Game{} directly. Internally delegates to Config module.
  defstruct player: %PlayerEmbed{},
            controller: %ControllerEmbed{},
            with_projectiles: true,
            max_projectiles: 5,
            with_items: false,
            max_items: 5,
            num_player_names: 112,
            with_distance: true,
            with_relative_pos: true,
            with_frame_count: true,
            stage_mode: :one_hot_compact

  @type stage_mode :: :one_hot_full | :one_hot_compact | :learned

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
          with_frame_count: boolean(),
          stage_mode: stage_mode()
        }

  @doc """
  Returns the default game embedding configuration.
  """
  @spec default_config() :: config()
  def default_config do
    %__MODULE__{
      player: PlayerEmbed.default_config(),
      controller: ControllerEmbed.default_config()
    }
  end

  # Size calculations - delegate to Config module (works with both struct types)
  defdelegate embedding_size(config), to: Config
  defdelegate raw_embedding_size(config), to: Config
  defdelegate continuous_embedding_size(config), to: Config
  defdelegate padding_for_alignment(size), to: Config

  # ID counts
  defdelegate num_action_ids(config), to: Config
  defdelegate num_character_ids(config), to: Config
  defdelegate num_stage_ids(config), to: Config
  defdelegate num_total_ids(config), to: Config

  # Learned embedding predicates
  defdelegate uses_learned_actions?(config), to: Config
  defdelegate uses_enhanced_nana?(config), to: Config
  defdelegate uses_learned_characters?(config), to: Config
  defdelegate uses_learned_stages?(config), to: Config
  defdelegate uses_learned_stage?(config), to: Config

  # ============================================================================
  # Stage Functions (delegated to Stage module)
  # ============================================================================

  @doc """
  Get the stage embedding size for the current config.
  """
  @spec stage_embedding_size(config()) :: non_neg_integer()
  def stage_embedding_size(config \\ Config.default()) do
    Stage.embedding_size(config.stage_mode)
  end

  @doc """
  Embed stage according to the configured mode.
  """
  @spec embed_stage(non_neg_integer() | nil, config()) :: Nx.Tensor.t() | :skip
  def embed_stage(stage_id, config \\ Config.default()) do
    Stage.embed(stage_id, config.stage_mode)
  end

  @doc """
  Embed stage as compact 7-dim one-hot.
  """
  defdelegate embed_stage_compact(stage_id), to: Stage, as: :embed_compact

  @doc """
  Batch embed stages according to the configured mode.
  """
  @spec embed_stages_batch(list(non_neg_integer() | nil), config()) :: Nx.Tensor.t() | :skip
  def embed_stages_batch(stage_ids, config \\ Config.default()) do
    Stage.embed_batch(stage_ids, config.stage_mode)
  end

  # Stage helpers
  defdelegate get_stage_id(game_state), to: Stage, as: :get_id
  defdelegate competitive_stage?(stage_id), to: Stage, as: :competitive?
  defdelegate competitive_stage_index(stage_id), to: Stage, as: :competitive_index
  defdelegate competitive_stages(), to: Stage

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
    config = Keyword.get(opts, :config, Config.default())
    name_id = Keyword.get(opts, :name_id, 0)

    # Get players in ego-centric order (own first, opponent second)
    {own, opponent} = get_players_ego(game_state, own_port)

    # Base embeddings (players, action, name)
    base_embeddings = [
      # Players (ego-centric)
      PlayerEmbed.embed(own, config.player),
      PlayerEmbed.embed(opponent, config.player),

      # Previous action
      ControllerEmbed.embed_continuous(prev_action),

      # Player name/tag ID (for style learning)
      Primitives.one_hot(name_id, size: config.num_player_names, clamp: true)
    ]

    # Stage embedding (only when using one-hot modes, not learned)
    embeddings =
      if config.stage_mode in [:one_hot_full, :one_hot_compact] do
        stage_emb = embed_stage(game_state.stage, config)
        # Insert stage after players, before prev_action
        [
          Enum.at(base_embeddings, 0),
          Enum.at(base_embeddings, 1),
          stage_emb | Enum.drop(base_embeddings, 2)
        ]
      else
        base_embeddings
      end

    # Optional: projectiles
    embeddings =
      if config.with_projectiles do
        [Projectiles.embed(game_state.projectiles, config.max_projectiles) | embeddings]
      else
        embeddings
      end

    # Optional: items (Link bombs, etc.)
    embeddings =
      if config.with_items do
        [Items.embed(game_state.items, own_port, config.max_items) | embeddings]
      else
        embeddings
      end

    # Optional: distance between players
    embeddings =
      if config.with_distance do
        [Spatial.embed_distance(game_state, own, opponent) | embeddings]
      else
        embeddings
      end

    # Optional: relative position (dx, dy from own to opponent)
    embeddings =
      if config.with_relative_pos do
        [Spatial.embed_relative_pos(own, opponent) | embeddings]
      else
        embeddings
      end

    # Optional: frame count (game timer)
    embeddings =
      if config.with_frame_count do
        [Spatial.embed_frame_count(game_state) | embeddings]
      else
        embeddings
      end

    # Append player action IDs at end when using learned embeddings
    embeddings =
      if config.player.action_mode == :learned do
        own_action = PlayerEmbed.get_action_id(own)
        opp_action = PlayerEmbed.get_action_id(opponent)
        action_ids = Nx.tensor([own_action, opp_action], type: :f32)
        [action_ids | embeddings]
      else
        embeddings
      end

    # Append Nana action IDs when using enhanced Nana mode
    embeddings =
      if config.player.nana_mode == :enhanced and config.player.with_nana do
        own_nana_action = PlayerEmbed.get_nana_action_id(own)
        opp_nana_action = PlayerEmbed.get_nana_action_id(opponent)
        nana_action_ids = Nx.tensor([own_nana_action, opp_nana_action], type: :f32)
        [nana_action_ids | embeddings]
      else
        embeddings
      end

    # Append character IDs when using learned character embeddings
    embeddings =
      if config.player.character_mode == :learned do
        own_char = PlayerEmbed.get_character_id(own)
        opp_char = PlayerEmbed.get_character_id(opponent)
        char_ids = Nx.tensor([own_char, opp_char], type: :f32)
        [char_ids | embeddings]
      else
        embeddings
      end

    # Append stage ID at very end when using learned stage embeddings
    embeddings =
      if config.stage_mode == :learned do
        stage_id = Nx.tensor([game_state.stage || 0], type: :f32)
        [stage_id | embeddings]
      else
        embeddings
      end

    # Add padding for tensor core alignment
    raw_size = Config.raw_embedding_size(config)
    padding_size = Config.padding_for_alignment(raw_size)

    embeddings =
      if padding_size > 0 do
        padding = Nx.broadcast(0.0, {padding_size}) |> Nx.as_type(:f32)
        [padding | embeddings]
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
    config = Keyword.get(opts, :config, Config.default())

    if Enum.empty?(game_states) do
      Nx.broadcast(0.0, {0, Config.embedding_size(config)})
    else
      batch_size = length(game_states)

      # name_id can be a single integer (broadcast to all) or a list (one per frame)
      name_ids_opt = Keyword.get(opts, :name_id, 0)

      name_ids =
        case name_ids_opt do
          id when is_integer(id) -> List.duplicate(id, batch_size)
          ids when is_list(ids) -> ids
          _ -> List.duplicate(0, batch_size)
        end

      # Extract own players and opponent players
      {own_players, opponent_players} =
        game_states
        |> Enum.map(fn gs -> get_players_ego(gs, own_port) end)
        |> Enum.unzip()

      # Extract stages
      stages = Enum.map(game_states, fn gs -> gs.stage || 0 end)

      # Batch embed players
      own_emb = PlayerEmbed.embed_batch(own_players, config.player)
      opp_emb = PlayerEmbed.embed_batch(opponent_players, config.player)

      # Batch embed name_ids (style-conditional training)
      name_emb =
        Primitives.batch_one_hot(Nx.tensor(name_ids, type: :s32),
          size: config.num_player_names,
          clamp: true
        )

      # Previous action (zeros since we don't have it in temporal sequences)
      prev_action_emb =
        Nx.broadcast(0.0, {batch_size, ControllerEmbed.continuous_embedding_size()})

      # Base embeddings (stage only included for one-hot modes)
      base_embs =
        if config.stage_mode in [:one_hot_full, :one_hot_compact] do
          stage_emb = Stage.embed_batch(stages, config.stage_mode)
          [own_emb, opp_emb, stage_emb, prev_action_emb, name_emb]
        else
          [own_emb, opp_emb, prev_action_emb, name_emb]
        end

      # Add distance if configured
      embs_with_distance =
        if config.with_distance do
          distances =
            Enum.zip([game_states, own_players, opponent_players])
            |> Enum.map(fn {gs, own, opp} ->
              Spatial.calculate_distance(gs, own, opp)
            end)

          distance_emb =
            Primitives.batch_float_embed(distances, scale: 1 / 200.0, lower: 0.0, upper: 1.5)

          base_embs ++ [distance_emb]
        else
          base_embs
        end

      # Add relative position if configured
      embs_with_rel_pos =
        if config.with_relative_pos do
          rel_positions =
            Enum.zip(own_players, opponent_players)
            |> Enum.map(fn {own, opp} -> Spatial.calculate_relative_pos(own, opp) end)

          dxs = Enum.map(rel_positions, fn {dx, _} -> dx end)
          dys = Enum.map(rel_positions, fn {_, dy} -> dy end)
          dx_emb = Primitives.batch_float_embed(dxs, scale: 1.0)
          dy_emb = Primitives.batch_float_embed(dys, scale: 1.0)
          embs_with_distance ++ [dx_emb, dy_emb]
        else
          embs_with_distance
        end

      # Add frame count if configured
      embs_with_frame =
        if config.with_frame_count do
          frame_emb = Spatial.embed_frame_counts_batch(game_states)
          embs_with_rel_pos ++ [frame_emb]
        else
          embs_with_rel_pos
        end

      # Add projectiles if configured
      embs_with_projectiles =
        if config.with_projectiles do
          proj_emb = Projectiles.embed_batch(game_states, config.max_projectiles)
          embs_with_frame ++ [proj_emb]
        else
          embs_with_frame
        end

      # Append player action IDs at end when using learned embeddings
      embs_with_actions =
        if config.player.action_mode == :learned do
          own_actions = Enum.map(own_players, &PlayerEmbed.get_action_id/1)
          opp_actions = Enum.map(opponent_players, &PlayerEmbed.get_action_id/1)
          action_ids =
            Nx.stack(
              [
                Nx.tensor(own_actions, type: :f32),
                Nx.tensor(opp_actions, type: :f32)
              ],
              axis: 1
            )

          embs_with_projectiles ++ [action_ids]
        else
          embs_with_projectiles
        end

      # Append Nana action IDs when using enhanced Nana mode
      embs_with_nana =
        if config.player.nana_mode == :enhanced and config.player.with_nana do
          own_nana_actions = Enum.map(own_players, &PlayerEmbed.get_nana_action_id/1)
          opp_nana_actions = Enum.map(opponent_players, &PlayerEmbed.get_nana_action_id/1)
          nana_action_ids =
            Nx.stack(
              [
                Nx.tensor(own_nana_actions, type: :f32),
                Nx.tensor(opp_nana_actions, type: :f32)
              ],
              axis: 1
            )

          embs_with_actions ++ [nana_action_ids]
        else
          embs_with_actions
        end

      # Append character IDs when using learned character embeddings
      embs_with_chars =
        if config.player.character_mode == :learned do
          own_chars = Enum.map(own_players, &PlayerEmbed.get_character_id/1)
          opp_chars = Enum.map(opponent_players, &PlayerEmbed.get_character_id/1)
          char_ids =
            Nx.stack(
              [
                Nx.tensor(own_chars, type: :f32),
                Nx.tensor(opp_chars, type: :f32)
              ],
              axis: 1
            )

          embs_with_nana ++ [char_ids]
        else
          embs_with_nana
        end

      # Append stage ID at very end when using learned stage embeddings
      embs_with_stage =
        if config.stage_mode == :learned do
          stage_ids = Nx.tensor(stages, type: :f32) |> Nx.reshape({:auto, 1})
          embs_with_chars ++ [stage_ids]
        else
          embs_with_chars
        end

      # Add padding for tensor core alignment
      raw_size = Config.raw_embedding_size(config)
      padding_size = Config.padding_for_alignment(raw_size)

      all_embs =
        if padding_size > 0 do
          padding = Nx.broadcast(0.0, {batch_size, padding_size}) |> Nx.as_type(:f32)
          embs_with_stage ++ [padding]
        else
          embs_with_stage
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
    config = Keyword.get(opts, :config, Config.default())

    {own, opponent} = get_players_ego(game_state, own_port)

    base_embeddings = [
      PlayerEmbed.embed(own, config.player),
      PlayerEmbed.embed(opponent, config.player)
    ]

    # Add stage embedding only for one-hot modes
    embeddings =
      if config.stage_mode in [:one_hot_full, :one_hot_compact] do
        base_embeddings ++ [embed_stage(game_state.stage, config)]
      else
        base_embeddings
      end

    Nx.concatenate(embeddings)
  end

  # ============================================================================
  # Batch Embedding
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
    config = Keyword.get(opts, :config, Config.default())
    Nx.broadcast(0.0, {Config.embedding_size(config)})
  end

  @doc """
  Create a batched dummy embedding.
  """
  @spec dummy_batch(non_neg_integer(), keyword()) :: Nx.Tensor.t()
  def dummy_batch(batch_size, opts \\ []) do
    config = Keyword.get(opts, :config, Config.default())
    Nx.broadcast(0.0, {batch_size, Config.embedding_size(config)})
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
end
