defmodule ExPhil.Embeddings do
  @moduledoc """
  Embeddings for converting Melee game state to neural network input.

  This module provides the main API for embedding game state into
  Nx tensors suitable for training and inference.

  ## Architecture

  ```
  GameState ──> embed() ──> Nx.Tensor [embedding_size]
                  │
                  ├── Player 0 embedding
                  ├── Player 1 embedding
                  ├── Stage embedding
                  ├── Previous action
                  └── Player name ID
  ```

  ## Usage

      # Basic embedding
      tensor = ExPhil.Embeddings.embed(game_state, prev_action)

      # With custom config
      config = ExPhil.Embeddings.config(with_speeds: true)
      tensor = ExPhil.Embeddings.embed(game_state, prev_action, config: config)

      # Character-specific embedding
      tensor = ExPhil.Embeddings.embed_for_character(game_state, prev_action, :mewtwo)

  ## Submodules

  - `ExPhil.Embeddings.Primitives` - Base embedding functions
  - `ExPhil.Embeddings.Player` - Player state embedding
  - `ExPhil.Embeddings.Controller` - Controller input embedding
  - `ExPhil.Embeddings.Game` - Full game state embedding
  - `ExPhil.Embeddings.Characters` - Character-specific extensions

  """

  alias ExPhil.Embeddings.{Game, Primitives}
  alias ExPhil.Bridge.{GameState, ControllerState}

  # Re-export main config type
  @type config :: Game.config()

  @doc """
  Create an embedding configuration with custom options.

  ## Options
    - `:with_speeds` - Include velocity values (default: false)
    - `:with_nana` - Include Ice Climbers partner (default: true)
    - `:with_projectiles` - Include projectile data (default: false)
    - `:axis_buckets` - Discretization for stick axes (default: 16)
    - `:shoulder_buckets` - Discretization for triggers (default: 4)
    - `:stage_mode` - Stage embedding mode:
      - `:one_hot_full` (default) - 64-dim one-hot
      - `:one_hot_compact` - 7-dim (6 competitive + "other")
      - `:learned` - 1 ID + trainable embedding

  """
  @spec config(keyword()) :: config()
  def config(opts \\ []) do
    # Start with struct defaults and merge any custom options
    base = default_config()

    player_opts =
      Keyword.take(opts, [
        :with_speeds,
        :with_nana,
        :xy_scale,
        :shield_scale,
        :speed_scale,
        :nana_mode,
        :with_frame_info,
        :with_stock,
        :with_ledge_distance,
        :jumps_normalized,
        :action_mode
      ])

    player_config = struct(base.player, player_opts)

    controller_opts = Keyword.take(opts, [:axis_buckets, :shoulder_buckets])
    controller_config = struct(base.controller, controller_opts)

    game_opts =
      Keyword.take(opts, [
        :with_projectiles,
        :max_projectiles,
        :with_items,
        :max_items,
        :num_player_names,
        :with_distance,
        :with_relative_pos,
        :with_frame_count,
        :stage_mode
      ])

    struct(base, [{:player, player_config}, {:controller, controller_config} | game_opts])
  end

  @doc """
  Get the default embedding configuration.
  """
  @spec default_config() :: config()
  def default_config, do: Game.default_config()

  @doc """
  Calculate the embedding size for a configuration.
  """
  @spec embedding_size(config()) :: non_neg_integer()
  def embedding_size(config \\ default_config()) do
    Game.embedding_size(config)
  end

  @doc """
  Embed a game state into an Nx tensor.

  This is the main entry point for converting game state to network input.

  ## Parameters
    - `game_state` - Current game state from the bridge
    - `prev_action` - Previous controller state (optional)
    - `opts` - Options:
      - `:own_port` - Which port the agent is on (default: 1)
      - `:config` - Embedding configuration
      - `:name_id` - Player name/tag ID for style learning

  ## Returns
    A 1D Nx tensor of shape [embedding_size].
  """
  @spec embed(GameState.t(), ControllerState.t() | nil, keyword()) :: Nx.Tensor.t()
  def embed(game_state, prev_action \\ nil, opts \\ []) do
    own_port = Keyword.get(opts, :own_port, 1)
    Game.embed(game_state, prev_action, own_port, opts)
  end

  @doc """
  Embed just the state (no previous action).

  Useful for value function networks that don't condition on actions.
  """
  @spec embed_state(GameState.t(), keyword()) :: Nx.Tensor.t()
  def embed_state(game_state, opts \\ []) do
    own_port = Keyword.get(opts, :own_port, 1)
    Game.embed_state(game_state, own_port, opts)
  end

  @doc """
  Embed a batch of states for training.

  Takes a list of {game_state, prev_action, own_port} tuples.
  """
  @spec embed_batch(list(), keyword()) :: Nx.Tensor.t()
  def embed_batch(states, opts \\ []) do
    Game.embed_batch(states, opts)
  end

  @doc """
  Embed with character-specific features.

  Different characters may benefit from additional features:
  - Mewtwo: Teleport charge, tail position
  - G&W: Bucket fill level, hammer RNG
  - Link: Bomb count, boomerang status
  - Ganondorf: (Standard embedding is usually sufficient)
  """
  @spec embed_for_character(GameState.t(), ControllerState.t() | nil, atom(), keyword()) ::
          Nx.Tensor.t()
  def embed_for_character(game_state, prev_action, character, opts \\ []) do
    base = embed(game_state, prev_action, opts)

    case character do
      :mewtwo -> add_mewtwo_features(base, game_state, opts)
      :game_and_watch -> add_gnw_features(base, game_state, opts)
      :link -> add_link_features(base, game_state, opts)
      # Standard embedding
      :ganondorf -> base
      _ -> base
    end
  end

  # ============================================================================
  # Character-Specific Feature Extensions
  # ============================================================================

  defp add_mewtwo_features(base, game_state, opts) do
    own_port = Keyword.get(opts, :own_port, 1)
    player = GameState.get_player(game_state, own_port)

    if player do
      features =
        Nx.concatenate([
          # Is currently teleporting (action state check)
          teleporting?(player),

          # Shadow Ball charge approximation (from action frame)
          shadow_ball_charge(player),

          # Confusion active (side-B)
          confusion_active?(player)
        ])

      Nx.concatenate([base, features])
    else
      # Pad with zeros if player not available
      Nx.concatenate([base, Nx.broadcast(0.0, {3})])
    end
  end

  defp teleporting?(player) do
    # Mewtwo teleport action states (approximate)
    teleport_actions = [353, 354, 355, 356]
    is_teleport = player.action in teleport_actions
    Primitives.bool_embed(is_teleport)
  end

  defp shadow_ball_charge(player) do
    # Shadow Ball charge actions
    if player.action in [341, 342] do
      # Approximate charge level from action frame
      charge = min(1.0, (player.action_frame || 0) / 120.0)
      Primitives.float_embed(charge)
    else
      Primitives.float_embed(0.0)
    end
  end

  defp confusion_active?(player) do
    confusion_action = 351
    Primitives.bool_embed(player.action == confusion_action)
  end

  defp add_gnw_features(base, game_state, opts) do
    own_port = Keyword.get(opts, :own_port, 1)
    player = GameState.get_player(game_state, own_port)

    if player do
      features =
        Nx.concatenate([
          # Bucket fill level (0-3 projectiles absorbed)
          # Would need to track this externally - placeholder
          Primitives.float_embed(0.0),

          # Judgment hammer number (RNG)
          # Would need to track - placeholder
          Primitives.float_embed(0.0)
        ])

      Nx.concatenate([base, features])
    else
      Nx.concatenate([base, Nx.broadcast(0.0, {2})])
    end
  end

  defp add_link_features(base, game_state, opts) do
    own_port = Keyword.get(opts, :own_port, 1)
    player = GameState.get_player(game_state, own_port)

    if player do
      # Count Link's projectiles
      own_projectiles = count_projectiles_by_owner(game_state.projectiles, own_port)

      features =
        Nx.concatenate([
          # Has bomb in hand (would need action state check)
          holding_bomb?(player),

          # Number of active arrows/boomerangs
          Primitives.float_embed(own_projectiles, scale: 0.5)
        ])

      Nx.concatenate([base, features])
    else
      Nx.concatenate([base, Nx.broadcast(0.0, {2})])
    end
  end

  defp holding_bomb?(player) do
    # Link bomb-related action states (approximate)
    bomb_actions = [300..310] |> Enum.to_list()
    Primitives.bool_embed(player.action in bomb_actions)
  end

  defp count_projectiles_by_owner(nil, _owner), do: 0

  defp count_projectiles_by_owner(projectiles, owner) do
    Enum.count(projectiles, fn p -> p.owner == owner end)
  end

  @doc """
  Create a dummy embedding for model initialization.
  """
  @spec dummy(keyword()) :: Nx.Tensor.t()
  def dummy(opts \\ []) do
    Game.dummy(opts)
  end

  @doc """
  Create a dummy batch for model initialization.
  """
  @spec dummy_batch(non_neg_integer(), keyword()) :: Nx.Tensor.t()
  def dummy_batch(batch_size, opts \\ []) do
    Game.dummy_batch(batch_size, opts)
  end
end
