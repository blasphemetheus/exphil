defmodule ExPhil.Embeddings.Player do
  @moduledoc """
  Player state embedding - converts player data to Nx tensors.

  A player's state includes position, action, damage, character, and more.
  This module converts all these values to a single concatenated tensor
  suitable for neural network input.

  ## Embedding Structure

  The default player embedding contains (in order):
  1. Percent (1 dim) - Damage percentage, scaled
  2. Facing (1 dim) - Direction facing (-1 left, +1 right)
  3. X position (1 dim) - Horizontal position, scaled
  4. Y position (1 dim) - Vertical position, scaled
  5. Action (399 dims) - One-hot action state
  6. Character (33 dims) - One-hot character
  7. Invulnerable (1 dim) - Boolean
  8. Jumps left (7 dims) - One-hot remaining jumps
  9. Shield strength (1 dim) - Current shield HP, scaled
  10. On ground (1 dim) - Boolean

  Optional additions:
  - Speed values (5 dims) - Various velocity components
  - Nana (Ice Climbers partner) - Same structure as above

  Total base size: ~447 dimensions
  """

  # Uses Nx with module prefix
  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Bridge.Player, as: PlayerState
  alias ExPhil.Bridge.Nana

  @doc """
  Configuration for player embedding.
  """
  defstruct [
    xy_scale: 0.05,
    shield_scale: 0.01,
    speed_scale: 0.5,
    with_speeds: false,
    with_nana: true
  ]

  @type config :: %__MODULE__{
    xy_scale: float(),
    shield_scale: float(),
    speed_scale: float(),
    with_speeds: boolean(),
    with_nana: boolean()
  }

  @doc """
  Default player embedding configuration.
  """
  @spec default_config() :: config()
  def default_config, do: %__MODULE__{}

  @doc """
  Calculate the embedding size for a given configuration.
  """
  @spec embedding_size(config()) :: non_neg_integer()
  def embedding_size(config \\ default_config()) do
    base_size = base_embedding_size()

    speed_size = if config.with_speeds, do: 5, else: 0

    nana_size = if config.with_nana do
      # Nana has base embedding + exists flag
      base_embedding_size() + 1
    else
      0
    end

    base_size + speed_size + nana_size
  end

  defp base_embedding_size do
    1 +  # percent
    1 +  # facing
    1 +  # x
    1 +  # y
    Primitives.embedding_size(:action) +
    Primitives.embedding_size(:character) +
    1 +  # invulnerable
    Primitives.embedding_size(:jumps_left) +
    1 +  # shield
    1    # on_ground
  end

  @doc """
  Embed a player state into an Nx tensor.

  ## Examples

      iex> player = %ExPhil.Bridge.Player{x: 0.0, y: 0.0, percent: 50.0, ...}
      iex> embed(player)
      #Nx.Tensor<f32[447]>

  """
  @spec embed(PlayerState.t(), config()) :: Nx.Tensor.t()
  def embed(player, config \\ default_config())

  def embed(nil, config) do
    # Return zeros for missing player
    Nx.broadcast(0.0, {embedding_size(config)})
  end

  def embed(%PlayerState{} = player, config) do
    embeddings = [
      embed_base(player, config)
    ]

    embeddings = if config.with_speeds do
      [embed_speeds(player, config) | embeddings]
    else
      embeddings
    end

    embeddings = if config.with_nana do
      [embed_nana(player.nana, config) | embeddings]
    else
      embeddings
    end

    # Reverse and concatenate (we prepended for efficiency)
    embeddings
    |> Enum.reverse()
    |> Nx.concatenate()
  end

  @doc """
  Embed the base player features (no speeds, no nana).
  """
  @spec embed_base(PlayerState.t(), config()) :: Nx.Tensor.t()
  def embed_base(%PlayerState{} = player, config) do
    Nx.concatenate([
      # Percent - scaled damage
      Primitives.percent_embed(player.percent || 0.0),

      # Facing direction (-1 left, +1 right)
      Primitives.facing_embed(player.facing),

      # Position
      Primitives.xy_embed(player.x || 0.0, scale: config.xy_scale),
      Primitives.xy_embed(player.y || 0.0, scale: config.xy_scale),

      # Action state (one-hot)
      Primitives.action_embed(player.action || 0),

      # Character (one-hot)
      Primitives.character_embed(player.character || 0),

      # Invulnerable flag
      Primitives.bool_embed(player.invulnerable || false),

      # Jumps remaining (one-hot)
      Primitives.jumps_left_embed(player.jumps_left || 0),

      # Shield strength
      Primitives.shield_embed(player.shield_strength || 0.0),

      # On ground flag
      Primitives.bool_embed(player.on_ground || false)
    ])
  end

  @doc """
  Embed speed/velocity values.
  """
  @spec embed_speeds(PlayerState.t(), config()) :: Nx.Tensor.t()
  def embed_speeds(%PlayerState{} = player, _config) do
    Nx.concatenate([
      Primitives.speed_embed(player.speed_air_x_self || 0.0),
      Primitives.speed_embed(player.speed_ground_x_self || 0.0),
      Primitives.speed_embed(player.speed_y_self || 0.0),
      Primitives.speed_embed(player.speed_x_attack || 0.0),
      Primitives.speed_embed(player.speed_y_attack || 0.0)
    ])
  end

  @doc """
  Embed Nana (Ice Climbers partner).
  """
  @spec embed_nana(Nana.t() | nil, config()) :: Nx.Tensor.t()
  def embed_nana(nil, _config) do
    # Nana doesn't exist - return zeros with exists=false
    size = base_embedding_size() + 1
    Nx.broadcast(0.0, {size})
  end

  def embed_nana(%Nana{} = nana, config) do
    # Convert Nana to a pseudo-player for embedding
    nana_as_player = %PlayerState{
      percent: nana.percent,
      facing: nana.facing,
      x: nana.x,
      y: nana.y,
      action: nana.action,
      character: 0,  # Nana is always same character as Popo
      invulnerable: false,
      jumps_left: 0,
      shield_strength: 0.0,
      on_ground: false,
      # Speeds not available for Nana
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil,
      stock: nana.stock,
      action_frame: 0,
      hitstun_frames_left: 0
    }

    base = embed_base(nana_as_player, config)
    exists = Primitives.bool_embed(true)

    Nx.concatenate([base, exists])
  end

  @doc """
  Embed both players from a game state.

  Returns a tensor of shape [2, player_embedding_size] or
  flattened to [2 * player_embedding_size] if `flatten: true`.
  """
  @spec embed_both(PlayerState.t(), PlayerState.t(), keyword()) :: Nx.Tensor.t()
  def embed_both(p0, p1, opts \\ []) do
    config = Keyword.get(opts, :config, default_config())
    flatten = Keyword.get(opts, :flatten, true)

    p0_embed = embed(p0, config)
    p1_embed = embed(p1, config)

    if flatten do
      Nx.concatenate([p0_embed, p1_embed])
    else
      Nx.stack([p0_embed, p1_embed])
    end
  end

  @doc """
  Embed players with ego-centric view (own player first, opponent second).

  This is the typical format for training - the agent always sees itself
  as "player 0" regardless of which port it's actually on.
  """
  @spec embed_ego(PlayerState.t(), PlayerState.t(), keyword()) :: Nx.Tensor.t()
  def embed_ego(own, opponent, opts \\ []) do
    embed_both(own, opponent, opts)
  end
end
