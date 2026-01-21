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
    with_speeds: true,  # Enabled for momentum/velocity info (+5 dims per player)
    with_nana: true,
    with_frame_info: true  # Hitstun frames + action frame (+2 dims per player)
  ]

  @type config :: %__MODULE__{
    xy_scale: float(),
    shield_scale: float(),
    speed_scale: float(),
    with_speeds: boolean(),
    with_nana: boolean(),
    with_frame_info: boolean()
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

    # Hitstun frames remaining + action frame (how far into animation)
    frame_info_size = if config.with_frame_info, do: 2, else: 0

    nana_size = if config.with_nana do
      # Nana has base embedding + exists flag + optional speeds + optional frame info
      nana_base = base_embedding_size() + 1
      nana_speeds = if config.with_speeds, do: 5, else: 0
      nana_frames = if config.with_frame_info, do: 2, else: 0
      nana_base + nana_speeds + nana_frames
    else
      0
    end

    base_size + speed_size + frame_info_size + nana_size
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

    embeddings = if config.with_frame_info do
      [embed_frame_info(player) | embeddings]
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
  Batch embed multiple players at once - MUCH faster than calling embed/2 in a loop.

  This extracts all values first (Elixir work), then does all Nx operations
  in batch, reducing overhead from O(N * ops) to O(ops).

  ## Returns
    Tensor of shape [batch_size, embedding_size]
  """
  @spec embed_batch([PlayerState.t() | nil], config()) :: Nx.Tensor.t()
  def embed_batch(players, config \\ default_config()) when is_list(players) do
    # Handle empty list
    if Enum.empty?(players) do
      Nx.broadcast(0.0, {0, embedding_size(config)})
    else
      embed_batch_base(players, config)
    end
  end

  defp embed_batch_base(players, config) do
    # Extract all values into lists (pure Elixir - fast)
    percents = Enum.map(players, fn p -> (p && p.percent) || 0.0 end)
    facings = Enum.map(players, fn p -> (p && p.facing) || false end)
    xs = Enum.map(players, fn p -> (p && p.x) || 0.0 end)
    ys = Enum.map(players, fn p -> (p && p.y) || 0.0 end)
    actions = Enum.map(players, fn p -> (p && p.action) || 0 end)
    characters = Enum.map(players, fn p -> (p && p.character) || 0 end)
    invulnerables = Enum.map(players, fn p -> (p && p.invulnerable) || false end)
    jumps = Enum.map(players, fn p -> (p && p.jumps_left) || 0 end)
    shields = Enum.map(players, fn p -> (p && p.shield_strength) || 0.0 end)
    on_grounds = Enum.map(players, fn p -> (p && p.on_ground) || false end)

    # Batch Nx operations (few calls instead of N)
    percent_emb = Primitives.batch_float_embed(percents, scale: 0.01, lower: 0.0, upper: 5.0)
    facing_emb = Primitives.batch_bool_embed(facings, on: 1.0, off: -1.0)
    x_emb = Primitives.batch_float_embed(xs, scale: config.xy_scale)
    y_emb = Primitives.batch_float_embed(ys, scale: config.xy_scale)
    action_emb = Primitives.batch_one_hot(Nx.tensor(actions, type: :s32), size: 399, clamp: true)
    char_emb = Primitives.batch_one_hot(Nx.tensor(characters, type: :s32), size: 33, clamp: true)
    invuln_emb = Primitives.batch_bool_embed(invulnerables)
    jumps_emb = Primitives.batch_one_hot(Nx.tensor(jumps, type: :s32), size: 7, clamp: true)
    shield_emb = Primitives.batch_float_embed(shields, scale: 0.01, lower: 0.0, upper: 1.0)
    ground_emb = Primitives.batch_bool_embed(on_grounds)

    # Base embeddings
    base_embs = [
      percent_emb,   # [batch, 1]
      facing_emb,    # [batch, 1]
      x_emb,         # [batch, 1]
      y_emb,         # [batch, 1]
      action_emb,    # [batch, 399]
      char_emb,      # [batch, 33]
      invuln_emb,    # [batch, 1]
      jumps_emb,     # [batch, 7]
      shield_emb,    # [batch, 1]
      ground_emb     # [batch, 1]
    ]

    # Add speeds if configured
    embs_with_speeds = if config.with_speeds do
      speed_air_x = Enum.map(players, fn p -> (p && p.speed_air_x_self) || 0.0 end)
      speed_ground_x = Enum.map(players, fn p -> (p && p.speed_ground_x_self) || 0.0 end)
      speed_y = Enum.map(players, fn p -> (p && p.speed_y_self) || 0.0 end)
      speed_x_attack = Enum.map(players, fn p -> (p && p.speed_x_attack) || 0.0 end)
      speed_y_attack = Enum.map(players, fn p -> (p && p.speed_y_attack) || 0.0 end)

      speed_embs = [
        Primitives.batch_float_embed(speed_air_x, scale: 0.5),
        Primitives.batch_float_embed(speed_ground_x, scale: 0.5),
        Primitives.batch_float_embed(speed_y, scale: 0.5),
        Primitives.batch_float_embed(speed_x_attack, scale: 0.5),
        Primitives.batch_float_embed(speed_y_attack, scale: 0.5)
      ]

      base_embs ++ speed_embs
    else
      base_embs
    end

    # Add frame info (hitstun + action frame) if configured
    embs_with_frame_info = if config.with_frame_info do
      # Hitstun frames remaining (0-120ish, normalized to 0-1)
      hitstun_frames = Enum.map(players, fn p -> (p && p.hitstun_frames_left) || 0 end)
      # Action frame (how far into the animation, typically 0-100+, normalized)
      action_frames = Enum.map(players, fn p -> (p && p.action_frame) || 0 end)

      frame_embs = [
        # Scale hitstun: 120 frames max is reasonable, clamp to 0-1
        Primitives.batch_float_embed(hitstun_frames, scale: 1/120, lower: 0.0, upper: 1.0),
        # Scale action_frame: most animations under 60 frames, but some longer
        Primitives.batch_float_embed(action_frames, scale: 1/60, lower: 0.0, upper: 2.0)
      ]

      embs_with_speeds ++ frame_embs
    else
      embs_with_speeds
    end

    # Add Nana (Ice Climbers partner) if configured
    all_embs = if config.with_nana do
      batch_size = length(players)
      nana_embs = embed_batch_nana(players, config, batch_size)
      embs_with_frame_info ++ [nana_embs]
    else
      embs_with_frame_info
    end

    # Concatenate all embeddings: [batch, total_embed_size]
    Nx.concatenate(all_embs, axis: 1)
  end

  # Batch embed Nana for all players
  # Returns tensor of shape [batch_size, base_embedding_size + 1]
  defp embed_batch_nana(players, config, batch_size) do
    # Extract Nana structs (nil if no Nana)
    nanas = Enum.map(players, fn p -> p && p.nana end)

    # Check if any Nana exists
    has_any_nana = Enum.any?(nanas, & &1)

    if not has_any_nana do
      # No Nanas at all - return zeros
      Nx.broadcast(0.0, {batch_size, base_embedding_size() + 1})
    else
      # Extract Nana values (defaulting to 0/false for nil Nanas)
      nana_percents = Enum.map(nanas, fn n -> (n && n.percent) || 0.0 end)
      nana_facings = Enum.map(nanas, fn n -> (n && n.facing) || false end)
      nana_xs = Enum.map(nanas, fn n -> (n && n.x) || 0.0 end)
      nana_ys = Enum.map(nanas, fn n -> (n && n.y) || 0.0 end)
      nana_actions = Enum.map(nanas, fn n -> (n && n.action) || 0 end)
      nana_exists = Enum.map(nanas, fn n -> n != nil end)

      # Batch embed Nana values
      nana_percent_emb = Primitives.batch_float_embed(nana_percents, scale: 0.01, lower: 0.0, upper: 5.0)
      nana_facing_emb = Primitives.batch_bool_embed(nana_facings, on: 1.0, off: -1.0)
      nana_x_emb = Primitives.batch_float_embed(nana_xs, scale: config.xy_scale)
      nana_y_emb = Primitives.batch_float_embed(nana_ys, scale: config.xy_scale)
      nana_action_emb = Primitives.batch_one_hot(Nx.tensor(nana_actions, type: :s32), size: 399, clamp: true)

      # Nana uses character 0 (same as Popo), no jumps/shield/invuln/on_ground data
      nana_char_emb = Primitives.batch_one_hot(Nx.tensor(List.duplicate(0, batch_size), type: :s32), size: 33, clamp: true)
      nana_invuln_emb = Nx.broadcast(0.0, {batch_size, 1})
      nana_jumps_emb = Primitives.batch_one_hot(Nx.tensor(List.duplicate(0, batch_size), type: :s32), size: 7, clamp: true)
      nana_shield_emb = Nx.broadcast(0.0, {batch_size, 1})
      nana_ground_emb = Nx.broadcast(0.0, {batch_size, 1})

      # Exists flag
      nana_exists_emb = Primitives.batch_bool_embed(nana_exists)

      # Base Nana embeddings
      base_nana_embs = [
        nana_percent_emb,    # [batch, 1]
        nana_facing_emb,     # [batch, 1]
        nana_x_emb,          # [batch, 1]
        nana_y_emb,          # [batch, 1]
        nana_action_emb,     # [batch, 399]
        nana_char_emb,       # [batch, 33]
        nana_invuln_emb,     # [batch, 1]
        nana_jumps_emb,      # [batch, 7]
        nana_shield_emb,     # [batch, 1]
        nana_ground_emb,     # [batch, 1]
        nana_exists_emb      # [batch, 1]
      ]

      # Add Nana speeds if configured (Nana has limited speed data, use zeros)
      nana_embs_with_speeds = if config.with_speeds do
        nana_speed_zeros = Nx.broadcast(0.0, {batch_size, 5})
        base_nana_embs ++ [nana_speed_zeros]
      else
        base_nana_embs
      end

      # Add Nana frame info if configured (Nana has limited frame data, use zeros)
      nana_embs_with_frame_info = if config.with_frame_info do
        nana_frame_zeros = Nx.broadcast(0.0, {batch_size, 2})
        nana_embs_with_speeds ++ [nana_frame_zeros]
      else
        nana_embs_with_speeds
      end

      Nx.concatenate(nana_embs_with_frame_info, axis: 1)
    end
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
  Embed frame timing info (hitstun remaining, action frame).
  Useful for knowing when player/opponent can act.
  """
  @spec embed_frame_info(PlayerState.t()) :: Nx.Tensor.t()
  def embed_frame_info(%PlayerState{} = player) do
    # Hitstun frames: typically 0-120, normalize to 0-1
    hitstun = min((player.hitstun_frames_left || 0) / 120, 1.0)
    # Action frame: how far into animation (0-60+ typically), normalize
    action_frame = min((player.action_frame || 0) / 60, 2.0)

    Nx.tensor([hitstun, action_frame], type: :f32)
  end

  @doc """
  Embed Nana (Ice Climbers partner).
  """
  @spec embed_nana(Nana.t() | nil, config()) :: Nx.Tensor.t()
  def embed_nana(nil, config) do
    # Nana doesn't exist - return zeros with exists=false
    # Size includes base + exists + optional speeds + optional frame_info
    base_size = base_embedding_size() + 1
    speed_size = if config.with_speeds, do: 5, else: 0
    frame_size = if config.with_frame_info, do: 2, else: 0
    total_size = base_size + speed_size + frame_size
    Nx.broadcast(0.0, {total_size})
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

    # Add optional speeds (zeros for Nana since data not available)
    embs = [base, exists]
    embs = if config.with_speeds do
      embs ++ [Nx.broadcast(0.0, {5})]
    else
      embs
    end

    # Add optional frame info (zeros for Nana since data not available)
    embs = if config.with_frame_info do
      embs ++ [Nx.broadcast(0.0, {2})]
    else
      embs
    end

    Nx.concatenate(embs)
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
