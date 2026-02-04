defmodule ExPhil.Embeddings.Game.Spatial do
  @moduledoc """
  Spatial and temporal features for game state embedding.

  Provides embeddings for:
  - Distance between players
  - Relative position (dx, dy from own to opponent)
  - Frame count (game timer)

  All features are normalized to reasonable ranges for neural network input.

  ## See Also

  - `ExPhil.Embeddings.Game` - Main embedding module
  - `ExPhil.Constants` - Normalization constants
  """

  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Bridge.GameState
  alias ExPhil.Constants

  # ============================================================================
  # Distance Embedding
  # ============================================================================

  @doc """
  Embed distance between players as a single normalized float.

  Uses GameState.distance if available, otherwise calculates from positions.
  Normalized by typical stage width (~200 units), clamped to [0, 1.5].
  """
  @spec embed_distance(GameState.t(), map() | nil, map() | nil) :: Nx.Tensor.t()
  def embed_distance(game_state, own, opponent) do
    distance = calculate_distance(game_state, own, opponent)
    # Normalize: typical stage width ~200, max distance ~300
    normalized = min(distance / 200.0, 1.5)
    Nx.tensor([normalized], type: :f32)
  end

  @doc """
  Calculate raw distance between players.
  """
  @spec calculate_distance(GameState.t(), map() | nil, map() | nil) :: float()
  def calculate_distance(game_state, own, opponent) do
    cond do
      game_state.distance && game_state.distance > 0 ->
        game_state.distance

      own && opponent ->
        dx = (opponent.x || 0.0) - (own.x || 0.0)
        dy = (opponent.y || 0.0) - (own.y || 0.0)
        :math.sqrt(dx * dx + dy * dy)

      true ->
        0.0
    end
  end

  @doc """
  Batch embed distances for multiple game states.

  Returns tensor of shape [batch_size, 1].
  """
  @spec embed_distances_batch(list({GameState.t(), map() | nil, map() | nil})) :: Nx.Tensor.t()
  def embed_distances_batch(state_player_tuples) do
    distances =
      Enum.map(state_player_tuples, fn {gs, own, opp} ->
        calculate_distance(gs, own, opp)
      end)

    Primitives.batch_float_embed(distances, scale: 1 / 200.0, lower: 0.0, upper: 1.5)
  end

  # ============================================================================
  # Relative Position Embedding
  # ============================================================================

  @doc """
  Embed relative position from own player to opponent (ego-centric).

  Returns 2-dim tensor: [dx, dy]
  - Positive dx = opponent is to the right
  - Positive dy = opponent is above
  """
  @spec embed_relative_pos(map() | nil, map() | nil) :: Nx.Tensor.t()
  def embed_relative_pos(own, opponent) do
    {dx, dy} = calculate_relative_pos(own, opponent)
    Nx.tensor([dx, dy], type: :f32)
  end

  @doc """
  Calculate relative position as {dx, dy} tuple.

  Normalized by stage dimensions (200 for x, 100 for y).
  """
  @spec calculate_relative_pos(map() | nil, map() | nil) :: {float(), float()}
  def calculate_relative_pos(own, opponent) do
    dx =
      if own && opponent do
        ((opponent.x || 0.0) - (own.x || 0.0)) / 200.0
      else
        0.0
      end

    dy =
      if own && opponent do
        ((opponent.y || 0.0) - (own.y || 0.0)) / 100.0
      else
        0.0
      end

    {dx, dy}
  end

  @doc """
  Batch embed relative positions.

  Returns tensor of shape [batch_size, 2].
  """
  @spec embed_relative_pos_batch(list({map() | nil, map() | nil})) :: Nx.Tensor.t()
  def embed_relative_pos_batch(player_pairs) do
    rel_positions = Enum.map(player_pairs, fn {own, opp} -> calculate_relative_pos(own, opp) end)

    dxs = Enum.map(rel_positions, fn {dx, _} -> dx end)
    dys = Enum.map(rel_positions, fn {_, dy} -> dy end)

    dx_emb = Primitives.batch_float_embed(dxs, scale: 1.0)
    dy_emb = Primitives.batch_float_embed(dys, scale: 1.0)

    Nx.concatenate([dx_emb, dy_emb], axis: 1)
  end

  # ============================================================================
  # Frame Count Embedding
  # ============================================================================

  @doc """
  Embed game frame count (useful for time pressure awareness).

  Normalized using Constants.normalize_frame/1 (0-1 range over 8 minute game).
  """
  @spec embed_frame_count(GameState.t()) :: Nx.Tensor.t()
  def embed_frame_count(game_state) do
    frame = game_state.frame || 0
    normalized = Constants.normalize_frame(frame)
    Nx.tensor([normalized], type: :f32)
  end

  @doc """
  Batch embed frame counts.

  Returns tensor of shape [batch_size, 1].
  """
  @spec embed_frame_counts_batch(list(GameState.t())) :: Nx.Tensor.t()
  def embed_frame_counts_batch(game_states) do
    frames = Enum.map(game_states, fn gs -> Constants.normalize_frame(gs.frame || 0) end)
    Primitives.batch_float_embed(frames, scale: 1.0, lower: 0.0, upper: 1.0)
  end
end
