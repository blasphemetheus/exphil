defmodule ExPhil.Embeddings.Game.Projectiles do
  @moduledoc """
  Projectile embedding for Melee game states.

  Embeds projectile information including:
  - Existence flag
  - Owner (player 1 or 2)
  - Position (x, y)
  - Type (Fox laser, Sheik needle, etc.)
  - Velocity (speed_x, speed_y)

  Important for characters like Link, Samus, Fox, Falco, Sheik who
  rely heavily on projectile play.

  ## Embedding Size

  Each projectile is 7 dimensions:
  - exists: 1
  - owner: 1
  - x, y: 2
  - type: 1
  - speed_x, speed_y: 2

  ## See Also

  - `ExPhil.Embeddings.Game` - Main embedding module
  - `ExPhil.Bridge.Projectile` - Projectile data structure
  """

  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Bridge.Projectile

  # ============================================================================
  # Constants
  # ============================================================================

  @doc """
  Embedding size per projectile (7 dimensions).
  """
  @spec embedding_size() :: non_neg_integer()
  def embedding_size do
    # exists (1) + owner (1) + x,y (2) + type (1) + speed_x,y (2)
    1 + 1 + 2 + 1 + 2
  end

  # ============================================================================
  # Single Projectile Embedding
  # ============================================================================

  @doc """
  Embed a single projectile.
  """
  @spec embed_single(Projectile.t(), pos_integer()) :: Nx.Tensor.t()
  def embed_single(%Projectile{} = proj, own_port) do
    Nx.concatenate([
      # exists
      Primitives.bool_embed(true),
      # mine? (INVARIANTS.md item 5: a ROLE, not the absolute port — the old
      # `owner * 0.5` read 0.5 for "my laser" on port 1 and 1.0 on port 2)
      Primitives.bool_embed(proj.owner == own_port),
      Primitives.xy_embed(proj.x),
      Primitives.xy_embed(proj.y),
      # simplified type
      Primitives.float_embed(proj.type, scale: 0.01),
      Primitives.speed_embed(proj.speed_x),
      Primitives.speed_embed(proj.speed_y)
    ])
  end

  # ============================================================================
  # List Embedding
  # ============================================================================

  @doc """
  Embed a list of projectiles, padding/truncating to max_projectiles.

  Returns tensor of shape [max_projectiles * embedding_size].
  """
  @spec embed(list(Projectile.t()) | nil, non_neg_integer(), pos_integer()) :: Nx.Tensor.t()
  def embed(nil, max_projectiles, _own_port) do
    Nx.broadcast(0.0, {max_projectiles * embedding_size()})
  end

  def embed(projectiles, max_projectiles, own_port) when is_list(projectiles) do
    # Pad or truncate to max_projectiles
    projectiles = Enum.take(projectiles, max_projectiles)
    num_existing = length(projectiles)
    padding_count = max_projectiles - num_existing

    embedded = Enum.map(projectiles, &embed_single(&1, own_port))

    padding =
      if padding_count > 0 do
        [Nx.broadcast(0.0, {padding_count * embedding_size()})]
      else
        []
      end

    Nx.concatenate(embedded ++ padding)
  end

  # ============================================================================
  # Batch Embedding
  # ============================================================================

  @doc """
  Batch embed projectiles for all game states.

  Returns tensor of shape [batch_size, max_projectiles * embedding_size].

  NOTE: Uses pure Elixir data extraction to avoid EXLA/Defn.Expr mismatch.
  Projectiles are extracted as floats, then converted to a single tensor.
  """
  @spec embed_batch(list(map()), non_neg_integer()) :: Nx.Tensor.t()
  def embed_batch(game_states, max_projectiles) do
    proj_size = embedding_size()

    # Extract projectile data as flat list of floats for each game state
    # This avoids creating intermediate tensors that cause backend mismatch
    all_proj_data =
      Enum.map(game_states, fn gs ->
        projectiles = gs.projectiles || []
        projectiles = Enum.take(projectiles, max_projectiles)
        own_port = ExPhil.Bridge.GameState.subject_port(gs, 1)

        # Embed each projectile as list of floats
        embedded =
          Enum.flat_map(projectiles, fn proj ->
            [
              # exists
              1.0,
              # mine? (role, not absolute port — INVARIANTS.md item 5)
              if(proj.owner == own_port, do: 1.0, else: 0.0),
              # x scaled
              (proj.x || 0.0) * 0.05,
              # y scaled
              (proj.y || 0.0) * 0.05,
              # type scaled
              (proj.type || 0) * 0.01,
              # speed_x scaled
              (proj.speed_x || 0.0) * 0.5,
              # speed_y scaled
              (proj.speed_y || 0.0) * 0.5
            ]
          end)

        # Pad with zeros for missing projectiles
        padding_count = max_projectiles - length(projectiles)
        padding = List.duplicate(0.0, padding_count * proj_size)

        embedded ++ padding
      end)

    # Convert to single tensor [batch, proj_total_size]
    Nx.tensor(all_proj_data, type: :f32)
  end
end
