defmodule ExPhil.Embeddings.Game.Stage do
  @moduledoc """
  Stage embedding for Melee game states.

  Supports three embedding modes:
  - `:one_hot_full` - 64-dim one-hot (all stages)
  - `:one_hot_compact` - 7-dim (6 competitive stages + 1 "other")
  - `:learned` - Stage ID appended for network embedding lookup

  ## Competitive Stages

  The 6 tournament-legal stages:
  - Fountain of Dreams (ID: 2)
  - Pokémon Stadium (ID: 3)
  - Yoshi's Story (ID: 8)
  - Dream Land (ID: 28)
  - Battlefield (ID: 31)
  - Final Destination (ID: 32)

  ## See Also

  - `ExPhil.Embeddings.Game.Config` - Configuration options
  - `ExPhil.Embeddings.Primitives` - Low-level encoding
  """

  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Bridge.GameState

  # ============================================================================
  # Stage Constants
  # ============================================================================

  # Competitive stage IDs (from replay files, NOT libmelee)
  @competitive_stages %{
    fountain_of_dreams: 2,
    pokemon_stadium: 3,
    yoshis_story: 8,
    dream_land: 28,
    battlefield: 31,
    final_destination: 32
  }

  # Reverse lookup: stage_id -> index (0-5 for compact one-hot)
  @competitive_stage_index %{
    # FoD
    2 => 0,
    # PS
    3 => 1,
    # YS
    8 => 2,
    # DL
    28 => 3,
    # BF
    31 => 4,
    # FD
    32 => 5
  }

  # Full one-hot size
  @num_stages_full 64
  # 6 competitive + 1 "other"
  @num_stages_compact 7

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Get the stage embedding size for the given mode.

  - `:one_hot_full` - 64 dimensions
  - `:one_hot_compact` - 7 dimensions (6 competitive + 1 other)
  - `:learned` - 0 dimensions (stage ID appended at end for network)
  """
  @spec embedding_size(atom()) :: non_neg_integer()
  def embedding_size(:one_hot_full), do: @num_stages_full
  def embedding_size(:one_hot_compact), do: @num_stages_compact
  def embedding_size(:learned), do: 0

  @doc """
  Embed stage according to the given mode.

  - `:one_hot_full` - 64-dim one-hot
  - `:one_hot_compact` - 7-dim: 6 competitive stages + 1 "other"
  - `:learned` - Returns `:skip` (stage ID appended at end for network)
  """
  @spec embed(non_neg_integer() | nil, atom()) :: Nx.Tensor.t() | :skip
  def embed(stage_id, mode \\ :one_hot_compact)

  def embed(nil, mode), do: embed(0, mode)

  def embed(stage_id, :one_hot_full) do
    Primitives.stage_embed(stage_id)
  end

  def embed(stage_id, :one_hot_compact) do
    embed_compact(stage_id)
  end

  def embed(_stage_id, :learned) do
    # No stage embedding - stage ID will be appended at end
    :skip
  end

  @doc """
  Embed stage as compact 7-dim one-hot (6 competitive + 1 other).
  """
  @spec embed_compact(non_neg_integer()) :: Nx.Tensor.t()
  def embed_compact(stage_id) do
    case Map.get(@competitive_stage_index, stage_id) do
      nil ->
        # Not a competitive stage - set "other" flag (index 6)
        Nx.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], type: :f32)

      idx ->
        # Competitive stage - one-hot at position idx (0-5)
        zeros = List.duplicate(0.0, @num_stages_compact)

        zeros
        |> List.replace_at(idx, 1.0)
        |> Nx.tensor(type: :f32)
    end
  end

  @doc """
  Batch embed stages according to the given mode.

  Returns `:skip` for learned mode (stage IDs appended separately).
  """
  @spec embed_batch(list(non_neg_integer() | nil), atom()) :: Nx.Tensor.t() | :skip
  def embed_batch(stage_ids, mode \\ :one_hot_compact)

  def embed_batch(stage_ids, :one_hot_full) do
    ids = Enum.map(stage_ids, &(&1 || 0))
    Primitives.batch_one_hot(Nx.tensor(ids, type: :s32), size: @num_stages_full, clamp: true)
  end

  def embed_batch(stage_ids, :one_hot_compact) do
    stage_ids
    |> Enum.map(&embed_compact(&1 || 0))
    |> Nx.stack()
  end

  def embed_batch(_stage_ids, :learned) do
    :skip
  end

  # ============================================================================
  # Stage ID Helpers
  # ============================================================================

  @doc """
  Get stage ID for learned embedding lookup.

  Returns the raw stage ID (0-63 range), suitable for embedding table lookup.
  """
  @spec get_id(GameState.t() | nil) :: non_neg_integer()
  def get_id(nil), do: 0
  def get_id(%GameState{stage: nil}), do: 0
  def get_id(%GameState{stage: stage}), do: stage

  @doc """
  Check if a stage ID is a competitive stage.
  """
  @spec competitive?(non_neg_integer()) :: boolean()
  def competitive?(stage_id) do
    Map.has_key?(@competitive_stage_index, stage_id)
  end

  @doc """
  Get the competitive stage index (0-5) or nil if not competitive.
  """
  @spec competitive_index(non_neg_integer()) :: non_neg_integer() | nil
  def competitive_index(stage_id) do
    Map.get(@competitive_stage_index, stage_id)
  end

  @doc """
  Get the map of competitive stage atoms to IDs.
  """
  @spec competitive_stages() :: map()
  def competitive_stages, do: @competitive_stages

  @doc """
  Get the number of stages for full one-hot encoding.
  """
  @spec num_stages_full() :: non_neg_integer()
  def num_stages_full, do: @num_stages_full

  @doc """
  Get the number of stages for compact one-hot encoding.
  """
  @spec num_stages_compact() :: non_neg_integer()
  def num_stages_compact, do: @num_stages_compact
end
