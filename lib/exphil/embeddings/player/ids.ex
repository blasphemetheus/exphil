defmodule ExPhil.Embeddings.Player.Ids do
  @moduledoc """
  ID extraction functions for learned embeddings.

  When using learned embeddings (`action_mode: :learned`, `character_mode: :learned`),
  the raw IDs are passed to the neural network which has trainable embedding layers.
  These functions extract the IDs from player states.

  ## Usage with Learned Embeddings

  Instead of one-hot encoding (399 dims for actions), learned embeddings:
  1. Extract raw IDs (1 integer per action)
  2. Pass IDs to network's embedding layer
  3. Network learns 64-dim representation

  This saves ~335 dims per action while allowing richer representations.

  ## See Also

  - `ExPhil.Embeddings.Player` - Uses these for learned embedding mode
  - `ExPhil.Embeddings.Game` - Combines player IDs with game state
  - `ExPhil.Networks.Policy.Embeddings` - Network embedding layers
  """

  alias ExPhil.Bridge.Player, as: PlayerState
  alias ExPhil.Bridge.Nana

  # ============================================================================
  # Action IDs
  # ============================================================================

  @doc """
  Get the action ID from a player state.

  Used when `action_mode: :learned` - the action ID is passed to the network
  which has a learned embedding layer.

  ## Returns
    Integer action ID (0-398), or 0 if player is nil.

  ## Examples

      iex> get_action_id(%PlayerState{action: 50})
      50

      iex> get_action_id(nil)
      0

  """
  @spec get_action_id(PlayerState.t() | nil) :: non_neg_integer()
  def get_action_id(nil), do: 0
  def get_action_id(%PlayerState{} = player), do: player.action || 0

  @doc """
  Get action IDs from a list of players as a tensor.

  ## Returns
    Tensor of shape [batch_size] with action IDs as integers.
  """
  @spec get_action_ids_batch([PlayerState.t() | nil]) :: Nx.Tensor.t()
  def get_action_ids_batch(players) when is_list(players) do
    players
    |> Enum.map(&get_action_id/1)
    |> Nx.tensor(type: :s32)
  end

  # ============================================================================
  # Nana Action IDs
  # ============================================================================

  @doc """
  Get Nana's action ID from a player state.

  Returns 0 if player is nil, has no Nana, or Nana has no action.
  Used in enhanced Nana mode for precise IC tech learning.

  ## Examples

      iex> get_nana_action_id(%PlayerState{nana: %Nana{action: 50}})
      50

      iex> get_nana_action_id(%PlayerState{nana: nil})
      0

      iex> get_nana_action_id(nil)
      0

  """
  @spec get_nana_action_id(PlayerState.t() | nil) :: non_neg_integer()
  def get_nana_action_id(nil), do: 0
  def get_nana_action_id(%PlayerState{nana: nil}), do: 0
  def get_nana_action_id(%PlayerState{nana: %Nana{action: action}}), do: action || 0

  @doc """
  Get Nana action IDs from a list of players as a tensor.

  ## Returns
    Tensor of shape [batch_size] with Nana action IDs as integers.
    Returns 0 for players without Nana.
  """
  @spec get_nana_action_ids_batch([PlayerState.t() | nil]) :: Nx.Tensor.t()
  def get_nana_action_ids_batch(players) when is_list(players) do
    players
    |> Enum.map(&get_nana_action_id/1)
    |> Nx.tensor(type: :s32)
  end

  # ============================================================================
  # Character IDs
  # ============================================================================

  @doc """
  Get a player's character ID for learned embedding.

  Returns the integer character ID (0-32 for Melee's 33 characters).
  Returns 0 if player is nil.

  ## Examples

      iex> get_character_id(%PlayerState{character: 10})
      10

      iex> get_character_id(nil)
      0

  """
  @spec get_character_id(PlayerState.t() | nil) :: non_neg_integer()
  def get_character_id(nil), do: 0
  def get_character_id(%PlayerState{} = player), do: player.character || 0

  @doc """
  Get character IDs from a list of players as a tensor.

  ## Returns
    Tensor of shape [batch_size] with character IDs as integers.
  """
  @spec get_character_ids_batch([PlayerState.t() | nil]) :: Nx.Tensor.t()
  def get_character_ids_batch(players) when is_list(players) do
    players
    |> Enum.map(&get_character_id/1)
    |> Nx.tensor(type: :s32)
  end
end
