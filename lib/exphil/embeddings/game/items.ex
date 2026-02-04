defmodule ExPhil.Embeddings.Game.Items do
  @moduledoc """
  Item embedding for Melee game states.

  Embeds item information including:
  - Existence flag
  - Position (x, y)
  - Category (bomb, melee, ranged, container, other)
  - Held status and owner
  - Timer (for bombs)

  Particularly important for Link (bombs, boomerang) and other
  item-using characters.

  ## Embedding Size

  Each item is 12 dimensions:
  - exists: 1
  - x, y: 2
  - category (one-hot): 6
  - is_held: 1
  - held_by_self: 1
  - timer: 1

  ## Item Categories

  - 0: none/unknown
  - 1: bomb (Link's bomb)
  - 2: melee weapon
  - 3: ranged weapon
  - 4: container
  - 5: other

  ## See Also

  - `ExPhil.Embeddings.Game` - Main embedding module
  - `ExPhil.Bridge.Item` - Item data structure
  """

  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Bridge.Item

  # ============================================================================
  # Constants
  # ============================================================================

  # Link's bomb timer is typically 0-180 frames (3 seconds)
  @bomb_timer_max 180.0

  @doc """
  Embedding size per item (12 dimensions).
  """
  @spec embedding_size() :: non_neg_integer()
  def embedding_size do
    # exists (1) + x,y (2) + category (6) + is_held (1) + held_by_self (1) + timer (1)
    1 + 2 + 6 + 1 + 1 + 1
  end

  # ============================================================================
  # Single Item Embedding
  # ============================================================================

  @doc """
  Embed a single item.
  """
  @spec embed_single(Item.t(), integer()) :: Nx.Tensor.t()
  def embed_single(%Item{} = item, own_port) do
    category = Item.item_category(item)
    is_held = Item.held?(item)
    held_by_self = is_held and item.held_by == own_port

    Nx.concatenate([
      # exists
      Primitives.bool_embed(true),
      Primitives.xy_embed(item.x),
      Primitives.xy_embed(item.y),
      # category (6 categories)
      Primitives.one_hot(category, size: 6, clamp: true),
      Primitives.bool_embed(is_held),
      Primitives.bool_embed(held_by_self),
      # normalized timer
      Primitives.float_embed(normalize_timer(item.timer))
    ])
  end

  # ============================================================================
  # List Embedding
  # ============================================================================

  @doc """
  Embed a list of items, padding/truncating to max_items.

  Returns tensor of shape [max_items * embedding_size].
  """
  @spec embed(list(Item.t()) | nil, integer(), non_neg_integer()) :: Nx.Tensor.t()
  def embed(nil, _own_port, max_items) do
    Nx.broadcast(0.0, {max_items * embedding_size()})
  end

  def embed(items, own_port, max_items) when is_list(items) do
    # Pad or truncate to max_items
    items = Enum.take(items, max_items)
    num_existing = length(items)
    padding_count = max_items - num_existing

    embedded = Enum.map(items, &embed_single(&1, own_port))

    padding =
      if padding_count > 0 do
        [Nx.broadcast(0.0, {padding_count * embedding_size()})]
      else
        []
      end

    Nx.concatenate(embedded ++ padding)
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  @doc """
  Normalize item timer to [0, 1] range.

  Link's bomb timer is typically 0-180 frames (3 seconds).
  """
  @spec normalize_timer(integer() | nil) :: float()
  def normalize_timer(nil), do: 0.0

  def normalize_timer(timer) when is_integer(timer) do
    min(1.0, timer / @bomb_timer_max)
  end

  def normalize_timer(_), do: 0.0
end
