defmodule ExPhil.Embeddings.Primitives do
  @moduledoc """
  Base embedding primitives for converting game values to tensors.

  These primitives mirror slippi-ai's embed.py but use Nx tensors.

  ## Embedding Types

  - `one_hot/3` - Categorical values (character, action, stage)
  - `float_embed/3` - Continuous values with optional scaling
  - `bool_embed/2` - Boolean values

  ## Design Notes

  Unlike slippi-ai which uses TensorFlow, we use Nx tensors directly.
  The embeddings are pure functions that transform data - no trainable
  parameters at this level (those live in the neural network).
  """

  # We use Nx directly with module prefix

  # ============================================================================
  # Constants (from slippi-ai)
  # ============================================================================

  # Action state count (some Kirby states go beyond this, so we clamp)
  # 399
  @action_size 0x18F

  # Character count (one larger than SANDBAG)
  # 33
  @character_size 0x21

  # Stage count (future-proofed for wacky stages)
  @stage_size 64

  # Max jumps (Puff and Kirby have 6, +1 for 0)
  @jumps_left_size 7

  # Item types (from Melee data)
  # +1 for zero, +1 for unknown
  @item_type_size 0xEC + 2

  # Item states
  # empirically determined + unknown
  @item_state_size 12 + 1

  # ============================================================================
  # One-Hot Encoding
  # ============================================================================

  @doc """
  One-hot encode a categorical value.

  ## Options
    - `:size` - Number of categories (required)
    - `:clamp` - Clamp values to valid range (default: true)

  ## Examples

      iex> one_hot(5, size: 10)
      #Nx.Tensor<f32[10]>  # [0,0,0,0,0,1,0,0,0,0]

  """
  @spec one_hot(integer() | Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def one_hot(value, opts) do
    size = Keyword.fetch!(opts, :size)
    clamp = Keyword.get(opts, :clamp, true)

    value = if is_integer(value), do: Nx.tensor(value, type: :s32), else: value

    value =
      if clamp do
        value
        |> Nx.max(0)
        |> Nx.min(size - 1)
      else
        value
      end

    Nx.equal(Nx.iota({size}), value)
    |> Nx.as_type(:f32)
  end

  @doc """
  One-hot encode with an extra dimension for unknown/invalid values.
  Values outside [0, size) are mapped to the extra dimension.
  """
  @spec one_hot_with_unknown(integer() | Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def one_hot_with_unknown(value, opts) do
    size = Keyword.fetch!(opts, :size)

    value = if is_integer(value), do: Nx.tensor(value, type: :s32), else: value

    # Check if value is in valid range
    valid = Nx.logical_and(Nx.greater_equal(value, 0), Nx.less(value, size))

    # Create one-hot for valid values, zeros for invalid
    one_hot_valid =
      Nx.select(
        valid,
        Nx.equal(Nx.iota({size}), value),
        Nx.broadcast(0, {size})
      )

    # Add extra dimension for unknown
    unknown_flag = Nx.logical_not(valid) |> Nx.as_type(:f32) |> Nx.reshape({1})

    Nx.concatenate([Nx.as_type(one_hot_valid, :f32), unknown_flag])
  end

  @doc """
  Batch one-hot encode multiple values.

  ## Examples

      iex> batch_one_hot(Nx.tensor([1, 3, 5]), size: 10)
      #Nx.Tensor<f32[3, 10]>

  """
  @spec batch_one_hot(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def batch_one_hot(values, opts) do
    size = Keyword.fetch!(opts, :size)
    clamp = Keyword.get(opts, :clamp, true)

    values =
      if clamp do
        values |> Nx.max(0) |> Nx.min(size - 1)
      else
        values
      end

    # Create indices tensor for comparison
    indices = Nx.iota({size})

    # Broadcast and compare
    # values: [batch], indices: [size]
    # result: [batch, size]
    Nx.new_axis(values, 1)
    |> Nx.equal(indices)
    |> Nx.as_type(:f32)
  end

  # ============================================================================
  # Float Embedding
  # ============================================================================

  @doc """
  Embed a float value with optional scaling and clamping.

  ## Options
    - `:scale` - Multiply value by this (default: 1.0)
    - `:bias` - Add this before scaling (default: 0.0)
    - `:lower` - Clamp to this minimum (default: -10.0)
    - `:upper` - Clamp to this maximum (default: 10.0)

  ## Examples

      iex> float_embed(50.0, scale: 0.01)
      #Nx.Tensor<f32[1]>  # [0.5]

  """
  @spec float_embed(number() | Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def float_embed(value, opts \\ []) do
    scale = Keyword.get(opts, :scale, 1.0)
    bias = Keyword.get(opts, :bias, 0.0)
    lower = Keyword.get(opts, :lower, -10.0)
    upper = Keyword.get(opts, :upper, 10.0)

    value = if is_number(value), do: Nx.tensor(value, type: :f32), else: Nx.as_type(value, :f32)

    value
    |> Nx.add(bias)
    |> Nx.multiply(scale)
    |> Nx.clip(lower, upper)
    |> ensure_trailing_dim()
  end

  @doc """
  Batch float embedding - takes a list of values, returns [batch, 1] tensor.

  ## Examples

      iex> batch_float_embed([50.0, 100.0, 150.0], scale: 0.01)
      #Nx.Tensor<f32[3, 1]>

  """
  @spec batch_float_embed([number()], keyword()) :: Nx.Tensor.t()
  def batch_float_embed(values, opts \\ []) when is_list(values) do
    scale = Keyword.get(opts, :scale, 1.0)
    bias = Keyword.get(opts, :bias, 0.0)
    lower = Keyword.get(opts, :lower, -10.0)
    upper = Keyword.get(opts, :upper, 10.0)

    values
    |> Nx.tensor(type: :f32)
    |> Nx.add(bias)
    |> Nx.multiply(scale)
    |> Nx.clip(lower, upper)
    |> Nx.reshape({:auto, 1})
  end

  @doc """
  Embed X/Y position with standard scaling.
  Default scale of 0.05 maps typical stage positions to [-5, 5] range.
  """
  @spec xy_embed(number() | Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def xy_embed(value, opts \\ []) do
    scale = Keyword.get(opts, :scale, 0.05)
    float_embed(value, scale: scale)
  end

  @doc """
  Embed percent/damage with standard scaling.
  Default scale of 0.01 maps 0-300% to [0, 3] range.
  """
  @spec percent_embed(number() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def percent_embed(value) do
    float_embed(value, scale: 0.01, lower: 0.0, upper: 5.0)
  end

  @doc """
  Embed shield strength with standard scaling.
  Default scale of 0.01 maps 0-60 to [0, 0.6] range.
  """
  @spec shield_embed(number() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def shield_embed(value) do
    float_embed(value, scale: 0.01, lower: 0.0, upper: 1.0)
  end

  @doc """
  Embed speed/velocity values.
  """
  @spec speed_embed(number() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def speed_embed(value) do
    float_embed(value, scale: 0.5)
  end

  # ============================================================================
  # Bool Embedding
  # ============================================================================

  @doc """
  Embed a boolean value as a single float.

  ## Options
    - `:on` - Value for true (default: 1.0)
    - `:off` - Value for false (default: 0.0)

  For facing direction, use `off: -1.0` to get -1/+1 encoding.
  """
  @spec bool_embed(boolean() | Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def bool_embed(value, opts \\ []) do
    on = Keyword.get(opts, :on, 1.0)
    off = Keyword.get(opts, :off, 0.0)

    value =
      cond do
        is_boolean(value) -> if value, do: on, else: off
        is_number(value) -> if value != 0, do: on, else: off
        true -> Nx.select(value, on, off)
      end

    Nx.tensor(value, type: :f32) |> ensure_trailing_dim()
  end

  @doc """
  Batch bool embedding - takes a list of booleans, returns [batch, 1] tensor.

  ## Examples

      iex> batch_bool_embed([true, false, true])
      #Nx.Tensor<f32[3, 1]>  # [[1.0], [0.0], [1.0]]

  """
  @spec batch_bool_embed([boolean() | number()], keyword()) :: Nx.Tensor.t()
  def batch_bool_embed(values, opts \\ []) when is_list(values) do
    on = Keyword.get(opts, :on, 1.0)
    off = Keyword.get(opts, :off, 0.0)

    values
    |> Enum.map(fn
      true -> on
      false -> off
      v when is_number(v) -> if v != 0, do: on, else: off
    end)
    |> Nx.tensor(type: :f32)
    |> Nx.reshape({:auto, 1})
  end

  @doc """
  Embed facing direction as -1 (left) or +1 (right).
  """
  @spec facing_embed(boolean() | integer() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def facing_embed(value) do
    bool_embed(value, on: 1.0, off: -1.0)
  end

  # ============================================================================
  # Specialized Embeddings
  # ============================================================================

  @doc """
  Embed action state (one-hot, 399 dimensions).
  """
  @spec action_embed(integer() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def action_embed(action) do
    one_hot(action, size: @action_size, clamp: true)
  end

  @doc """
  Embed character (one-hot, 33 dimensions).
  """
  @spec character_embed(integer() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def character_embed(character) do
    one_hot(character, size: @character_size, clamp: true)
  end

  @doc """
  Embed stage (one-hot, 64 dimensions).
  """
  @spec stage_embed(integer() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def stage_embed(stage) do
    one_hot(stage, size: @stage_size, clamp: true)
  end

  @doc """
  Embed jumps left (one-hot, 7 dimensions).
  """
  @spec jumps_left_embed(integer() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def jumps_left_embed(jumps) do
    one_hot(jumps, size: @jumps_left_size, clamp: true)
  end

  @doc """
  Embed jumps left as normalized float (1 dimension).

  Normalizes jumps to [0, 1] range by dividing by max jumps (6 for Puff/Kirby).
  This is more efficient than one-hot while preserving ordinal information
  (e.g., 2 jumps > 1 jump is meaningful for recovery decisions).

  ## Examples

      iex> jumps_left_normalized_embed(2)
      #Nx.Tensor<f32[1]>  # [0.333...]

      iex> jumps_left_normalized_embed(6)
      #Nx.Tensor<f32[1]>  # [1.0]

  """
  @spec jumps_left_normalized_embed(integer() | number()) :: Nx.Tensor.t()
  def jumps_left_normalized_embed(jumps) do
    # Max jumps is 6 (Puff/Kirby), most characters have 2
    # Normalize to [0, 1] range
    normalized = min((jumps || 0) / 6, 1.0)
    Nx.tensor([normalized], type: :f32)
  end

  @doc """
  Embed item type with unknown category.
  """
  @spec item_type_embed(integer() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def item_type_embed(item_type) do
    one_hot_with_unknown(item_type, size: @item_type_size - 1)
  end

  @doc """
  Embed item state with unknown category.
  """
  @spec item_state_embed(integer() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def item_state_embed(item_state) do
    one_hot_with_unknown(item_state, size: @item_state_size - 1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  defp ensure_trailing_dim(tensor) do
    case Nx.shape(tensor) do
      {} -> Nx.reshape(tensor, {1})
      {_} -> tensor
      _ -> tensor
    end
  end

  @doc """
  Get the embedding size for a given type.
  """
  @spec embedding_size(atom()) :: non_neg_integer()
  def embedding_size(:action), do: @action_size
  def embedding_size(:character), do: @character_size
  def embedding_size(:stage), do: @stage_size
  def embedding_size(:jumps_left), do: @jumps_left_size
  def embedding_size(:item_type), do: @item_type_size
  def embedding_size(:item_state), do: @item_state_size
  def embedding_size(:float), do: 1
  def embedding_size(:bool), do: 1
  def embedding_size(:xy), do: 1
  def embedding_size(:percent), do: 1
  def embedding_size(:shield), do: 1
  def embedding_size(:facing), do: 1
end
