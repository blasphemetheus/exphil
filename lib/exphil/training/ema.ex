defmodule ExPhil.Training.EMA do
  @moduledoc """
  Exponential Moving Average (EMA) for model weights.

  Maintains a shadow copy of model parameters that's updated as:

      ema_weights = decay * ema_weights + (1 - decay) * current_weights

  EMA weights often generalize better than raw training weights because
  they smooth out noise from stochastic gradient descent.

  ## Usage

      # Initialize EMA from model params
      ema = EMA.new(model_params, decay: 0.999)

      # Update after each training step
      ema = EMA.update(ema, new_params)

      # Get EMA weights for inference/evaluation
      ema_params = EMA.get_params(ema)

  ## Typical Decay Values

  - 0.999: Standard choice, ~1000 step averaging window
  - 0.9999: Slower updates, smoother but lags more
  - 0.99: Faster updates, less smoothing

  """

  import Nx.Defn

  defstruct [:params, :decay, :step]

  @type t :: %__MODULE__{
          params: map(),
          decay: float(),
          step: non_neg_integer()
        }

  @doc """
  Create a new EMA tracker initialized with the given parameters.

  ## Options

    * `:decay` - EMA decay rate (default: 0.999)

  """
  @spec new(map(), keyword()) :: t()
  def new(params, opts \\ []) do
    decay = Keyword.get(opts, :decay, 0.999)

    # Deep copy params to EMA
    ema_params = deep_copy_params(params)

    %__MODULE__{
      params: ema_params,
      decay: decay,
      step: 0
    }
  end

  @doc """
  Update EMA weights with new model parameters.

  Uses the formula: ema = decay * ema + (1 - decay) * new_params
  """
  @spec update(t(), map()) :: t()
  def update(%__MODULE__{} = ema, new_params) do
    updated_params = update_params(ema.params, new_params, ema.decay)

    %{ema | params: updated_params, step: ema.step + 1}
  end

  @doc """
  Update EMA with bias correction for early training steps.

  Bias correction compensates for the fact that EMA is initialized
  to the first weights rather than a true average. More important
  when decay is high (0.999+).

  Corrected EMA = EMA / (1 - decay^step)
  """
  @spec update_with_bias_correction(t(), map()) :: t()
  def update_with_bias_correction(%__MODULE__{} = ema, new_params) do
    # First do normal update
    updated = update(ema, new_params)

    # Apply bias correction
    correction = 1.0 - :math.pow(ema.decay, updated.step)

    corrected_params =
      if correction > 0.0 do
        apply_bias_correction(updated.params, correction)
      else
        updated.params
      end

    %{updated | params: corrected_params}
  end

  @doc """
  Get the current EMA parameters.
  """
  @spec get_params(t()) :: map()
  def get_params(%__MODULE__{params: params}), do: params

  @doc """
  Get the current step count.
  """
  @spec get_step(t()) :: non_neg_integer()
  def get_step(%__MODULE__{step: step}), do: step

  @doc """
  Copy EMA parameters back to model format for saving/inference.

  Returns params in the same structure as the original model.
  """
  @spec to_model_params(t()) :: map()
  def to_model_params(%__MODULE__{params: params}) do
    # Ensure params are on BinaryBackend for serialization
    deep_copy_params(params, backend: Nx.BinaryBackend)
  end

  @doc """
  Serialize EMA state to binary for checkpointing.
  """
  @spec serialize(t()) :: binary()
  def serialize(%__MODULE__{} = ema) do
    # Convert params to BinaryBackend before serialization
    serializable_params = deep_copy_params(ema.params, backend: Nx.BinaryBackend)

    state = %{
      params: serializable_params,
      decay: ema.decay,
      step: ema.step
    }

    :erlang.term_to_binary(state)
  end

  @doc """
  Deserialize EMA state from binary.
  """
  @spec deserialize(binary()) :: t()
  def deserialize(binary) when is_binary(binary) do
    state = :erlang.binary_to_term(binary)

    %__MODULE__{
      params: state.params,
      decay: state.decay,
      step: state.step
    }
  end

  # Private functions

  defp deep_copy_params(params, opts \\ []) do
    backend = Keyword.get(opts, :backend, Nx.BinaryBackend)

    map_nested(params, fn tensor ->
      Nx.backend_copy(tensor, backend)
    end)
  end

  defp update_params(ema_params, new_params, decay) do
    map_nested_pair(ema_params, new_params, fn ema_tensor, new_tensor ->
      ema_update(ema_tensor, new_tensor, decay)
    end)
  end

  defp apply_bias_correction(params, correction) do
    map_nested(params, fn tensor ->
      Nx.divide(tensor, correction)
    end)
  end

  # EMA update: ema = decay * ema + (1 - decay) * new
  defnp ema_update(ema_tensor, new_tensor, decay) do
    Nx.add(
      Nx.multiply(ema_tensor, decay),
      Nx.multiply(new_tensor, 1.0 - decay)
    )
  end

  # Map over nested param structure (handles maps and tuples)
  # Note: Check for Nx.Tensor first since is_map/1 returns true for structs
  defp map_nested(%Nx.Tensor{} = tensor, fun) do
    fun.(tensor)
  end

  defp map_nested(params, fun) when is_map(params) do
    Map.new(params, fn {k, v} ->
      {k, map_nested(v, fun)}
    end)
  end

  defp map_nested(params, fun) when is_tuple(params) do
    params
    |> Tuple.to_list()
    |> Enum.map(&map_nested(&1, fun))
    |> List.to_tuple()
  end

  defp map_nested(other, _fun), do: other

  # Map over two nested structures in parallel
  # Note: Check for Nx.Tensor first since is_map/1 returns true for structs
  defp map_nested_pair(%Nx.Tensor{} = t1, %Nx.Tensor{} = t2, fun) do
    fun.(t1, t2)
  end

  defp map_nested_pair(params1, params2, fun) when is_map(params1) and is_map(params2) do
    Map.new(params1, fn {k, v1} ->
      v2 = Map.fetch!(params2, k)
      {k, map_nested_pair(v1, v2, fun)}
    end)
  end

  defp map_nested_pair(params1, params2, fun) when is_tuple(params1) and is_tuple(params2) do
    list1 = Tuple.to_list(params1)
    list2 = Tuple.to_list(params2)

    Enum.zip(list1, list2)
    |> Enum.map(fn {v1, v2} -> map_nested_pair(v1, v2, fun) end)
    |> List.to_tuple()
  end

  defp map_nested_pair(other, _other2, _fun), do: other
end
