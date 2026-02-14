defmodule ExPhil.Training.Utils do
  @moduledoc """
  Utility functions for training.
  """

  @doc """
  Ensures params are wrapped in Axon.ModelState.

  Axon's inference methods now require ModelState instead of plain maps.
  This helper provides backwards compatibility with existing checkpoints
  that store params as maps.

  ## Examples

      # Already ModelState - returned as-is
      iex> state = %Axon.ModelState{data: %{}, state: %{}}
      iex> Utils.ensure_model_state(state)
      %Axon.ModelState{data: %{}, state: %{}}

      # Plain map - wrapped in ModelState
      iex> Utils.ensure_model_state(%{"layer" => %{"kernel" => tensor}})
      %Axon.ModelState{data: %{"layer" => %{"kernel" => tensor}}, state: %{}}
  """
  @spec ensure_model_state(%Axon.ModelState{} | map()) :: %Axon.ModelState{}
  def ensure_model_state(%Axon.ModelState{} = state), do: state

  def ensure_model_state(map) when is_map(map) do
    %Axon.ModelState{data: map, state: %{}}
  end
end
