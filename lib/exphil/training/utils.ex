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

  @doc """
  Build an Axon model with EXLA graph compilation when available.

  Without `compiler: EXLA`, each `pred_fn.(params, input)` call re-traces
  all Axon layer callbacks (~700-2800ms for typical models). With graph
  compilation, XLA compiles the graph once and caches it (~2-4ms per call).

  Falls back gracefully to uncompiled build when EXLA is not loaded.

  ## Examples

      {init_fn, pred_fn} = Utils.build_compiled(model)
      {init_fn, pred_fn} = Utils.build_compiled(model, mode: :inference)
  """
  @spec build_compiled(Axon.t(), keyword()) :: {function(), function()}
  def build_compiled(model, opts \\ []) do
    opts =
      if Code.ensure_loaded?(EXLA) do
        Keyword.put_new(opts, :compiler, EXLA)
      else
        opts
      end

    Axon.build(model, opts)
  end
end
