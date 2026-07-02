defmodule ExPhil.NxSafe do
  @moduledoc """
  Bounds-checked wrappers for index-driven tensor operations.

  XLA's `gather` **clamps** out-of-bounds indices to the nearest valid index
  instead of raising. When indices and tensor come from different sources
  (dataset splits, cached embeddings, rollout minibatches), a size mismatch
  therefore produces silently corrupted data rather than an error — this is
  exactly how a stale embedding-cache entry masqueraded as a "scaling
  collapse" for four months (see GOTCHAS.md #51).

  ## Policy

  Use `NxSafe.take/3` wherever the index source and the tensor source can
  drift independently — anything crossing an IO/cache/dataset boundary:

    * dataset train/val splits
    * batch assembly from cached embedding tensors
    * PPO/rollout minibatch shuffling
    * any indices loaded from or derived from disk

  Structurally-bounded gathers do NOT need it (indices provably in range by
  construction): `argmax` outputs indexing the same axis they came from,
  constant index lists over fixed layouts, `iota`-derived indices.

  The check costs two scalar reductions per call — negligible next to the
  gather itself for data-sized tensors.
  """

  @doc """
  `Nx.take/3` that raises `ArgumentError` on out-of-bounds indices instead of
  silently clamping.

  Accepts indices as a list of integers or an integer `Nx.Tensor`.

  ## Options
    * `:axis` - axis to take along (default: 0)
    * `:label` - short description used in the error message to identify the
      call site (default: "NxSafe.take")
  """
  @spec take(Nx.Tensor.t(), [integer()] | Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def take(tensor, indices, opts \\ [])

  def take(tensor, indices, opts) when is_list(indices) do
    axis = Keyword.get(opts, :axis, 0)

    case indices do
      [] ->
        label = Keyword.get(opts, :label, "NxSafe.take")

        raise ArgumentError,
              "#{label}: empty index list — Nx cannot represent zero-size tensors; " <>
                "guard the call site instead of gathering nothing"

      _ ->
        {min_idx, max_idx} = Enum.min_max(indices)
        validate_bounds!(tensor, axis, min_idx, max_idx, opts)
        Nx.take(tensor, Nx.tensor(indices, type: :s64), axis: axis)
    end
  end

  def take(tensor, %Nx.Tensor{} = indices, opts) do
    axis = Keyword.get(opts, :axis, 0)
    min_idx = indices |> Nx.reduce_min() |> Nx.to_number()
    max_idx = indices |> Nx.reduce_max() |> Nx.to_number()
    validate_bounds!(tensor, axis, min_idx, max_idx, opts)
    Nx.take(tensor, indices, axis: axis)
  end

  defp validate_bounds!(tensor, axis, min_idx, max_idx, opts) do
    size = Nx.axis_size(tensor, axis)

    if min_idx < 0 or max_idx >= size do
      label = Keyword.get(opts, :label, "NxSafe.take")

      raise ArgumentError,
            "#{label}: indices out of bounds — index range #{min_idx}..#{max_idx} " <>
              "but tensor has #{size} entries on axis #{axis}. XLA gather would " <>
              "silently clamp these to valid rows, corrupting the result " <>
              "(see GOTCHAS.md #51). The tensor does not cover what the indices " <>
              "were built against (stale cache? size drift?)."
    end
  end
end
