defmodule ExPhil.Interp.Steering do
  @moduledoc """
  Activation steering by rank-1 projection removal (Tier-0 shield-lock A/B).

  A steering vector is a unit direction `v̂` in trunk-feature space, extracted
  offline by contrasting mean trunk activations between two behavioral
  regimes (see `scripts/extract_steer_vector.exs`). At inference the hook
  removes a fraction `alpha` of each feature vector's component along `v̂`:

      steer(x) = x - alpha * (x · v̂) v̂

  `alpha = 0` is the identity; `alpha = 1` is full projection out of the
  direction (rank-1 LEACE-style erasure without the whitening). Unlike the
  LEACE eraser hook, steering works on BOTH inference paths: as an `Axon.nx`
  layer between trunk and heads (windowed), and applied directly to the
  step-path features before `heads_predict_fn` (stateful) — the vector
  lives in the same `{hidden}` space either way.

  File format: `:erlang.term_to_binary(%{v: unit_tensor, meta: map})` where
  `v` is a `{hidden}` f32 tensor on `Nx.BinaryBackend`.
  """

  @doc """
  Load a steering vector file written by `scripts/extract_steer_vector.exs`.

  Returns `%{v: tensor, meta: map}` with `v` defensively re-normalized to
  unit length. Raises on zero-norm vectors or malformed files.
  """
  def load!(path) do
    %{v: %Nx.Tensor{} = v} = data = path |> File.read!() |> :erlang.binary_to_term()

    norm = v |> Nx.LinAlg.norm() |> Nx.to_number()

    if norm == 0 do
      raise ArgumentError, "steering vector has zero norm: #{path}"
    end

    %{data | v: Nx.divide(Nx.as_type(v, :f32), norm)}
  end

  @doc """
  Remove `alpha` of the component of `x` along unit vector `v`.

  `x` is `{batch, hidden}` (or `{hidden}`); `v` is `{hidden}`. `alpha = 0`
  returns `x` unchanged; `alpha = 1` leaves `x` orthogonal to `v`.
  Pure tensor math — safe inside `Axon.nx` closures (jitted) and eagerly on
  step-path features.
  """
  def steer(x, v, alpha) do
    v = Nx.as_type(v, Nx.type(x))
    # {batch} (or scalar for rank-1 x)
    proj = Nx.dot(x, v)
    Nx.subtract(x, Nx.multiply(alpha, Nx.multiply(Nx.new_axis(proj, -1), v)))
  end
end
