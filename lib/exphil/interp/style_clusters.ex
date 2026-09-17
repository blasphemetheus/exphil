defmodule ExPhil.Interp.StyleClusters do
  @moduledoc """
  STYLE_IDENTITY.md S3b: perceived players / styles for games the closed-set
  matcher could not name. K-means (Lloyd, k-means++ seeding) in the learned
  metric space; each cluster becomes a pseudo-tag `~cNN`.

  `purity/2` scores a clustering with tagged games: for each tag, the share
  of its games that fall in the tag's modal cluster. High purity means the
  clusters follow players; low purity means they follow something else
  (matchup, stage) — the report says which.
  """

  @doc "Fit k-means. Returns `%{centroids: Nx.Tensor {k, d}, assignments: [int], inertia: float}`."
  @spec fit([[float()]], pos_integer(), keyword()) :: map()
  def fit(vectors, k, opts \\ []) do
    iters = Keyword.get(opts, :iters, 50)
    seed = Keyword.get(opts, :seed, 3)
    x = Nx.tensor(vectors, type: :f32)
    {n, _d} = Nx.shape(x)
    key = Nx.Random.key(seed)
    cent = seed_pp(x, k, key)

    {cent, assign} =
      Enum.reduce(1..iters, {cent, nil}, fn _, {cent, prev} ->
        assign = nearest(x, cent)

        if prev != nil and Nx.to_number(Nx.all(Nx.equal(assign, prev))) == 1 do
          {cent, assign}
        else
          onehot = Nx.equal(Nx.new_axis(assign, 1), Nx.iota({1, k})) |> Nx.as_type(:f32)
          counts = Nx.sum(onehot, axes: [0]) |> Nx.new_axis(1)
          sums = Nx.dot(Nx.transpose(onehot), x)
          # empty clusters keep their old centroid
          mask = Nx.broadcast(Nx.greater(counts, 0), Nx.shape(cent))
          new = Nx.select(mask, Nx.divide(sums, Nx.max(counts, 1)), cent)
          {new, assign}
        end
      end)

    assign = assign || nearest(x, cent)
    d2 = sq_dists(x, cent)
    inertia = d2 |> Nx.take_along_axis(Nx.new_axis(assign, 1), axis: 1) |> Nx.sum() |> Nx.to_number()
    %{centroids: Nx.backend_copy(cent, Nx.BinaryBackend), assignments: Nx.to_flat_list(assign), inertia: inertia, n: n, k: k}
  end

  @doc "Cluster index (and distance) of each vector under a fitted model."
  @spec predict(map(), [[float()]]) :: [{non_neg_integer(), float()}]
  def predict(%{centroids: cent}, vectors) do
    x = Nx.tensor(vectors, type: :f32)
    d2 = sq_dists(x, cent)
    idx = Nx.argmin(d2, axis: 1)
    d = d2 |> Nx.take_along_axis(Nx.new_axis(idx, 1), axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.sqrt()
    Enum.zip(Nx.to_flat_list(idx), Nx.to_flat_list(d))
  end

  @doc """
  Purity of a clustering w.r.t. tagged games: `[{tag, cluster_index}]` ->
  weighted mean over tags of (games in the tag's modal cluster / games).
  """
  @spec purity([{String.t(), non_neg_integer()}]) :: %{purity: float(), tags: non_neg_integer(), clusters_used: non_neg_integer()}
  def purity(pairs) do
    by_tag = Enum.group_by(pairs, &elem(&1, 0), &elem(&1, 1))
    total = length(pairs)

    modal =
      Enum.map(by_tag, fn {_, cs} -> cs |> Enum.frequencies() |> Map.values() |> Enum.max() end)

    %{
      purity: if(total == 0, do: 0.0, else: Enum.sum(modal) / total),
      tags: map_size(by_tag),
      clusters_used: pairs |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> length()
    }
  end

  # k-means++ seeding
  defp seed_pp(x, k, key) do
    {n, _} = Nx.shape(x)
    {first, key} = Nx.Random.randint(key, 0, n)
    cent = Nx.take(x, Nx.reshape(first, {1}))

    {cent, _} =
      Enum.reduce(2..k//1, {cent, key}, fn _, {cent, key} ->
        d2 = sq_dists(x, cent) |> Nx.reduce_min(axes: [1])
        p = Nx.divide(d2, Nx.max(Nx.sum(d2), 1.0e-9))
        {u, key} = Nx.Random.uniform(key)
        idx = Nx.cumulative_sum(p) |> Nx.greater_equal(u) |> Nx.argmax()
        {Nx.concatenate([cent, Nx.take(x, Nx.reshape(idx, {1}))]), key}
      end)

    cent
  end

  defp nearest(x, cent), do: sq_dists(x, cent) |> Nx.argmin(axis: 1)

  defp sq_dists(x, cent) do
    xx = Nx.sum(Nx.multiply(x, x), axes: [1]) |> Nx.new_axis(1)
    cc = Nx.sum(Nx.multiply(cent, cent), axes: [1]) |> Nx.new_axis(0)
    Nx.max(Nx.subtract(Nx.add(xx, cc), Nx.multiply(2, Nx.dot(x, Nx.transpose(cent)))), 0.0)
  end
end
