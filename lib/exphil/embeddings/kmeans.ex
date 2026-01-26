defmodule ExPhil.Embeddings.KMeans do
  @moduledoc """
  K-means clustering for stick discretization.

  Instead of uniform bucket discretization, K-means clustering on actual
  stick positions from replays creates non-uniform buckets that better capture:
  - Cardinal/diagonal concentrations (most inputs near 0, 0.5, 1)
  - Deadzone awareness (cluster centers avoid ~0.2875 analog deadzone)
  - Character-specific patterns

  ## Usage

      # Fit K-means to stick data
      centers = KMeans.fit(stick_positions, k: 21, max_iters: 100)

      # Use centers for discretization
      bucket = KMeans.discretize(value, centers)

      # Convert back to continuous
      value = KMeans.undiscretize(bucket, centers)

  ## Research Background

  slippi-ai found that 21 K-means clusters outperformed 16 uniform buckets:
  - ~5% improvement in stick prediction accuracy
  - Better on rare but important inputs (wavedash angles, shield drops)
  """

  @doc """
  Fit K-means clustering to 1D data points.

  ## Options
    - `:k` - Number of clusters (default: 21)
    - `:max_iters` - Maximum iterations (default: 100)
    - `:tol` - Convergence tolerance (default: 1.0e-4)
    - `:seed` - Random seed for initialization (default: 42)
    - `:progress_fn` - Optional callback `fn iter, max_iters, diff -> :ok end`
    - `:init_progress_fn` - Optional callback `fn step, total -> :ok end` for init phase

  ## Returns
    Sorted tensor of cluster centers, shape {k}
  """
  @spec fit(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def fit(data, opts \\ []) do
    k = Keyword.get(opts, :k, 21)
    max_iters = Keyword.get(opts, :max_iters, 100)
    tol = Keyword.get(opts, :tol, 1.0e-4)
    seed = Keyword.get(opts, :seed, 42)
    progress_fn = Keyword.get(opts, :progress_fn)
    init_progress_fn = Keyword.get(opts, :init_progress_fn)

    # Ensure data is 1D float tensor
    data = data |> Nx.flatten() |> Nx.as_type(:f32)

    # Initialize centers using K-means++ for better convergence
    centers = kmeans_plusplus_init(data, k, seed, init_progress_fn)

    # Run K-means iterations
    centers = iterate(data, centers, max_iters, tol, progress_fn: progress_fn)

    # Sort centers for consistent ordering
    Nx.sort(centers)
  end

  @doc """
  Initialize cluster centers using K-means++ algorithm.

  K-means++ chooses initial centers with probability proportional to
  squared distance from nearest existing center, leading to better
  convergence than random initialization.
  """
  @spec kmeans_plusplus_init(Nx.Tensor.t(), non_neg_integer(), integer(), function() | nil) ::
          Nx.Tensor.t()
  def kmeans_plusplus_init(data, k, seed, progress_fn \\ nil) do
    n = Nx.axis_size(data, 0)
    key = Nx.Random.key(seed)

    # Choose first center randomly
    {idx, key} = Nx.Random.randint(key, 0, n, shape: {})
    first_center = data |> Nx.slice([Nx.to_number(idx)], [1]) |> Nx.squeeze()

    # Choose remaining centers
    centers_list = [Nx.to_number(first_center)]
    if progress_fn, do: progress_fn.(1, k)

    {centers_list, _key} =
      Enum.reduce(1..(k - 1), {centers_list, key}, fn i, {centers_list, key} ->
        centers = Nx.tensor(centers_list, type: :f32)

        # Compute squared distances to nearest center
        distances = compute_min_distances(data, centers)

        # Sample proportional to squared distance
        total = Nx.sum(distances) |> Nx.to_number()

        probs =
          if total > 0 do
            Nx.divide(distances, total + 1.0e-10)
          else
            Nx.broadcast(1.0 / n, {n})
          end

        # Sample next center
        {new_center_idx, key} = sample_weighted(key, probs)
        new_center = data |> Nx.slice([new_center_idx], [1]) |> Nx.squeeze() |> Nx.to_number()

        if progress_fn, do: progress_fn.(i + 1, k)

        {centers_list ++ [new_center], key}
      end)

    Nx.tensor(centers_list, type: :f32)
  end

  # Compute minimum squared distance from each point to nearest center
  defp compute_min_distances(data, centers) do
    # data: {n}, centers: {k}
    # Expand for broadcasting: data {n, 1} - centers {1, k} = {n, k}
    data_exp = Nx.reshape(data, {:auto, 1})
    centers_exp = Nx.reshape(centers, {1, :auto})

    squared_dists = Nx.pow(Nx.subtract(data_exp, centers_exp), 2)
    Nx.reduce_min(squared_dists, axes: [1])
  end

  # Sample index weighted by probabilities
  defp sample_weighted(key, probs) do
    {u, key} = Nx.Random.uniform(key, shape: {})
    u_num = Nx.to_number(u)
    cumsum = Nx.cumulative_sum(probs) |> Nx.to_flat_list()

    # Find first index where cumsum >= u
    idx = Enum.find_index(cumsum, fn c -> c >= u_num end) || length(cumsum) - 1
    {idx, key}
  end

  @doc """
  Run K-means iterations until convergence or max_iters.

  ## Options
    - `:progress_fn` - Optional callback `fn iter, max_iters, diff -> :ok end` for progress
  """
  @spec iterate(Nx.Tensor.t(), Nx.Tensor.t(), non_neg_integer(), float(), keyword()) ::
          Nx.Tensor.t()
  def iterate(data, centers, max_iters, tol, opts \\ []) do
    progress_fn = Keyword.get(opts, :progress_fn)

    {final_centers, _iter} =
      Enum.reduce_while(1..max_iters, {centers, 0}, fn iter, {centers, _} ->
        new_centers = update_centers(data, centers)

        # Check convergence
        diff = Nx.abs(Nx.subtract(new_centers, centers)) |> Nx.reduce_max() |> Nx.to_number()

        # Report progress if callback provided
        if progress_fn, do: progress_fn.(iter, max_iters, diff)

        if diff < tol do
          {:halt, {new_centers, iter}}
        else
          {:cont, {new_centers, iter}}
        end
      end)

    final_centers
  end

  @doc """
  Single K-means update step: assign points to clusters, recompute centers.
  """
  @spec update_centers(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def update_centers(data, centers) do
    k = Nx.axis_size(centers, 0)

    # Assign each point to nearest center
    assignments = assign_to_clusters(data, centers)

    # Compute new centers as mean of assigned points
    new_centers =
      Enum.map(0..(k - 1), fn cluster_idx ->
        mask = Nx.equal(assignments, cluster_idx)
        count = Nx.sum(mask) |> Nx.to_number()

        if count > 0 do
          sum = Nx.sum(Nx.multiply(data, mask)) |> Nx.to_number()
          sum / count
        else
          # Keep old center if no points assigned
          centers |> Nx.slice([cluster_idx], [1]) |> Nx.reshape({}) |> Nx.to_number()
        end
      end)

    Nx.tensor(new_centers, type: :f32)
  end

  @doc """
  Assign each data point to the nearest cluster center.

  ## Returns
    Tensor of cluster indices, shape {n}
  """
  @spec assign_to_clusters(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def assign_to_clusters(data, centers) do
    # data: {n}, centers: {k}
    data_exp = Nx.reshape(data, {:auto, 1})
    centers_exp = Nx.reshape(centers, {1, :auto})

    # Squared distances: {n, k}
    squared_dists = Nx.pow(Nx.subtract(data_exp, centers_exp), 2)

    # Argmin for each point
    Nx.argmin(squared_dists, axis: 1)
  end

  @doc """
  Discretize a continuous value [0, 1] to cluster index.

  ## Parameters
    - `value` - Continuous value in [0, 1]
    - `centers` - Sorted tensor of cluster centers

  ## Returns
    Cluster index (0 to k-1)
  """
  @spec discretize(float() | Nx.Tensor.t(), Nx.Tensor.t()) :: non_neg_integer()
  def discretize(value, centers) when is_number(value) do
    value = max(0.0, min(1.0, value))
    distances = Nx.abs(Nx.subtract(centers, value))
    Nx.argmin(distances) |> Nx.to_number()
  end

  def discretize(value, centers) when is_struct(value, Nx.Tensor) do
    value = Nx.clip(value, 0.0, 1.0)
    distances = Nx.abs(Nx.subtract(centers, value))
    Nx.argmin(distances) |> Nx.to_number()
  end

  @doc """
  Discretize a batch of values to cluster indices.

  ## Parameters
    - `values` - Tensor of values, shape {n}
    - `centers` - Sorted tensor of cluster centers, shape {k}

  ## Returns
    Tensor of cluster indices, shape {n}
  """
  @spec discretize_batch(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def discretize_batch(values, centers) do
    values = Nx.clip(values, 0.0, 1.0)
    assign_to_clusters(values, centers)
  end

  @doc """
  Convert a cluster index back to continuous value (the cluster center).

  ## Parameters
    - `index` - Cluster index (0 to k-1)
    - `centers` - Sorted tensor of cluster centers

  ## Returns
    Cluster center value in [0, 1]
  """
  @spec undiscretize(non_neg_integer(), Nx.Tensor.t()) :: float()
  def undiscretize(index, centers) do
    centers
    |> Nx.slice([index], [1])
    |> Nx.reshape({})
    |> Nx.to_number()
  end

  @doc """
  Get the number of clusters (k) from a centers tensor.
  """
  @spec k(Nx.Tensor.t()) :: non_neg_integer()
  def k(centers) do
    Nx.axis_size(centers, 0)
  end

  @doc """
  Save cluster centers to a file.
  """
  @spec save(Nx.Tensor.t(), String.t()) :: :ok | {:error, term()}
  def save(centers, path) do
    binary = Nx.serialize(centers)
    File.write(path, binary)
  end

  @doc """
  Load cluster centers from a file.
  """
  @spec load(String.t()) :: {:ok, Nx.Tensor.t()} | {:error, term()}
  def load(path) do
    case File.read(path) do
      {:ok, binary} -> {:ok, Nx.deserialize(binary)}
      error -> error
    end
  end

  @doc """
  Load cluster centers, returning nil if file doesn't exist.
  """
  @spec load!(String.t()) :: Nx.Tensor.t() | nil
  def load!(path) do
    case load(path) do
      {:ok, centers} -> centers
      {:error, _} -> nil
    end
  end

  @doc """
  Generate default cluster centers using uniform spacing.

  This is a fallback when K-means has not been trained. Uses the same
  spacing as uniform bucket discretization for compatibility.

  ## Parameters
    - `k` - Number of clusters

  ## Returns
    Tensor of uniformly spaced centers in [0, 1]
  """
  @spec default_centers(non_neg_integer()) :: Nx.Tensor.t()
  def default_centers(k) do
    # Place centers at bucket midpoints, same as uniform discretization
    Enum.map(0..(k - 1), fn i -> i / (k - 1) end)
    |> Nx.tensor(type: :f32)
  end
end
