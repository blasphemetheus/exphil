defmodule ExPhil.Interp.StyleMetric do
  @moduledoc """
  STYLE_IDENTITY.md step 2b: a learned linear metric over z-scored
  fingerprint features.

  Fits a projection `W` (d x k) by Neighbourhood Component Analysis
  (Goldberger et al. 2005): for every anchor game, the softmax over
  negative projected distances to the other games in the batch should put
  its mass on the SAME player's games. Linear, so it cannot overfit the way
  a deep embedding would on a few thousand tagged games, and the learned
  weights are readable (which features carry identity).

  Train on one set of players, evaluate on players never seen — that is
  what the open-set use (hashed erickfm games) needs.
  """

  import Nx.Defn

  @doc """
  Fit the projection. `vectors` is a list of equal-length float lists
  (already z-scored), `labels` a parallel list of integer player ids.

  Options: `:dim` (k, default 24), `:steps` (default 400), `:batch`
  (default 512), `:lr` (default 5.0e-3), `:seed`, `:l2` (default 1.0e-4).
  Returns `%{w: Nx.Tensor, dim: k, loss: [float]}`.
  """
  @spec fit([[float()]], [integer()], keyword()) :: %{w: Nx.Tensor.t(), dim: pos_integer(), loss: [float()]}
  def fit(vectors, labels, opts \\ []) do
    k = Keyword.get(opts, :dim, 24)
    steps = Keyword.get(opts, :steps, 400)
    batch = min(Keyword.get(opts, :batch, 512), length(vectors))
    lr = Keyword.get(opts, :lr, 5.0e-3)
    seed = Keyword.get(opts, :seed, 7)
    l2 = Keyword.get(opts, :l2, 1.0e-4)

    x = Nx.tensor(vectors, type: :f32)
    y = Nx.tensor(labels, type: :s64)
    {n, d} = Nx.shape(x)

    key = Nx.Random.key(seed)
    {w, key} = Nx.Random.normal(key, 0.0, :math.sqrt(1.0 / d), shape: {d, k}, type: :f32)

    {init_fn, update_fn} = Polaris.Optimizers.adam(learning_rate: lr)
    opt_state = init_fn.(%{w: w})

    # Everything the gradient touches is an ARGUMENT, not a closure (GOTCHA
    # #3: captured tensors on EXLA vs the evaluator backend crash grad).
    step = Nx.Defn.jit(&train_step/6)
    l2_t = Nx.tensor(l2, type: :f32)

    {w, _, _, losses} =
      Enum.reduce(1..steps, {w, opt_state, key, []}, fn _, {w, opt_state, key, losses} ->
        {idx, key} = Nx.Random.shuffle(key, Nx.iota({n}))
        idx = Nx.slice(idx, [0], [batch])
        xb = Nx.take(x, idx)
        yb = Nx.take(y, idx)
        {w, opt_state, loss} = step.(w, opt_state, xb, yb, l2_t, update_fn)
        {w, opt_state, key, [Nx.to_number(loss) | losses]}
      end)

    %{w: Nx.backend_copy(w, Nx.BinaryBackend), dim: k, loss: Enum.reverse(losses)}
  end

  defn train_step(w, opt_state, xb, yb, l2, update_fn) do
    {loss, grad} = value_and_grad(w, fn w -> nca_loss(w, xb, yb, l2) end)
    {updates, opt_state} = update_fn.(%{w: grad}, opt_state, %{w: w})
    {Polaris.Updates.apply_updates(%{w: w}, updates).w, opt_state, loss}
  end

  defn nca_loss(w, x, y, l2) do
    z = Nx.dot(x, w)
    # squared pairwise distances via the expansion, clamped at 0
    sq = Nx.sum(z * z, axes: [1])
    d2 = Nx.new_axis(sq, 1) + Nx.new_axis(sq, 0) - 2 * Nx.dot(z, Nx.transpose(z))
    d2 = Nx.max(d2, 0.0)
    b = Nx.axis_size(x, 0)
    eye = Nx.eye(b)
    # softmax over j != i of exp(-d2)
    logits = -d2 - eye * 1.0e9
    # stable log-softmax: subtract the row max before exponentiating
    # (all-large distances underflowed exp() to 0 -> log(0) -> NaN, 09-17)
    row_max = Nx.reduce_max(logits, axes: [1], keep_axes: true)
    logp = logits - row_max - Nx.log(Nx.sum(Nx.exp(logits - row_max), axes: [1], keep_axes: true))
    same = Nx.equal(Nx.new_axis(y, 1), Nx.new_axis(y, 0)) |> Nx.as_type(:f32)
    same = same * (1.0 - eye)
    has_pos = Nx.sum(same, axes: [1]) > 0
    p_same = Nx.sum(Nx.exp(logp) * same, axes: [1])
    per_anchor = -Nx.log(p_same + 1.0e-9)
    n_valid = Nx.max(Nx.sum(Nx.as_type(has_pos, :f32)), 1.0)
    Nx.sum(Nx.select(has_pos, per_anchor, 0.0)) / n_valid + l2 * Nx.sum(w * w)
  end

  @doc "Project one z-scored vector (list) through a fitted `W`."
  @spec project(%{w: Nx.Tensor.t()}, [float()]) :: [float()]
  def project(%{w: w}, vec) do
    Nx.tensor(vec, type: :f32, backend: Nx.BinaryBackend) |> Nx.dot(w) |> Nx.to_flat_list()
  end

  @doc "Feature importance: L2 norm of each input row of `W` (parallel to `StyleFingerprint.keys/0`)."
  @spec importance(%{w: Nx.Tensor.t()}) :: [float()]
  def importance(%{w: w}), do: w |> Nx.multiply(w) |> Nx.sum(axes: [1]) |> Nx.sqrt() |> Nx.to_flat_list()

  @doc "Serialize / restore a fitted metric."
  def save!(metric, path), do: File.write!(path, :erlang.term_to_binary(%{w: Nx.to_flat_list(metric.w), shape: Nx.shape(metric.w), dim: metric.dim}))

  def load!(path) do
    %{w: flat, shape: shape, dim: dim} = path |> File.read!() |> :erlang.binary_to_term()
    %{w: Nx.tensor(flat, type: :f32, backend: Nx.BinaryBackend) |> Nx.reshape(shape), dim: dim, loss: []}
  end
end
