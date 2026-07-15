defmodule ExPhil.Interp.Erase do
  @moduledoc """
  Closed-form LEACE (LEAst-squares Concept Erasure) — Belrose et al. 2023.

  Fits an affine eraser that removes ALL linear information about a concept
  `Z` from representations `X`, with the least-squares-minimal change to X:

      x' = x − W⁺ P W (x − μ)      P = U Uᵀ,  U = colspace(W Σ_xz)

  where `W = Σ_xx^(-1/2)` whitens X. After erasure, no linear probe can
  recover Z from X' (guaranteed), while everything linearly independent of
  Z is preserved as much as possible.

  First use (interp P3/P6, 2026-07-14): erasing the "copy signal" — the
  trunk subspace that predicts the policy's PREVIOUS buttons — to cure the
  causal-confusion pathologies (shield-lock, jump suppression, dropped
  fair follow-ups) without amputating the whole prev-action channel the
  way test-time ablation does.

  Note: `Edifice.Interpretability.LEACE` is the trainable architecture;
  this is the exact closed form (no training loop). Worth upstreaming.
  """

  @doc """
  Fit an eraser from representations `x` `{n, d}` (f32) and concept
  labels `z` `{n, k}` (any numeric type; centered internally).

  Options:
    - `:shrinkage` - ridge added to Σ_xx diagonal for stable inversion
      (default 1.0e-4, relative to mean variance)
    - `:rank` - cap the erased subspace rank (default: full rank of Σ_xz,
      i.e. up to k)

  Returns `%{mu: {d}, a: {d, d}}` — apply as `x - (x - mu) @ a_T`… use
  `apply/2`.
  """
  def fit(x, z, opts \\ []) do
    shrink = Keyword.get(opts, :shrinkage, 1.0e-4)
    x = Nx.as_type(x, :f64)
    z = Nx.as_type(z, :f64)

    n = Nx.axis_size(x, 0)
    d = Nx.axis_size(x, 1)

    mu_x = Nx.mean(x, axes: [0])
    mu_z = Nx.mean(z, axes: [0])
    xc = Nx.subtract(x, mu_x)
    zc = Nx.subtract(z, mu_z)

    # The O(n·d²) covariance products run on the inputs' backend (put X on
    # EXLA for speed); the small {d,d} eigh/svd stage below runs on
    # BinaryBackend where LinAlg support is total.
    sigma_xx = Nx.dot(Nx.transpose(xc), xc) |> Nx.divide(n)

    sigma_xz = Nx.dot(Nx.transpose(xc), zc) |> Nx.divide(n)

    # Small-matrix stage (eigh/svd) runs on BinaryBackend with the DEFAULT
    # backend scoped to match — two hard-won lessons (2026-07-14, CLAUDE.md
    # "Script Logging"): EXLA compiles Nx.LinAlg's unrolled defn eigh/svd
    # for 20+ minutes on a 256x256 f64, and mixed backends crash because
    # ops create internal tensors (iotas) on the default backend.
    sigma_xx = Nx.backend_transfer(sigma_xx, Nx.BinaryBackend)
    sigma_xz = Nx.backend_transfer(sigma_xz, Nx.BinaryBackend)
    mu_x = Nx.backend_transfer(mu_x, Nx.BinaryBackend)

    prev_backend = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)

    result =
      try do
        fit_small_stage(sigma_xx, sigma_xz, mu_x, d, shrink, opts)
      after
        Nx.default_backend(prev_backend)
      end

    result
  end

  defp fit_small_stage(sigma_xx, sigma_xz, mu_x, d, shrink, opts) do
    # Ridge for numerical stability, scaled to the mean variance
    mean_var = Nx.take_diagonal(sigma_xx) |> Nx.mean()
    ridge = Nx.multiply(mean_var, shrink)
    sigma_xx = Nx.add(sigma_xx, Nx.multiply(Nx.eye(d, type: :f64), ridge))

    # Whitening via CHOLESKY (Σ = LLᵀ, W = L⁻¹, W⁺ = L) — exact,
    # non-iterative triangular ops. The original ZCA whitener (eigh-based
    # Σ^(-1/2)) failed on real 256-d GRU activations: BinaryBackend's
    # iterative eigh silently under-converges at eigenvalue spread ~1e6,
    # producing a non-whitener and an eraser that VIOLATED the guarantee
    # (cross-cov 0.26 → 3.74, 2026-07-14). Cholesky whitening keeps the
    # erasure guarantee (whitener-agnostic: cross-cov of erased X with Z
    # is zero for ANY valid W); the edit is minimal in the L-whitened
    # metric rather than exactly ZCA-minimal — an accepted deviation.
    l = Nx.LinAlg.cholesky(sigma_xx)
    eye = Nx.eye(d, type: :f64)
    w = Nx.LinAlg.triangular_solve(l, eye, lower: true)
    w_pinv = l

    # Whitened cross-covariance and its column space
    wxz = Nx.dot(w, sigma_xz)

    {u, s, _vt} = Nx.LinAlg.svd(wxz, full_matrices?: false)

    # Keep directions with non-negligible singular value (and optional cap)
    tol = Nx.multiply(Nx.reduce_max(s), 1.0e-6)
    keep = Nx.greater(s, tol) |> Nx.as_type(:f64)

    keep =
      case Keyword.get(opts, :rank) do
        nil -> keep
        r -> Nx.multiply(keep, Nx.less(Nx.iota(Nx.shape(s)), r) |> Nx.as_type(:f64))
      end

    # P = U diag(keep) Uᵀ  (projection onto kept concept directions)
    p = Nx.dot(Nx.multiply(u, Nx.reshape(keep, {1, :auto})), Nx.transpose(u))

    # Full eraser matrix A = W⁺ P W ; x' = x − (x − μ) Aᵀ
    a = w_pinv |> Nx.dot(p) |> Nx.dot(w)

    %{
      mu: Nx.as_type(mu_x, :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      a: Nx.as_type(a, :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      rank: Nx.sum(keep) |> Nx.to_number() |> trunc()
    }
  end

  @doc """
  Erase a concept from representations `{n, d}`: `x − (x − μ) Aᵀ`.
  """
  def erase(%{mu: mu, a: a}, x) do
    xc = Nx.subtract(x, mu)
    Nx.subtract(x, Nx.dot(xc, Nx.transpose(a)))
  end

  @doc """
  Verify erasure: after `apply/2`, a linear probe for `z` should be at
  chance. Returns the before/after balanced accuracy of
  `ExPhil.Interp.Probe` on the first column of `z` binarized — a quick
  smoke metric, not a full audit.
  """
  def verify(eraser, x, z, num_classes \\ 2) do
    alias ExPhil.Interp.Probe

    y = z |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.as_type(:s64)
    n = Nx.axis_size(x, 0)
    half = div(n, 2)

    slice = fn t, start, len -> Nx.slice_along_axis(t, start, len, axis: 0) end

    before_r =
      Probe.fit_eval(slice.(x, 0, half), slice.(y, 0, half), slice.(x, half, n - half), slice.(y, half, n - half), num_classes)

    xe = erase(eraser, x)

    after_r =
      Probe.fit_eval(slice.(xe, 0, half), slice.(y, 0, half), slice.(xe, half, n - half), slice.(y, half, n - half), num_classes)

    %{before: before_r.balanced_accuracy, after: after_r.balanced_accuracy}
  end
end
