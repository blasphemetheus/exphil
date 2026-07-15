defmodule ExPhil.Interp.Probe do
  @moduledoc """
  Linear probing harness (INTERP_ROADMAP Phase 1).

  Trains `Edifice.Interpretability.LinearProbe` models on frozen activations
  to test whether a feature is linearly decodable, and evaluates with
  balanced accuracy (mean per-class recall) — the drill's probe targets are
  heavily imbalanced (opp_knockdown ≈ 3%), where plain accuracy rewards
  predicting the majority class.

  Design choices:
  - **Split by replay, never by frame** — frames within a game are strongly
    correlated; a frame-level split leaks and inflates accuracy.
  - **Class-weighted cross-entropy** (inverse frequency) so rare-class
    signal isn't drowned.
  - **Z-score standardization** with train-set statistics only.
  - Rows labeled `-1` are masked out (e.g. tech_choice outside knockdown
    lifecycles).

  Training is full-batch gradient descent with momentum, JIT-compiled via
  EXLA — probes are a 256×k dense layer; hundreds of steps take seconds.
  """

  import Nx.Defn

  alias Edifice.Interpretability.LinearProbe
  alias ExPhil.Training.Utils

  @default_steps 300
  @default_lr 0.05
  @default_l2 1.0e-4

  @doc """
  Fit a probe on `{x, y}` and evaluate on `{x_eval, y_eval}`.

  ## Arguments
    - `x` / `x_eval` - `{n, d}` f32 activations
    - `y` / `y_eval` - `{n}` integer labels; `-1` rows are masked out
    - `num_classes` - number of classes (2 for binary features)

  ## Returns
    `%{balanced_accuracy, accuracy, majority_baseline, per_class_recall,
       n_train, n_eval, params}`
  """
  def fit_eval(x, y, x_eval, y_eval, num_classes, opts \\ []) do
    steps = Keyword.get(opts, :steps, @default_steps)
    lr = Keyword.get(opts, :lr, @default_lr)
    l2 = Keyword.get(opts, :l2, @default_l2)

    # Heavy row ops (gather/take, standardization) must NOT run on
    # BinaryBackend — its pure-Elixir single-threaded gather turns {32k,256}
    # takes into tens of minutes while the GPU idles (observed 2026-07-14).
    # Copy to the accelerator up front; scalars come back at the end anyway.
    to_dev = fn t -> Nx.backend_copy(t, EXLA.Backend) end
    {x, y, x_eval, y_eval} = {to_dev.(x), to_dev.(y), to_dev.(x_eval), to_dev.(y_eval)}

    {x, y} = mask_rows(x, y)
    {x_eval, y_eval} = mask_rows(x_eval, y_eval)

    n_train = Nx.axis_size(x, 0)
    n_eval = Nx.axis_size(x_eval, 0)

    if n_train == 0 or n_eval == 0 do
      %{
        balanced_accuracy: nil,
        accuracy: nil,
        majority_baseline: nil,
        per_class_recall: nil,
        n_train: n_train,
        n_eval: n_eval,
        params: nil
      }
    else
      # Standardize with train statistics
      mean = Nx.mean(x, axes: [0], keep_axes: true)
      std = Nx.standard_deviation(x, axes: [0], keep_axes: true) |> Nx.max(1.0e-6)
      xs = Nx.divide(Nx.subtract(x, mean), std)
      xs_eval = Nx.divide(Nx.subtract(x_eval, mean), std)

      # Inverse-frequency class weights from the train labels
      counts = class_counts(y, num_classes)
      weights = Nx.divide(n_train / num_classes, Nx.max(counts, 1))

      d = Nx.axis_size(x, 1)

      # Edifice probe (task: :regression → raw logits) supplies the model;
      # we train its single dense layer directly.
      _probe_model = LinearProbe.build(input_size: d, num_classes: num_classes, task: :regression)

      key = Nx.Random.key(42)
      {w, _} = Nx.Random.normal(key, 0.0, 0.01, shape: {d, num_classes}, type: :f32)
      b = Nx.broadcast(Nx.tensor(0.0, type: :f32), {num_classes})

      y_onehot = Nx.equal(Nx.new_axis(y, 1), Nx.iota({1, num_classes})) |> Nx.as_type(:f32)

      {w, b} = train_jit(xs, y_onehot, weights, w, b, steps, lr, l2)

      pred = predict_jit(w, b, xs_eval)
      per_class = per_class_recall(pred, y_eval, num_classes)

      recalls = per_class |> Enum.reject(&is_nil/1)

      # Guard: with <2 classes present in eval, "balanced accuracy" is
      # degenerate (predicting the only class scores 1.0). This is exactly
      # how the tech_choice=1.000 artifact slipped through on 2026-07-13 —
      # the tech dummy never teched, so the label had one class.
      balanced =
        if length(recalls) >= 2, do: Enum.sum(recalls) / length(recalls), else: nil

      %{
        balanced_accuracy: balanced,
        accuracy: Nx.mean(Nx.equal(pred, y_eval)) |> Nx.to_number(),
        majority_baseline: majority_baseline(y_eval, num_classes),
        per_class_recall: per_class,
        n_train: n_train,
        n_eval: n_eval,
        params: %{w: w, b: b, mean: mean, std: std}
      }
    end
  end

  @doc """
  Split a capture (from `ExPhil.Interp.Activations.capture/3`) into train and
  eval by replay index. `eval_replays` is a list of replay positions.
  """
  def split_by_replay(capture, eval_replays) do
    idx = capture.replay_index
    eval_mask = Enum.reduce(eval_replays, Nx.broadcast(0, Nx.shape(idx)), fn r, acc ->
      Nx.logical_or(acc, Nx.equal(idx, r))
    end)

    train_rows = mask_to_indices(Nx.logical_not(eval_mask))
    eval_rows = mask_to_indices(eval_mask)

    take = fn t, rows -> Nx.take(t, rows, axis: 0) end

    %{
      x_train: take.(capture.activations, train_rows),
      x_eval: take.(capture.activations, eval_rows),
      labels_train: Map.new(capture.labels, fn {k, v} -> {k, take.(v, train_rows)} end),
      labels_eval: Map.new(capture.labels, fn {k, v} -> {k, take.(v, eval_rows)} end)
    }
  end

  @doc """
  Run the standard probe suite over a split: all binary ground-truth
  features (2 classes) plus tech_choice and next_kd_choice (4 classes,
  masked to lifecycle/lookahead rows).

  Returns `%{feature => fit_eval_result}`.
  """
  def suite(split, opts \\ []) do
    binary = Enum.map(ExPhil.Interp.GroundTruth.binary_features(), &{&1, 2})
    multi = [{:tech_choice, 4}, {:next_kd_choice, 4}]

    Map.new(binary ++ multi, fn {feature, k} ->
      y_train = Map.fetch!(split.labels_train, feature) |> Nx.as_type(:s64)
      y_eval = Map.fetch!(split.labels_eval, feature) |> Nx.as_type(:s64)

      {feature, fit_eval(split.x_train, y_train, split.x_eval, y_eval, k, opts)}
    end)
  end

  @doc """
  v2 suite: memory-dependent targets that cannot be copied from the current
  frame (the v1 suite saturated — action-state targets leak from the raw
  embedding). `opp_knockdown` is retained as the known-leaky reference.
  """
  def suite_v2(split, opts \\ []) do
    targets = [
      {:time_since_kd_bucket, 5},
      {:frames_until_kd_bucket, 5},
      {:opp_damaged_recent, 2},
      {:opp_knockdown, 2}
    ]

    Map.new(targets, fn {feature, k} ->
      y_train = Map.fetch!(split.labels_train, feature) |> Nx.as_type(:s64)
      y_eval = Map.fetch!(split.labels_eval, feature) |> Nx.as_type(:s64)

      {feature, fit_eval(split.x_train, y_train, split.x_eval, y_eval, k, opts)}
    end)
  end

  @doc """
  Hewitt-Liang-style control: fit on SHUFFLED train labels, evaluate on true
  eval labels. A sound probe setup scores ~chance here; anything above
  chance means the probe itself (not the representation) is doing the work.
  """
  def shuffled_control(split, feature, num_classes, opts \\ []) do
    y_train = Map.fetch!(split.labels_train, feature) |> Nx.as_type(:s64)
    y_eval = Map.fetch!(split.labels_eval, feature) |> Nx.as_type(:s64)

    n = Nx.axis_size(y_train, 0)
    :rand.seed(:exsss, {7, 7, 7})
    perm = 0..(n - 1) |> Enum.shuffle() |> Nx.tensor(type: :s64)
    y_shuffled = Nx.take(y_train, perm)

    fit_eval(split.x_train, y_shuffled, split.x_eval, y_eval, num_classes, opts)
  end

  @doc """
  Decodability-vs-lead-time curve: probe `next_kd_choice` separately on rows
  restricted to each `frames_until_kd_bucket` value. Bucket 0 = entry frame
  (should be trivially decodable), rising buckets = earlier lead times. The
  instrument for both the P4 reaction-vs-prediction question and the v1
  next_kd=1.000 mystery.

  Returns `[{bucket, balanced_accuracy, n_eval}]`.
  """
  def lead_time_curve(split, opts \\ []) do
    restrict = fn labels, b ->
      y = Map.fetch!(labels, :next_kd_choice) |> Nx.as_type(:s64)
      bk = Map.fetch!(labels, :frames_until_kd_bucket) |> Nx.as_type(:s64)
      Nx.select(Nx.equal(bk, b), y, Nx.tensor(-1, type: :s64))
    end

    Enum.map(0..4, fn b ->
      r =
        fit_eval(
          split.x_train,
          restrict.(split.labels_train, b),
          split.x_eval,
          restrict.(split.labels_eval, b),
          4,
          opts
        )

      {b, r.balanced_accuracy, r.n_eval}
    end)
  end

  @doc """
  Mean balanced accuracy across suite results (nil-safe) — the single
  scalar used for the probe↔conversion correlation.
  """
  def mean_balanced_accuracy(suite_results) do
    scores =
      suite_results
      |> Map.values()
      |> Enum.map(& &1.balanced_accuracy)
      |> Enum.reject(&is_nil/1)

    Enum.sum(scores) / max(length(scores), 1)
  end

  # ============================================================================
  # Private — Nx internals
  # ============================================================================

  defp mask_rows(x, y) do
    keep = mask_to_indices(Nx.greater_equal(y, 0))

    if Nx.axis_size(keep, 0) == 0 do
      {Nx.broadcast(0.0, {0, Nx.axis_size(x, 1)}), Nx.broadcast(0, {0})}
    else
      {Nx.take(x, keep, axis: 0), Nx.take(y, keep, axis: 0)}
    end
  end

  defp mask_to_indices(mask) do
    n = Nx.axis_size(mask, 0)
    count = Nx.sum(mask) |> Nx.to_number()

    if count == 0 do
      Nx.broadcast(0, {0})
    else
      mask
      |> Nx.as_type(:s64)
      |> Nx.multiply(Nx.iota({n}) |> Nx.add(1))
      |> Nx.to_flat_list()
      |> Enum.filter(&(&1 > 0))
      |> Enum.map(&(&1 - 1))
      |> Nx.tensor(type: :s64)
    end
  end

  defp class_counts(y, num_classes) do
    Nx.equal(Nx.new_axis(y, 1), Nx.iota({1, num_classes}))
    |> Nx.sum(axes: [0])
    |> Nx.as_type(:f32)
  end

  defp majority_baseline(y_eval, num_classes) do
    counts = class_counts(y_eval, num_classes)
    # Balanced accuracy of always predicting the majority class:
    # recall = 1 for that class, 0 elsewhere → 1/num_present
    present = Nx.greater(counts, 0) |> Nx.sum() |> Nx.to_number()
    1.0 / max(present, 1)
  end

  defp per_class_recall(pred, y, num_classes) do
    Enum.map(0..(num_classes - 1), fn c ->
      in_class = Nx.equal(y, c)
      total = Nx.sum(in_class) |> Nx.to_number()

      if total == 0 do
        nil
      else
        hit = Nx.logical_and(in_class, Nx.equal(pred, c)) |> Nx.sum() |> Nx.to_number()
        hit / total
      end
    end)
  end

  defp train_jit(xs, y_onehot, weights, w, b, steps, lr, l2) do
    fun = fn xs, y_onehot, weights, w, b ->
      train_loop(xs, y_onehot, weights, w, b, steps: steps, lr: lr, l2: l2)
    end

    Nx.Defn.jit_apply(fun, [xs, y_onehot, weights, w, b], compiler: EXLA)
  end

  defp predict_jit(w, b, xs) do
    Nx.Defn.jit_apply(&predict_n/3, [w, b, xs], compiler: EXLA)
  end

  defnp predict_n(w, b, xs) do
    xs |> Nx.dot(w) |> Nx.add(b) |> Nx.argmax(axis: 1)
  end

  defnp train_loop(xs, y_onehot, weights, w0, b0, opts \\ []) do
    steps = opts[:steps]
    lr = opts[:lr]
    l2 = opts[:l2]

    {{w, b}, _} =
      while {{w = w0, b = b0}, {xs, y_onehot, weights, mw = w0 * 0.0, mb = b0 * 0.0}},
            _i <- 0..(steps - 1) do
        {gw, gb} = grad({w, b}, fn {w, b} -> ce_loss(w, b, xs, y_onehot, weights, l2) end)

        mw = 0.9 * mw + gw
        mb = 0.9 * mb + gb
        {{w - lr * mw, b - lr * mb}, {xs, y_onehot, weights, mw, mb}}
      end

    {w, b}
  end

  defnp ce_loss(w, b, xs, y_onehot, weights, l2) do
    logits = Nx.dot(xs, w) + b
    max_l = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = logits - max_l
    log_probs = shifted - Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))

    # Class-weighted NLL
    row_w = Nx.sum(y_onehot * Nx.reshape(weights, {1, :auto}), axes: [1])
    nll = -Nx.sum(log_probs * y_onehot, axes: [1])

    Nx.sum(nll * row_w) / Nx.sum(row_w) + l2 * Nx.sum(w * w)
  end
end
