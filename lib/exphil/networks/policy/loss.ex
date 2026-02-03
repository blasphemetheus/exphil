defmodule ExPhil.Networks.Policy.Loss do
  @moduledoc """
  Loss functions for policy network training.

  Provides cross-entropy losses for imitation learning with support for:
  - Label smoothing (reduces overconfidence)
  - Focal loss (focuses on hard/rare examples)
  - Per-bucket weighting (addresses neutral↔edge confusion)

  ## Usage

      # Standard imitation loss
      loss = Loss.imitation_loss(logits, targets)

      # With label smoothing and focal loss
      loss = Loss.imitation_loss(logits, targets,
        label_smoothing: 0.1,
        focal_loss: true,
        focal_gamma: 2.0
      )

  ## Loss Components

  The imitation loss combines:
  - **Button loss**: Binary cross-entropy for 8 independent button predictions
  - **Stick losses**: Categorical cross-entropy for discretized main/c-stick positions
  - **Shoulder loss**: Categorical cross-entropy for trigger pressure

  ## Focal Loss

  Focal loss down-weights easy examples: `(1 - p_t)^gamma * CE(p, y)`

  This helps with rare actions (Z, L, R buttons are pressed <2% of the time)
  by preventing the model from ignoring them in favor of easy negatives.

  ## See Also

  - `ExPhil.Networks.Policy` - Main policy module
  - `ExPhil.Networks.Policy.Sampling` - Action sampling
  """

  @doc """
  Compute policy loss (cross-entropy) for imitation learning.

  ## Parameters
    - `logits` - Map of logits from forward pass
    - `targets` - Map of target actions (ground truth from replay)

  ## Returns
    Total loss (sum of button BCE + categorical CE for sticks/shoulder)
  """
  @spec imitation_loss(map(), map()) :: Nx.Tensor.t()
  def imitation_loss(logits, targets) do
    imitation_loss(logits, targets, label_smoothing: 0.0)
  end

  @doc """
  Compute imitation loss with optional label smoothing and focal loss.

  Label smoothing prevents overconfidence by replacing hard targets with soft targets:
  - For categorical targets: target class gets (1-ε), others get ε/(num_classes-1)
  - For binary targets: target gets (1-ε), non-target gets ε

  Focal loss down-weights easy examples and focuses on hard/rare ones:
  - Formula: (1 - p_t)^gamma * CE(p, y)
  - Helps with rare actions like Z, L, R buttons (~2% frequency)

  ## Options
    - `:label_smoothing` - Smoothing factor ε (default: 0.0, typical: 0.1)
    - `:focal_loss` - Enable focal loss (default: false)
    - `:focal_gamma` - Focal loss gamma parameter (default: 2.0)
    - `:button_weight` - Multiply button loss to balance against stick losses (default: 1.0)
    - `:stick_edge_weight` - Weight edge buckets higher than center (default: nil)
  """
  @spec imitation_loss(map(), map(), keyword()) :: Nx.Tensor.t()
  def imitation_loss(logits, targets, opts) do
    label_smoothing = Keyword.get(opts, :label_smoothing, 0.0)
    focal_loss = Keyword.get(opts, :focal_loss, false)
    focal_gamma = Keyword.get(opts, :focal_gamma, 2.0)
    # Button weight: multiply button loss to balance against 5 categorical losses
    # Default 1.0 = no change; try 3.0-5.0 to boost button learning
    button_weight = Keyword.get(opts, :button_weight, 1.0)
    # Stick edge weight: weight edge buckets higher than center
    # nil = disabled, 2.0 = edges weighted 2x center
    stick_edge_weight = Keyword.get(opts, :stick_edge_weight, nil)

    # Choose loss functions based on focal_loss flag
    {button_loss_fn, cat_loss_fn} =
      if focal_loss do
        {
          fn logits, targets, smooth ->
            focal_binary_cross_entropy(logits, targets, smooth, focal_gamma)
          end,
          fn logits, targets, smooth ->
            focal_categorical_cross_entropy(logits, targets, smooth, focal_gamma)
          end
        }
      else
        {&binary_cross_entropy/3, &categorical_cross_entropy/3}
      end

    # Choose main stick loss function based on stick_edge_weight
    # Only apply edge weighting to main stick (most important for movement)
    main_stick_loss_fn =
      if stick_edge_weight && stick_edge_weight > 1.0 do
        fn logits, targets, smooth ->
          weighted_categorical_cross_entropy(logits, targets, smooth, stick_edge_weight)
        end
      else
        cat_loss_fn
      end

    # Button loss (binary cross-entropy with optional label smoothing + focal)
    # Apply button_weight to boost button loss relative to stick/shoulder losses
    button_loss =
      Nx.multiply(
        button_loss_fn.(logits.buttons, targets.buttons, label_smoothing),
        button_weight
      )

    # Main stick losses (with optional edge weighting)
    main_x_loss = main_stick_loss_fn.(logits.main_x, targets.main_x, label_smoothing)
    main_y_loss = main_stick_loss_fn.(logits.main_y, targets.main_y, label_smoothing)

    # C-stick and shoulder losses (standard categorical CE)
    c_x_loss = cat_loss_fn.(logits.c_x, targets.c_x, label_smoothing)
    c_y_loss = cat_loss_fn.(logits.c_y, targets.c_y, label_smoothing)
    shoulder_loss = cat_loss_fn.(logits.shoulder, targets.shoulder, label_smoothing)

    # Combine losses
    Nx.add(
      button_loss,
      Nx.add(
        main_x_loss,
        Nx.add(
          main_y_loss,
          Nx.add(
            c_x_loss,
            Nx.add(c_y_loss, shoulder_loss)
          )
        )
      )
    )
  end

  @doc """
  Binary cross-entropy loss for buttons with optional label smoothing.

  With label smoothing ε > 0:
  - Target 1 becomes (1-ε)
  - Target 0 becomes ε

  This prevents the model from becoming overconfident on button predictions.
  """
  @spec binary_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def binary_cross_entropy(logits, targets) do
    binary_cross_entropy(logits, targets, 0.0)
  end

  @spec binary_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def binary_cross_entropy(logits, targets, label_smoothing) do
    # Apply label smoothing to targets
    # Smoothed targets: t_smooth = t * (1 - ε) + (1 - t) * ε = t * (1 - 2ε) + ε
    smoothed_targets =
      if label_smoothing > 0.0 do
        Nx.add(
          Nx.multiply(targets, 1.0 - 2.0 * label_smoothing),
          label_smoothing
        )
      else
        targets
      end

    # Numerically stable BCE
    # loss = max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
    max_val = Nx.max(logits, 0)
    abs_logits = Nx.abs(logits)

    loss = Nx.subtract(max_val, Nx.multiply(logits, smoothed_targets))
    loss = Nx.add(loss, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))

    Nx.mean(loss)
  end

  @doc """
  Categorical cross-entropy loss for discretized sticks/shoulder with optional label smoothing.

  With label smoothing ε > 0:
  - Target class gets probability (1-ε)
  - Other classes share ε equally: ε/(num_classes-1)

  This encourages the model to be less overconfident on stick/shoulder predictions.
  """
  @spec categorical_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def categorical_cross_entropy(logits, targets) do
    categorical_cross_entropy(logits, targets, 0.0)
  end

  @spec categorical_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def categorical_cross_entropy(logits, targets, label_smoothing) do
    # targets are indices, logits are [batch, num_classes]
    # Use log_softmax for numerical stability
    log_probs = log_softmax(logits)

    # Gather the log probability of the target class
    # targets: [batch], log_probs: [batch, num_classes]
    batch_size = Nx.axis_size(logits, 0)
    num_classes = Nx.axis_size(logits, 1)

    # Create one-hot targets
    targets_one_hot =
      Nx.equal(
        Nx.iota({batch_size, num_classes}, axis: 1),
        Nx.reshape(targets, {batch_size, 1})
      )

    # Apply label smoothing if enabled
    # Smoothed one-hot: (1-ε) for target class, ε/(n-1) for others
    smoothed_targets =
      if label_smoothing > 0.0 do
        # off_value = ε / (n-1)
        # on_value = 1 - ε
        off_value = label_smoothing / (num_classes - 1)
        on_value = 1.0 - label_smoothing

        # Start with uniform ε/(n-1), then add (1-ε - ε/(n-1)) to target class
        # = off_value + targets_one_hot * (on_value - off_value)
        Nx.add(
          off_value,
          Nx.multiply(targets_one_hot, on_value - off_value)
        )
      else
        targets_one_hot
      end

    # Cross-entropy with soft targets: -sum(p * log_q)
    nll = Nx.negate(Nx.sum(Nx.multiply(log_probs, smoothed_targets), axes: [1]))
    Nx.mean(nll)
  end

  @doc """
  Categorical cross-entropy with per-bucket weighting for stick inputs.

  Edge buckets (0, num_buckets-1) are weighted higher than center buckets,
  with linear interpolation between. This addresses the neutral↔far confusion
  where the model defaults to neutral and misses important edge positions.

  ## Arguments
    - `logits` - Raw model output [batch, num_classes]
    - `targets` - Target bucket indices [batch]
    - `label_smoothing` - Label smoothing factor (0.0 = none)
    - `edge_weight` - Weight for edge buckets (e.g., 2.0 = 2x weight for edges)

  ## Weight Calculation
  For a 17-bucket system (0-16, center at 8):
    - Bucket 8 (center): weight = 1.0
    - Bucket 0 or 16 (edges): weight = edge_weight
    - Intermediate buckets: linear interpolation

  Formula: weight[i] = 1.0 + (edge_weight - 1.0) * |i - center| / center
  """
  @spec weighted_categorical_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) ::
          Nx.Tensor.t()
  def weighted_categorical_cross_entropy(logits, targets, label_smoothing, edge_weight) do
    # targets are indices, logits are [batch, num_classes]
    log_probs = log_softmax(logits)

    batch_size = Nx.axis_size(logits, 0)
    num_classes = Nx.axis_size(logits, 1)
    center = div(num_classes - 1, 2)

    # Create one-hot targets
    targets_one_hot =
      Nx.equal(
        Nx.iota({batch_size, num_classes}, axis: 1),
        Nx.reshape(targets, {batch_size, 1})
      )

    # Apply label smoothing if enabled
    smoothed_targets =
      if label_smoothing > 0.0 do
        off_value = label_smoothing / (num_classes - 1)
        on_value = 1.0 - label_smoothing

        Nx.add(
          off_value,
          Nx.multiply(targets_one_hot, on_value - off_value)
        )
      else
        targets_one_hot
      end

    # Compute per-sample weights based on target bucket
    # weight = 1.0 + (edge_weight - 1.0) * distance_from_center / center
    # targets: [batch], we need to compute weight for each target
    distance_from_center = Nx.abs(Nx.subtract(targets, center))

    # Normalize by center to get 0.0 at center, 1.0 at edges
    normalized_distance = Nx.divide(distance_from_center, center)

    # Interpolate: 1.0 at center, edge_weight at edges
    sample_weights = Nx.add(1.0, Nx.multiply(edge_weight - 1.0, normalized_distance))

    # Cross-entropy with soft targets: -sum(p * log_q)
    nll = Nx.negate(Nx.sum(Nx.multiply(log_probs, smoothed_targets), axes: [1]))

    # Apply per-sample weights
    weighted_nll = Nx.multiply(nll, sample_weights)

    Nx.mean(weighted_nll)
  end

  @doc """
  Focal binary cross-entropy loss for buttons.

  Focal loss down-weights easy examples and focuses on hard ones:
  - Formula: (1 - p_t)^gamma * BCE(p, y)
  - p_t is the probability assigned to the correct class
  - gamma=2.0 is typical, higher values focus more on hard examples

  This helps with rare button presses (Z, L, R are used <2% of the time)
  by preventing the model from ignoring them in favor of easy negatives.
  """
  @spec focal_binary_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) ::
          Nx.Tensor.t()
  def focal_binary_cross_entropy(logits, targets, label_smoothing, gamma) do
    # Apply label smoothing to targets
    smoothed_targets =
      if label_smoothing > 0.0 do
        Nx.add(
          Nx.multiply(targets, 1.0 - 2.0 * label_smoothing),
          label_smoothing
        )
      else
        targets
      end

    # Compute probabilities via sigmoid
    probs = Nx.sigmoid(logits)

    # p_t = p if y=1, (1-p) if y=0
    # Using: p_t = p * y + (1-p) * (1-y) = y*(2p-1) + (1-p)
    p_t =
      Nx.add(
        Nx.multiply(smoothed_targets, Nx.subtract(Nx.multiply(probs, 2), 1)),
        Nx.subtract(1, probs)
      )

    # Focal weight: (1 - p_t)^gamma
    focal_weight = Nx.pow(Nx.subtract(1.0, p_t), gamma)

    # Standard BCE (numerically stable)
    max_val = Nx.max(logits, 0)
    abs_logits = Nx.abs(logits)
    bce = Nx.subtract(max_val, Nx.multiply(logits, smoothed_targets))
    bce = Nx.add(bce, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))

    # Apply focal weight
    focal_bce = Nx.multiply(focal_weight, bce)
    Nx.mean(focal_bce)
  end

  @doc """
  Focal categorical cross-entropy loss for sticks/shoulder.

  Focal loss down-weights easy examples and focuses on hard ones:
  - Formula: (1 - p_t)^gamma * CE(p, y)
  - p_t is the probability assigned to the correct class
  - gamma=2.0 is typical, higher values focus more on hard examples

  This helps with rare stick positions and shoulder values.
  """
  @spec focal_categorical_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) ::
          Nx.Tensor.t()
  def focal_categorical_cross_entropy(logits, targets, label_smoothing, gamma) do
    # Compute log probabilities and probabilities
    log_probs = log_softmax(logits)
    probs = Nx.exp(log_probs)

    # Get batch size and number of classes
    {batch_size, num_classes} = Nx.shape(logits)

    # Convert targets to one-hot
    targets_one_hot =
      Nx.equal(
        Nx.reshape(targets, {batch_size, 1}),
        Nx.iota({1, num_classes})
      )
      |> Nx.as_type(:f32)

    # Apply label smoothing
    smoothed_targets =
      if label_smoothing > 0.0 do
        off_value = label_smoothing / (num_classes - 1)
        on_value = 1.0 - label_smoothing

        Nx.add(
          off_value,
          Nx.multiply(targets_one_hot, on_value - off_value)
        )
      else
        targets_one_hot
      end

    # p_t = sum(p * y) for each sample (probability of correct class)
    p_t = Nx.sum(Nx.multiply(probs, smoothed_targets), axes: [1])

    # Focal weight: (1 - p_t)^gamma
    focal_weight = Nx.pow(Nx.subtract(1.0, p_t), gamma)

    # Standard cross-entropy: -sum(y * log_p)
    ce = Nx.negate(Nx.sum(Nx.multiply(log_probs, smoothed_targets), axes: [1]))

    # Apply focal weight
    focal_ce = Nx.multiply(focal_weight, ce)
    Nx.mean(focal_ce)
  end

  @doc """
  Numerically stable log-softmax computation.
  """
  @spec log_softmax(Nx.Tensor.t()) :: Nx.Tensor.t()
  def log_softmax(logits) do
    max_val = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_val)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    Nx.subtract(shifted, log_sum_exp)
  end
end
