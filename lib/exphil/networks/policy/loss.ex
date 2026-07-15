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

  # Apply reduction to per-sample loss tensor
  # Follows Axon.Losses convention: :mean (default), :sum, :none
  defp apply_reduction(loss, :mean), do: Nx.mean(loss)
  defp apply_reduction(loss, :sum), do: Nx.sum(loss)
  defp apply_reduction(loss, :none), do: loss
  defp apply_reduction(loss, nil), do: Nx.mean(loss)

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
    - `:button_pos_weight` - Per-button positive class weights [8] tensor (default: nil)
      Scales loss for pressed buttons by per-button factors (inverse frequency).
      Fixes mode collapse where model predicts all buttons as "not pressed".
    - `:stick_edge_weight` - Weight edge buckets higher than center (default: nil)
    - `:entropy_weight` - Entropy regularization weight (default: 0.0)
      Adds -weight * entropy(predictions) to encourage diverse outputs.
      Prevents mode collapse by penalizing collapsed output distributions.
    - `:frame_weights` - Per-frame loss weights tensor [batch] (default: nil)
      Action frames get higher weight than neutral frames to prevent
      the model from collapsing to "predict neutral" on large datasets.
  """
  @spec imitation_loss(map(), map(), keyword()) :: Nx.Tensor.t()
  def imitation_loss(logits, targets, opts) do
    # Loss math (and therefore its BACKWARD under autodiff) always runs in
    # f32, whatever precision the network computes in. Grad-localizer
    # forensics (2026-07-14, GRAD_DETONATION step 344,769) caught NaN
    # gradients born in the buttons-BCE backward under bf16 compute with a
    # FINITE forward loss — the root cause of every "endemic NaN" drill
    # death that week (mixed precision didn't help because it also runs the
    # backward in bf16). Casting here, at the loss math entry point,
    # protects every caller: training loss_and_grad builders, eval loss,
    # and future call sites. Cost is negligible (loss ops are tiny next to
    # the network).
    logits = Map.new(logits, fn {k, v} -> {k, Nx.as_type(v, :f32)} end)

    # Clamp logits away from the exp-overflow cliff (f32 exp overflows at
    # |x| ~ 88.7). Crime-scene autopsy 2026-07-14 (scene_step284154): the
    # fatal batch carried button logits at -89.9, and the NaN only occurs
    # in the training-compiled program — XLA's algebraic simplifier can
    # rewrite the stable BCE form (and its adjoint) into exp(+x) variants
    # during fusion, so stability depends on compilation luck. At |x|=60,
    # sigmoid saturates to within 1e-26 and the true gradient is ~0: the
    # clamp is mathematically inert but removes the cliff under ANY
    # compiler rewrite. This closed the week-long "endemic NaN" saga
    # (six runs: bf16, f32-master, f32-loss, w16, seed, LR all refuted;
    # per-layer localizer + crime-scene replay named it).
    #
    # min∘max instead of Nx.clip: Nx 0.11's autodiff produces INCORRECT
    # gradients for Nx.clip in this expression (wrong on EXLA and the pure
    # evaluator alike — upstream Nx bug; minimal reproducer in
    # scripts/nx_clip_grad_repro.exs). A clip-based clamp silently killed a
    # training run from step 0 (garbage grads → logits blown to the rails →
    # zero-grad saturation, 2026-07-14). min/max grads verified correct
    # against the analytic reference.
    logits = Map.new(logits, fn {k, v} -> {k, Nx.min(Nx.max(v, -60.0), 60.0)} end)

    label_smoothing = Keyword.get(opts, :label_smoothing, 0.0)
    focal_loss = Keyword.get(opts, :focal_loss, false)
    focal_gamma = Keyword.get(opts, :focal_gamma, 2.0)
    entropy_weight = Keyword.get(opts, :entropy_weight, 0.0)
    frame_weights = Keyword.get(opts, :frame_weights, nil)
    head_normalize = Keyword.get(opts, :head_normalize, false)
    # Button weight: multiply button loss to balance against 5 categorical losses
    # Default 1.0 = no change; try 3.0-5.0 to boost button learning
    button_weight = Keyword.get(opts, :button_weight, 1.0)
    # Per-button positive class weights: [8] tensor or nil
    # Scales BCE loss for pressed buttons by per-button factors
    # Must be on BinaryBackend to avoid EXLA/Expr mismatch when captured in defn closure
    button_pos_weight =
      case Keyword.get(opts, :button_pos_weight, nil) do
        %Nx.Tensor{} = t -> Nx.backend_copy(t, Nx.BinaryBackend)
        other -> other
      end
    # Stick edge weight: weight edge buckets higher than center
    # nil = disabled, 2.0 = edges weighted 2x center
    stick_edge_weight = Keyword.get(opts, :stick_edge_weight, nil)

    # Choose loss functions based on focal_loss flag
    # Buttons are NEVER label-smoothed (pass 0.0 regardless of the option):
    # smoothing × pos_weight moves the BCE optimum above the press threshold
    # for rare buttons — see compute_button_loss_per_sample for the math.
    {button_loss_fn, cat_loss_fn} =
      if focal_loss do
        {
          fn logits, targets, _smooth ->
            focal_binary_cross_entropy(logits, targets, 0.0, focal_gamma,
              pos_weight: button_pos_weight
            )
          end,
          fn logits, targets, smooth ->
            focal_categorical_cross_entropy(logits, targets, smooth, focal_gamma)
          end
        }
      else
        {
          fn logits, targets, _smooth ->
            binary_cross_entropy(logits, targets, 0.0, pos_weight: button_pos_weight)
          end,
          &categorical_cross_entropy/3
        }
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

    # Compute per-sample losses (reduction: :none) for frame weighting
    # Then combine, weight, and reduce
    use_frame_weights = frame_weights != nil

    total_loss =
      if use_frame_weights do
        # Per-sample path: get per-sample loss from each head, weight, then mean
        btn_ps = compute_button_loss_per_sample(logits.buttons, targets.buttons, label_smoothing,
          focal_loss, focal_gamma, button_pos_weight)
        btn_ps = Nx.multiply(btn_ps, button_weight)

        mx_ps = compute_cat_loss_per_sample(logits.main_x, targets.main_x, label_smoothing, stick_edge_weight)
        my_ps = compute_cat_loss_per_sample(logits.main_y, targets.main_y, label_smoothing, stick_edge_weight)
        cx_ps = compute_cat_loss_per_sample(logits.c_x, targets.c_x, label_smoothing, nil)
        cy_ps = compute_cat_loss_per_sample(logits.c_y, targets.c_y, label_smoothing, nil)
        sh_ps = compute_cat_loss_per_sample(logits.shoulder, targets.shoulder, label_smoothing, nil)

        # Per-head normalization: normalize each head's per-sample loss by its
        # stop_gradient mean so each head contributes ~1.0 to the total gradient.
        # Prevents easy heads (buttons=0.4) from being ignored while hard heads (mx=1.8) dominate.
        {btn_ps, mx_ps, my_ps, cx_ps, cy_ps, sh_ps} =
          if head_normalize do
            normalize_per_sample = fn ps ->
              # Clamp denominator to min 0.1 — caps max amplification at 10x
              head_mean = Nx.mean(ps) |> Nx.max(0.1)
              Nx.divide(ps, Nx.Defn.Kernel.stop_grad(head_mean))
            end
            {normalize_per_sample.(btn_ps), normalize_per_sample.(mx_ps),
             normalize_per_sample.(my_ps), normalize_per_sample.(cx_ps),
             normalize_per_sample.(cy_ps), normalize_per_sample.(sh_ps)}
          else
            {btn_ps, mx_ps, my_ps, cx_ps, cy_ps, sh_ps}
          end

        # Total per-sample loss: {batch}
        total_per_sample = btn_ps |> Nx.add(mx_ps) |> Nx.add(my_ps) |> Nx.add(cx_ps) |> Nx.add(cy_ps) |> Nx.add(sh_ps)

        # Apply frame weights and compute weighted mean
        weighted = Nx.multiply(total_per_sample, frame_weights)
        Nx.sum(weighted) |> Nx.divide(Nx.sum(frame_weights))
      else
        # Standard path: mean reduction per head, sum across heads
        button_loss =
          Nx.multiply(
            button_loss_fn.(logits.buttons, targets.buttons, label_smoothing),
            button_weight
          )

        main_x_loss = main_stick_loss_fn.(logits.main_x, targets.main_x, label_smoothing)
        main_y_loss = main_stick_loss_fn.(logits.main_y, targets.main_y, label_smoothing)
        c_x_loss = cat_loss_fn.(logits.c_x, targets.c_x, label_smoothing)
        c_y_loss = cat_loss_fn.(logits.c_y, targets.c_y, label_smoothing)
        shoulder_loss = cat_loss_fn.(logits.shoulder, targets.shoulder, label_smoothing)

        if head_normalize do
          # Normalize each head loss by stop_gradient(loss) so each contributes ~1.0
          # Clamp denominator to min 0.1 — caps max amplification at 10x
          normalize = fn loss ->
            Nx.divide(loss, Nx.max(Nx.Defn.Kernel.stop_grad(loss), 0.1))
          end
          normalize.(button_loss)
          |> Nx.add(normalize.(main_x_loss))
          |> Nx.add(normalize.(main_y_loss))
          |> Nx.add(normalize.(c_x_loss))
          |> Nx.add(normalize.(c_y_loss))
          |> Nx.add(normalize.(shoulder_loss))
        else
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
      end

    # Entropy regularization: penalize collapsed output distributions
    # Encourages the model to maintain diverse predictions, preventing mode collapse
    # H(p) = -sum(p * log(p)), we SUBTRACT entropy_weight * H to maximize entropy
    if entropy_weight > 0.0 do
      # Button entropy: sigmoid → per-element binary entropy
      btn_probs = Nx.sigmoid(logits.buttons)
      btn_entropy = Nx.negate(
        Nx.add(
          Nx.multiply(btn_probs, Nx.log(Nx.max(btn_probs, 1.0e-7))),
          Nx.multiply(Nx.subtract(1.0, btn_probs), Nx.log(Nx.max(Nx.subtract(1.0, btn_probs), 1.0e-7)))
        )
      ) |> Nx.mean()

      # Stick entropy: softmax → categorical entropy per head
      stick_entropy = Enum.map([logits.main_x, logits.main_y, logits.c_x, logits.c_y, logits.shoulder], fn head_logits ->
        probs = Nx.exp(log_softmax(head_logits))
        Nx.negate(Nx.sum(Nx.multiply(probs, Nx.log(Nx.max(probs, 1.0e-7))), axes: [-1])) |> Nx.mean()
      end) |> Enum.reduce(&Nx.add/2)

      total_entropy = Nx.add(btn_entropy, stick_entropy)

      # Subtract weighted entropy from loss (lower loss = higher entropy = more diverse)
      Nx.subtract(total_loss, Nx.multiply(entropy_weight, total_entropy))
    else
      total_loss
    end
  end

  # Per-sample loss helpers for frame weighting
  # Returns {batch} tensor (one loss value per sample)

  defp compute_button_loss_per_sample(logits, targets, _label_smoothing, focal_loss, focal_gamma, pos_weight) do
    # NEVER smooth the button targets: smoothing composes catastrophically
    # with per-button pos_weight. A soft target ε on a never-pressed button,
    # scaled by pos_weight w, moves the BCE optimum to p* = wε/(wε + 1-ε) —
    # for w=30, ε=0.1 that's p*=0.77, ABOVE the press threshold: the loss
    # minimum instructs the model to hold rare buttons (taunt at w=30,
    # Z-grab at w=12) on every frame. Observed live as constant taunts and
    # shine-grabs; offline as training 'flatlining' at the (pathological)
    # loss floor. Categorical heads keep smoothing — softmax renormalizes.
    smoothed = targets

    max_val = Nx.max(logits, 0)
    abs_logits = Nx.abs(logits)

    bce = if pos_weight do
      scale = Nx.add(1.0, Nx.multiply(Nx.subtract(pos_weight, 1.0), smoothed))
      base = Nx.add(max_val, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))
      Nx.subtract(Nx.multiply(base, scale), Nx.multiply(pos_weight, Nx.multiply(smoothed, logits)))
    else
      loss = Nx.subtract(max_val, Nx.multiply(logits, smoothed))
      Nx.add(loss, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))
    end

    per_sample = if focal_loss do
      probs = Nx.sigmoid(logits)
      p_t = Nx.add(Nx.multiply(smoothed, Nx.subtract(Nx.multiply(probs, 2), 1)), Nx.subtract(1, probs))
      focal_weight = Nx.pow(Nx.subtract(1.0, p_t), focal_gamma)
      Nx.mean(Nx.multiply(focal_weight, bce), axes: [-1])
    else
      Nx.mean(bce, axes: [-1])  # mean over 8 buttons → {batch}
    end

    per_sample
  end

  defp compute_cat_loss_per_sample(logits, targets, label_smoothing, edge_weight) do
    log_probs = log_softmax(logits)
    batch_size = Nx.axis_size(logits, 0)
    num_classes = Nx.axis_size(logits, 1)

    targets_one_hot = Nx.equal(
      Nx.iota({batch_size, num_classes}, axis: 1),
      Nx.reshape(targets, {batch_size, 1})
    )

    smoothed = if label_smoothing > 0.0 do
      off_value = label_smoothing / (num_classes - 1)
      on_value = 1.0 - label_smoothing
      Nx.add(off_value, Nx.multiply(targets_one_hot, on_value - off_value))
    else
      targets_one_hot
    end

    nll = Nx.negate(Nx.sum(Nx.multiply(log_probs, smoothed), axes: [1]))  # {batch}

    if edge_weight && edge_weight > 1.0 do
      center = div(num_classes - 1, 2)
      distance = Nx.abs(Nx.subtract(targets, center))
      normalized = Nx.divide(distance, center)
      sample_weights = Nx.add(1.0, Nx.multiply(edge_weight - 1.0, normalized))
      Nx.multiply(nll, sample_weights)
    else
      nll
    end
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
    binary_cross_entropy(logits, targets, 0.0, [])
  end

  @spec binary_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float(), keyword()) :: Nx.Tensor.t()
  def binary_cross_entropy(logits, targets, label_smoothing, opts \\ []) do
    pos_weight = Keyword.get(opts, :pos_weight, nil)
    reduction = Keyword.get(opts, :reduction, :mean)

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

    per_sample_loss =
      if pos_weight do
        max_val = Nx.max(logits, 0)
        abs_logits = Nx.abs(logits)
        base = Nx.add(max_val, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))
        scale = Nx.add(1.0, Nx.multiply(Nx.subtract(pos_weight, 1.0), smoothed_targets))
        Nx.subtract(Nx.multiply(base, scale), Nx.multiply(pos_weight, Nx.multiply(smoothed_targets, logits)))
      else
        max_val = Nx.max(logits, 0)
        abs_logits = Nx.abs(logits)
        loss = Nx.subtract(max_val, Nx.multiply(logits, smoothed_targets))
        Nx.add(loss, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))
      end

    apply_reduction(per_sample_loss, reduction)
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

  @spec categorical_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float(), keyword()) :: Nx.Tensor.t()
  def categorical_cross_entropy(logits, targets, label_smoothing, opts \\ []) do
    reduction = Keyword.get(opts, :reduction, :mean)

    log_probs = log_softmax(logits)

    batch_size = Nx.axis_size(logits, 0)
    num_classes = Nx.axis_size(logits, 1)

    targets_one_hot =
      Nx.equal(
        Nx.iota({batch_size, num_classes}, axis: 1),
        Nx.reshape(targets, {batch_size, 1})
      )

    smoothed_targets =
      if label_smoothing > 0.0 do
        off_value = label_smoothing / (num_classes - 1)
        on_value = 1.0 - label_smoothing
        Nx.add(off_value, Nx.multiply(targets_one_hot, on_value - off_value))
      else
        targets_one_hot
      end

    per_sample = Nx.negate(Nx.sum(Nx.multiply(log_probs, smoothed_targets), axes: [1]))
    apply_reduction(per_sample, reduction)
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

    apply_reduction(weighted_nll, :mean)
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
  @spec focal_binary_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float(), float(), keyword()) ::
          Nx.Tensor.t()
  def focal_binary_cross_entropy(logits, targets, label_smoothing, gamma, opts \\ []) do
    pos_weight = Keyword.get(opts, :pos_weight, nil)
    reduction = Keyword.get(opts, :reduction, :mean)

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

    if pos_weight do
      # Weighted focal BCE: focal_weight * [(1+(w-1)*t) * base - w*t*x]
      max_val = Nx.max(logits, 0)
      abs_logits = Nx.abs(logits)
      base = Nx.add(max_val, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))

      scale = Nx.add(1.0, Nx.multiply(Nx.subtract(pos_weight, 1.0), smoothed_targets))
      bce = Nx.subtract(Nx.multiply(base, scale), Nx.multiply(pos_weight, Nx.multiply(smoothed_targets, logits)))
      apply_reduction(Nx.multiply(focal_weight, bce), reduction)
    else
      # Standard focal BCE
      max_val = Nx.max(logits, 0)
      abs_logits = Nx.abs(logits)
      bce = Nx.subtract(max_val, Nx.multiply(logits, smoothed_targets))
      bce = Nx.add(bce, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))
      apply_reduction(Nx.multiply(focal_weight, bce), reduction)
    end
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
    apply_reduction(focal_ce, :mean)
  end

  @doc """
  Compute per-button press rates for predictions and targets.

  Returns a map with `:predicted` and `:actual` keys, each containing
  a list of 8 floats (press rate per button). Use to detect mode collapse:
  if predicted rates are near 0 but actual rates are >0, the model has collapsed.

  Button order: A, B, X, Y, Z, L, R, D-Up
  """
  @spec button_press_rates(Nx.Tensor.t(), Nx.Tensor.t()) :: %{
          predicted: [float()],
          actual: [float()]
        }
  def button_press_rates(button_logits, button_targets) do
    probs = Nx.sigmoid(button_logits)
    predicted = probs |> Nx.greater(0.5) |> Nx.mean(axes: [0]) |> Nx.to_flat_list()
    actual = button_targets |> Nx.mean(axes: [0]) |> Nx.to_flat_list()
    %{predicted: predicted, actual: actual}
  end

  @doc """
  Compute button positive class weights from training data.

  Uses sqrt((1 - press_rate) / press_rate) instead of the raw inverse frequency.
  Raw inverse frequency (1/rate) produces extreme weights (50-100x) for rare buttons
  which causes 3-5x over-prediction. The sqrt dampens this while still preventing
  mode collapse on rare buttons.

  Empirically validated: sqrt weights produce ~1-2x pred/actual ratios after 5 epochs,
  vs 3-8x with raw inverse frequency. Buttons like L (12% press rate) calibrate
  nearly perfectly (pred=15.6% vs actual=14.1%).

  ## Parameters
    - `button_targets` - All button labels [N, 8] binary
    - `max_weight` - Cap to prevent extreme weights (default: 30.0)
  """
  @spec compute_pos_weights(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def compute_pos_weights(button_targets, max_weight \\ 30.0) do
    press_rate = Nx.mean(button_targets, axes: [0])
    compute_pos_weights_from_rates(press_rate, max_weight)
  end

  @doc """
  Compute button positive class weights from pre-computed press rates.

  Takes a [8] tensor of press rates (0.0 to 1.0) directly, avoiding
  the need to access the full button labels tensor (which may be on GPU).
  """
  @spec compute_pos_weights_from_rates(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def compute_pos_weights_from_rates(press_rate, max_weight \\ 30.0) do
    press_rate = Nx.max(press_rate, 1.0e-4)
    weights = Nx.sqrt(Nx.divide(Nx.subtract(1.0, press_rate), press_rate))
    weights |> Nx.min(max_weight) |> Nx.max(1.0)
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
