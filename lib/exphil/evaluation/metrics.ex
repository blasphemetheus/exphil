defmodule ExPhil.Evaluation.Metrics do
  @moduledoc """
  Reusable metric computation functions for model evaluation.

  These functions compute accuracy, calibration, and other metrics
  from model predictions and targets. They are used by the eval script
  and can be used in training callbacks or other analysis tools.

  ## Usage

      alias ExPhil.Evaluation.Metrics

      # Compute accuracy from logits and targets
      acc = Metrics.accuracy(logits, targets)

      # Compute top-k accuracy
      top3_acc = Metrics.top_k_accuracy(logits, targets, 3)

      # Track calibration across batches
      calibration = Metrics.empty_calibration_bins()
      calibration = Metrics.update_calibration(calibration, probs, targets)

  """

  @button_labels [:a, :b, :x, :y, :z, :l, :r, :d_up]

  @doc """
  Returns the standard button labels in order.
  """
  @spec button_labels() :: [atom()]
  def button_labels, do: @button_labels

  @doc """
  Compute accuracy from logits and targets.

  Handles both sparse indices and one-hot encoded targets.

  ## Parameters
    - `logits` - Model output logits, shape {batch, num_classes}
    - `targets` - Target labels, shape {batch} or {batch, num_classes}

  ## Returns
    Accuracy as a float in [0, 1]
  """
  @spec accuracy(Nx.Tensor.t(), Nx.Tensor.t()) :: float()
  def accuracy(logits, targets) do
    predictions = Nx.argmax(logits, axis: -1)
    target_indices = to_indices(targets)
    correct = Nx.equal(predictions, target_indices)
    Nx.mean(correct) |> Nx.to_number()
  end

  @doc """
  Compute top-k accuracy from logits and targets.

  A prediction is correct if the true label is in the top k predictions.

  ## Parameters
    - `logits` - Model output logits, shape {batch, num_classes}
    - `targets` - Target labels, shape {batch} or {batch, num_classes}
    - `k` - Number of top predictions to consider

  ## Returns
    Top-k accuracy as a float in [0, 1]
  """
  @spec top_k_accuracy(Nx.Tensor.t(), Nx.Tensor.t(), pos_integer()) :: float()
  def top_k_accuracy(logits, targets, k) do
    {_, top_k_indices} = Nx.top_k(logits, k: k)
    target_indices = to_indices(targets)
    expanded_targets = Nx.reshape(target_indices, {:auto, 1})
    matches = Nx.equal(top_k_indices, expanded_targets)
    any_match = Nx.any(matches, axes: [-1])
    Nx.mean(any_match) |> Nx.to_number()
  end

  @doc """
  Compute button accuracy from logits and binary targets.

  Uses sigmoid activation and 0.5 threshold.

  ## Parameters
    - `logits` - Model output logits, shape {batch, 8}
    - `targets` - Binary targets, shape {batch, 8}

  ## Returns
    Average accuracy across all buttons as a float in [0, 1]
  """
  @spec button_accuracy(Nx.Tensor.t(), Nx.Tensor.t()) :: float()
  def button_accuracy(logits, targets) do
    probs = Nx.sigmoid(logits)
    predictions = Nx.greater(probs, 0.5)
    correct = Nx.equal(predictions, targets)
    Nx.mean(correct) |> Nx.to_number()
  end

  @doc """
  Compute per-button accuracy.

  ## Parameters
    - `logits` - Model output logits, shape {batch, 8}
    - `targets` - Binary targets, shape {batch, 8}

  ## Returns
    Map of button label to accuracy, e.g., %{a: 0.95, b: 0.92, ...}
  """
  @spec per_button_accuracy(Nx.Tensor.t(), Nx.Tensor.t()) :: %{atom() => float()}
  def per_button_accuracy(logits, targets) do
    probs = Nx.sigmoid(logits)
    predictions = Nx.greater(probs, 0.5)
    correct = Nx.equal(predictions, targets)

    @button_labels
    |> Enum.with_index()
    |> Enum.map(fn {label, idx} ->
      col_correct = Nx.slice_along_axis(correct, idx, 1, axis: -1)
      acc = Nx.mean(col_correct) |> Nx.to_number()
      {label, acc}
    end)
    |> Map.new()
  end

  @doc """
  Compute button press rates for predictions and actuals.

  ## Parameters
    - `logits` - Model output logits, shape {batch, 8}
    - `targets` - Binary targets, shape {batch, 8}

  ## Returns
    Tuple of {predicted_rates, actual_rates}, each a map of button -> rate
  """
  @spec button_rates(Nx.Tensor.t(), Nx.Tensor.t()) :: {%{atom() => float()}, %{atom() => float()}}
  def button_rates(logits, targets) do
    probs = Nx.sigmoid(logits)
    predictions = Nx.greater(probs, 0.5)

    predicted_rates =
      @button_labels
      |> Enum.with_index()
      |> Enum.map(fn {label, idx} ->
        col = Nx.slice_along_axis(predictions, idx, 1, axis: -1)
        rate = Nx.mean(col) |> Nx.to_number()
        {label, rate}
      end)
      |> Map.new()

    actual_rates =
      @button_labels
      |> Enum.with_index()
      |> Enum.map(fn {label, idx} ->
        col = Nx.slice_along_axis(targets, idx, 1, axis: -1)
        rate = Nx.mean(col) |> Nx.to_number()
        {label, rate}
      end)
      |> Map.new()

    {predicted_rates, actual_rates}
  end

  @doc """
  Update stick confusion matrix with batch predictions.

  Only tracks errors (predicted != actual) and skips neutral→neutral.

  ## Parameters
    - `confusion` - Existing confusion map of {pred, actual} -> count
    - `logits` - Model output logits, shape {batch, num_buckets}
    - `targets` - Target labels, shape {batch} or {batch, num_buckets}
    - `axis_buckets` - Number of buckets (e.g., 16 or 20)

  ## Returns
    Updated confusion map
  """
  @spec update_stick_confusion(map(), Nx.Tensor.t(), Nx.Tensor.t(), non_neg_integer()) :: map()
  def update_stick_confusion(confusion, logits, targets, axis_buckets) do
    predictions = Nx.argmax(logits, axis: -1)
    target_indices = to_indices(targets)
    pred_flat = Nx.to_flat_list(predictions)
    target_flat = Nx.to_flat_list(target_indices)

    neutral = div(axis_buckets, 2)

    Enum.zip(pred_flat, target_flat)
    |> Enum.reduce(confusion, fn {pred, actual}, acc ->
      if pred != actual and (pred != neutral or actual != neutral) do
        key = {pred, actual}
        Map.update(acc, key, 1, &(&1 + 1))
      else
        acc
      end
    end)
  end

  @doc """
  Create empty calibration bins (10 bins, 0-100% confidence).

  Each bin tracks {correct_count, total_count}.
  """
  @spec empty_calibration_bins() :: map()
  def empty_calibration_bins do
    Map.new(0..9, fn i -> {i, {0, 0}} end)
  end

  @doc """
  Update calibration bins with batch predictions.

  Bins predictions by confidence (0-10%, 10-20%, ..., 90-100%)
  and tracks accuracy within each bin.

  ## Parameters
    - `calibration_bins` - Existing bins from empty_calibration_bins/0
    - `probs` - Softmax probabilities, shape {batch, num_classes}
    - `targets` - Target labels, shape {batch} or {batch, num_classes}

  ## Returns
    Updated calibration bins
  """
  @spec update_calibration(map(), Nx.Tensor.t(), Nx.Tensor.t()) :: map()
  def update_calibration(calibration_bins, probs, targets) do
    max_probs = Nx.reduce_max(probs, axes: [-1]) |> Nx.to_flat_list()
    predictions = Nx.argmax(probs, axis: -1) |> Nx.to_flat_list()
    target_indices = to_indices(targets) |> Nx.to_flat_list()

    Enum.zip([max_probs, predictions, target_indices])
    |> Enum.reduce(calibration_bins, fn {conf, pred, actual}, bins ->
      bin_idx = min(floor(conf * 10), 9)
      correct = if pred == actual, do: 1, else: 0
      {prev_correct, prev_total} = Map.get(bins, bin_idx, {0, 0})
      Map.put(bins, bin_idx, {prev_correct + correct, prev_total + 1})
    end)
  end

  @doc """
  Compute Expected Calibration Error (ECE) from calibration bins.

  ECE is the weighted average of |accuracy - confidence| across bins.
  Lower is better. A perfectly calibrated model has ECE = 0.

  ## Parameters
    - `calibration_bins` - Bins from update_calibration/3

  ## Returns
    ECE as a float in [0, 1]
  """
  @spec expected_calibration_error(map()) :: float()
  def expected_calibration_error(calibration_bins) do
    total_samples = calibration_bins
    |> Map.values()
    |> Enum.map(fn {_, total} -> total end)
    |> Enum.sum()

    if total_samples == 0 do
      0.0
    else
      calibration_bins
      |> Enum.map(fn {bin_idx, {correct, total}} ->
        if total > 0 do
          accuracy = correct / total
          confidence = (bin_idx * 10 + 5) / 100  # Bin midpoint
          weight = total / total_samples
          weight * abs(accuracy - confidence)
        else
          0.0
        end
      end)
      |> Enum.sum()
    end
  end

  @doc """
  Compute softmax probabilities from logits.

  ## Parameters
    - `logits` - Raw model output, shape {batch, num_classes}

  ## Returns
    Softmax probabilities, shape {batch, num_classes}
  """
  @spec softmax(Nx.Tensor.t()) :: Nx.Tensor.t()
  def softmax(logits) do
    shifted = Nx.subtract(logits, Nx.reduce_max(logits, axes: [-1], keep_axes: true))
    exp = Nx.exp(shifted)
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
  end

  @doc """
  Compute average confidence (max probability) from logits.

  ## Parameters
    - `logits` - Raw model output, shape {batch, num_classes}

  ## Returns
    Average max probability as a float in [0, 1]
  """
  @spec avg_confidence(Nx.Tensor.t()) :: float()
  def avg_confidence(logits) do
    probs = softmax(logits)
    max_probs = Nx.reduce_max(probs, axes: [-1])
    Nx.mean(max_probs) |> Nx.to_number()
  end

  @doc """
  Compute prediction entropy (uncertainty measure).

  Higher entropy = more uncertain predictions.

  ## Parameters
    - `logits` - Raw model output, shape {batch, num_classes}

  ## Returns
    Average entropy across batch
  """
  @spec avg_entropy(Nx.Tensor.t()) :: float()
  def avg_entropy(logits) do
    probs = softmax(logits)
    # Entropy = -sum(p * log(p)), with p clamped to avoid log(0)
    clamped = Nx.max(probs, 1.0e-10)
    entropy = Nx.negate(Nx.sum(Nx.multiply(probs, Nx.log(clamped)), axes: [-1]))
    Nx.mean(entropy) |> Nx.to_number()
  end

  @doc """
  Compute random baseline accuracy for stick predictions.

  ## Parameters
    - `num_buckets` - Number of buckets (e.g., 17 for uniform, 21 for K-means)

  ## Returns
    Random accuracy as 1/num_buckets
  """
  @spec random_baseline(non_neg_integer()) :: float()
  def random_baseline(num_buckets) do
    1.0 / num_buckets
  end

  @doc """
  Format bucket index as direction string.

  ## Parameters
    - `bucket` - Bucket index
    - `num_buckets` - Total number of buckets

  ## Returns
    Direction string like "neutral", "neg", "far_neg", "pos", "far_pos"
  """
  @spec format_bucket(non_neg_integer(), non_neg_integer()) :: String.t()
  def format_bucket(bucket, num_buckets) do
    neutral = div(num_buckets, 2)
    cond do
      bucket == neutral -> "neutral"
      bucket < neutral - 2 -> "far_neg"
      bucket < neutral -> "neg"
      bucket > neutral + 2 -> "far_pos"
      bucket > neutral -> "pos"
      true -> "~neutral"
    end
  end

  @doc """
  Merge accumulated metrics from multiple batches.

  ## Parameters
    - `metrics1` - First metrics map
    - `metrics2` - Second metrics map (or batch metrics)

  ## Returns
    Merged metrics map with summed values
  """
  @spec merge_metrics(map(), map()) :: map()
  def merge_metrics(metrics1, metrics2) when map_size(metrics1) == 0, do: metrics2
  def merge_metrics(metrics1, metrics2) do
    Map.merge(metrics1, metrics2, fn _k, v1, v2 -> v1 + v2 end)
  end

  @doc """
  Average accumulated metrics by dividing by batch count.

  ## Parameters
    - `metrics` - Accumulated metrics map
    - `num_batches` - Number of batches accumulated

  ## Returns
    Averaged metrics map
  """
  @spec average_metrics(map(), pos_integer()) :: map()
  def average_metrics(metrics, num_batches) do
    Map.new(metrics, fn {k, v} -> {k, v / num_batches} end)
  end

  # Helper to convert targets to indices (handles one-hot and sparse)
  defp to_indices(targets) do
    if tuple_size(Nx.shape(targets)) > 1 do
      Nx.argmax(targets, axis: -1)
    else
      targets
    end
  end

  # ============================================================================
  # Action State Categories
  # ============================================================================

  @doc """
  Categorize a player's action state into high-level categories.

  Returns one of:
  - :grounded - Standing, walking, crouching, etc.
  - :aerial - Jumping, falling, aerials
  - :hitstun - In hitstun or tumble
  - :shielding - Shielding or shield-related states
  - :grabbing - Grabbing or being grabbed
  - :ledge - On ledge or ledge actions
  - :special - Special moves (B moves)
  - :other - Uncategorized states

  ## Parameters
    - `player` - Player struct with :action, :on_ground, :hitstun_frames_left fields

  ## Returns
    Action state category atom
  """
  @spec categorize_action_state(map()) :: atom()
  def categorize_action_state(player) when is_map(player) do
    # Use Map.get to handle both structs and maps
    action = Map.get(player, :action, 0)
    on_ground = Map.get(player, :on_ground, false)
    hitstun = Map.get(player, :hitstun_frames_left, 0)

    cond do
      # Hitstun takes priority
      hitstun > 0 -> :hitstun

      # Check action ID ranges (Melee action state IDs)
      action in 0x00..0x0A -> :dead_or_entry  # Dead, entry, rebirth
      action in 0x0E..0x18 -> :grounded       # Standing, walking
      action in 0x19..0x1C -> :grounded       # Turn, dash, run
      action in 0x1D..0x26 -> :grounded       # Crouch, slide
      action in 0x27..0x2C -> :aerial         # Jump squat, jump
      action in 0x2D..0x3A -> :aerial         # Aerial movement, fall
      action in 0x3B..0x45 -> :landing        # Landing
      action in 0x46..0x52 -> :attacking      # Ground attacks
      action in 0x53..0x5F -> :attacking      # More ground attacks
      action in 0x60..0x6F -> :aerial_attack  # Aerial attacks
      action in 0xB2..0xC3 -> :shielding      # Shield
      action in 0xC4..0xD3 -> :grabbing       # Grab, throw
      action in 0xD4..0xE3 -> :grabbed        # Being grabbed
      action in 0xFC..0xFF -> :ledge          # Ledge actions
      action in 0x112..0x140 -> :special      # B moves

      # Fallback to ground/air based on on_ground flag
      on_ground -> :grounded
      true -> :aerial
    end
  end

  @doc """
  Extract action state statistics from a list of frames.

  ## Parameters
    - `frames` - List of frame maps with :game_state containing player data

  ## Returns
    Map of category -> count
  """
  @spec action_state_distribution(list(map())) :: %{atom() => non_neg_integer()}
  def action_state_distribution(frames) do
    frames
    |> Enum.reduce(%{}, fn frame, acc ->
      # Handle both struct and map access patterns
      game_state = frame[:game_state] || frame.game_state
      players = if is_struct(game_state), do: game_state.players, else: game_state[:players]
      player = players[1] || players["1"]

      if player do
        category = categorize_action_state(player)
        Map.update(acc, category, 1, &(&1 + 1))
      else
        acc
      end
    end)
  end

  @doc """
  Get simplified action state category (fewer categories for clearer analysis).

  Maps detailed categories to broader groups:
  - :grounded (includes landing, attacking)
  - :aerial (includes aerial_attack)
  - :hitstun
  - :defensive (shielding, ledge)
  - :grab_related (grabbing, grabbed)
  - :other
  """
  @spec simplify_action_category(atom()) :: atom()
  def simplify_action_category(category) do
    case category do
      cat when cat in [:grounded, :landing, :attacking] -> :grounded
      cat when cat in [:aerial, :aerial_attack] -> :aerial
      :hitstun -> :hitstun
      cat when cat in [:shielding, :ledge] -> :defensive
      cat when cat in [:grabbing, :grabbed] -> :grab_related
      :special -> :special
      _ -> :other
    end
  end

  # =============================================================================
  # Action Transition Analysis
  # =============================================================================

  @doc """
  Analyze prediction transitions (how often predictions change frame-to-frame).

  Returns a map with transition statistics:
  - :transition_rate - Fraction of frames where prediction changed
  - :stable_streak_avg - Average length of stable prediction runs
  - :jitter_score - Measure of prediction instability (0=stable, 1=chaotic)

  ## Parameters
    - `predictions` - List of prediction values (bucket indices)

  ## Examples

      iex> Metrics.analyze_transitions([8, 8, 8, 7, 7, 8, 8])
      %{transition_rate: 0.33, stable_streak_avg: 2.33, jitter_score: 0.33}
  """
  @spec analyze_transitions([integer()]) :: %{
    transition_rate: float(),
    stable_streak_avg: float(),
    jitter_score: float()
  }
  def analyze_transitions(predictions) when length(predictions) < 2 do
    %{transition_rate: 0.0, stable_streak_avg: 0.0, jitter_score: 0.0}
  end

  def analyze_transitions(predictions) do
    pairs = Enum.zip(predictions, tl(predictions))
    transitions = Enum.count(pairs, fn {a, b} -> a != b end)
    total_pairs = length(pairs)

    transition_rate = if total_pairs > 0, do: transitions / total_pairs, else: 0.0

    # Calculate stable streaks
    streaks = calculate_streaks(predictions)
    stable_streak_avg = if length(streaks) > 0 do
      Enum.sum(streaks) / length(streaks)
    else
      0.0
    end

    # Jitter score: consider single-frame changes as jitter
    # A transition followed immediately by another transition is jittery
    jitter_count =
      predictions
      |> Enum.chunk_every(3, 1, :discard)
      |> Enum.count(fn [a, b, c] -> a == c and a != b end)  # A-B-A pattern

    jitter_score = if total_pairs > 1 do
      jitter_count / (total_pairs - 1)
    else
      0.0
    end

    %{
      transition_rate: Float.round(transition_rate, 4),
      stable_streak_avg: Float.round(stable_streak_avg, 2),
      jitter_score: Float.round(jitter_score, 4)
    }
  end

  @doc """
  Calculate lengths of consecutive runs of the same value.

  ## Examples

      iex> Metrics.calculate_streaks([1, 1, 1, 2, 2, 1])
      [3, 2, 1]
  """
  @spec calculate_streaks([any()]) :: [pos_integer()]
  def calculate_streaks([]), do: []
  def calculate_streaks([_single]), do: [1]
  def calculate_streaks(list) do
    list
    |> Enum.chunk_by(& &1)
    |> Enum.map(&length/1)
  end

  @doc """
  Track transition patterns between bucket values.

  Returns a map of `{from, to} => count` for common transitions.

  ## Parameters
    - `predictions` - List of prediction values
    - `opts` - Options:
      - `:min_count` - Minimum count to include (default: 1)

  ## Examples

      iex> Metrics.transition_patterns([8, 7, 8, 7, 8])
      %{{8, 7} => 2, {7, 8} => 2}
  """
  @spec transition_patterns([integer()], keyword()) :: %{{integer(), integer()} => pos_integer()}
  def transition_patterns(predictions, opts \\ []) do
    min_count = Keyword.get(opts, :min_count, 1)

    predictions
    |> Enum.zip(tl(predictions))
    |> Enum.filter(fn {a, b} -> a != b end)  # Only track actual transitions
    |> Enum.frequencies()
    |> Enum.filter(fn {_, count} -> count >= min_count end)
    |> Map.new()
  end

  @doc """
  Compute transition statistics for a batch of predictions.

  Takes a tensor of predictions (shape {batch}) and returns aggregate stats.

  ## Parameters
    - `predictions` - Nx tensor of predicted bucket indices, shape {batch}

  ## Returns
    Map with :transition_rate, :stable_streak_avg, :jitter_score
  """
  @spec batch_transition_stats(Nx.Tensor.t()) :: map()
  def batch_transition_stats(predictions) do
    pred_list = Nx.to_flat_list(predictions)
    analyze_transitions(pred_list)
  end
end
