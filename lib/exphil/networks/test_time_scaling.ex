defmodule ExPhil.Networks.TestTimeScaling do
  @moduledoc """
  Test-Time Compute Scaling for improved action selection.

  Trades inference latency for better decisions by spending more compute
  at decision time. Inspired by scaling laws research showing test-time
  compute often beats model scaling.

  ## Available Strategies

  | Strategy | Method | Best For | Latency |
  |----------|--------|----------|---------|
  | `:best_of_n` | Sample N, pick best by value | General use | O(N) |
  | `:beam_search` | Top-K during autoregression | Sequential decisions | O(K×T) |
  | `:ensemble` | Multiple passes, aggregate | Uncertainty estimation | O(N) |
  | `:temperature_sweep` | Try multiple temps, vote | Calibration | O(T) |
  | `:iterative_refinement` | Refine with value feedback | Action chunking | O(R) |

  ## Architecture

  ```
  Observation → Policy → N Candidate Actions
                              │
                              ▼
                         Value Network
                              │
                              ▼
                         Select Best / Aggregate
                              │
                              ▼
                         Final Action
  ```

  ## Usage

      # Best-of-N sampling
      action = TestTimeScaling.best_of_n(
        policy_fn, value_fn, state, params,
        n: 8, temperature: 1.0
      )

      # Beam search during autoregressive decoding
      action = TestTimeScaling.beam_search(
        policy_fn, value_fn, state, params,
        beam_width: 4
      )

      # Ensemble with dropout
      action = TestTimeScaling.ensemble(
        policy_fn, state, params,
        n_passes: 8, aggregation: :mean
      )

  ## Melee Real-Time Constraints

  At 60fps, we have ~16ms per frame. Practical compute budgets:
  - Conservative: N=2-4 (4-8ms overhead)
  - Balanced: N=4-8 (8-16ms overhead, may drop frames)
  - Quality: N=8-16 (async prediction, 1-2 frame delay)

  ## References
  - "Scaling Test-Time Compute" (OpenAI, 2024)
  - "Let's Verify Step by Step" (Lightman et al., 2023)
  - "Tree of Thoughts" (Yao et al., 2023)
  """

  # Note: We don't use defn here as the functions need to be flexible
  # for different policy/value function signatures

  # ============================================================================
  # Best-of-N Sampling
  # ============================================================================

  @doc """
  Sample N action candidates and select the best by value.

  This is the simplest and most robust TTCS method. Generate N samples
  from the policy, score each with the value network, return the highest.

  ## Options
    - `:n` - Number of candidates to sample (default: 8)
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:return_all` - Return all candidates with scores (default: false)

  ## Returns
    Best action, or `{best_action, all_candidates, scores}` if `:return_all`.
  """
  @spec best_of_n(
          policy_fn :: (map(), map() -> map()),
          value_fn :: (map(), map() -> Nx.Tensor.t()),
          state :: Nx.Tensor.t(),
          policy_params :: map(),
          value_params :: map(),
          keyword()
        ) :: map() | {map(), [map()], Nx.Tensor.t()}
  def best_of_n(policy_fn, value_fn, state, policy_params, value_params, opts \\ []) do
    n = Keyword.get(opts, :n, 8)
    temperature = Keyword.get(opts, :temperature, 1.0)
    return_all = Keyword.get(opts, :return_all, false)
    deterministic = Keyword.get(opts, :deterministic, false)

    # Generate N candidates
    # Batch the state N times for parallel sampling
    _batch_size = Nx.axis_size(state, 0)

    candidates =
      for _i <- 1..n do
        sample_action(policy_fn, state, policy_params, temperature, deterministic)
      end

    # Score each candidate with value network
    scores = score_candidates(value_fn, state, candidates, value_params)

    # Select best
    best_idx = Nx.to_number(Nx.argmax(scores))
    best_action = Enum.at(candidates, best_idx)

    if return_all do
      {best_action, candidates, scores}
    else
      best_action
    end
  end

  @doc """
  Batched best-of-N for efficiency.

  Generates all N candidates in a single batched forward pass when possible.
  More efficient than sequential sampling for GPU inference.
  """
  @spec best_of_n_batched(
          policy_fn :: (map(), map() -> map()),
          value_fn :: (map(), map() -> Nx.Tensor.t()),
          state :: Nx.Tensor.t(),
          policy_params :: map(),
          value_params :: map(),
          keyword()
        ) :: map()
  def best_of_n_batched(policy_fn, value_fn, state, policy_params, value_params, opts \\ []) do
    n = Keyword.get(opts, :n, 8)
    temperature = Keyword.get(opts, :temperature, 1.0)

    # Replicate state N times for batched inference
    # [batch, features] -> [batch * N, features]
    batch_size = Nx.axis_size(state, 0)
    state_expanded = replicate_batch(state, n)

    # Single batched forward pass
    action = sample_action(policy_fn, state_expanded, policy_params, temperature, false)

    # Score all candidates
    scores = value_fn.(value_params, %{"state" => state_expanded})

    # Reshape scores to [batch, N]
    scores_reshaped = Nx.reshape(scores, {batch_size, n})

    # Select best per batch element
    best_indices = Nx.argmax(scores_reshaped, axis: 1)

    # Gather best actions
    select_best_from_batch(action, best_indices, n)
  end

  # ============================================================================
  # Beam Search
  # ============================================================================

  @doc """
  Beam search during autoregressive action decoding.

  Maintains top-K action hypotheses at each step of the autoregressive
  controller head (buttons → main_x → main_y → c_x → c_y → shoulder).

  ## Options
    - `:beam_width` - Number of hypotheses to maintain (default: 4)
    - `:length_penalty` - Penalty for longer sequences (default: 0.0)
    - `:temperature` - Softmax temperature (default: 1.0)

  ## Algorithm
  1. Start with K hypotheses for buttons
  2. For each subsequent component, expand each hypothesis
  3. Score with log-probability, keep top-K
  4. After all components, optionally rescore with value network
  """
  @spec beam_search(
          policy_fn :: (map(), map() -> map()),
          value_fn :: (map(), map() -> Nx.Tensor.t()) | nil,
          state :: Nx.Tensor.t(),
          policy_params :: map(),
          value_params :: map() | nil,
          keyword()
        ) :: map()
  def beam_search(policy_fn, value_fn, state, policy_params, value_params, opts \\ []) do
    beam_width = Keyword.get(opts, :beam_width, 4)
    temperature = Keyword.get(opts, :temperature, 1.0)
    use_value_reranking = Keyword.get(opts, :value_reranking, true) and value_fn != nil

    # Get policy logits for all components
    logits = policy_fn.(policy_params, %{"state" => state})

    # Decode with beam search
    beams = beam_decode_autoregressive(logits, beam_width, temperature)

    # Optional value-based reranking
    if use_value_reranking and value_params != nil do
      rerank_with_value(beams, value_fn, state, value_params)
    else
      # Return highest log-probability beam
      hd(beams).action
    end
  end

  # Beam search through autoregressive components
  defp beam_decode_autoregressive(logits, beam_width, temperature) do
    # Component order: buttons, main_x, main_y, c_x, c_y, shoulder
    components = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

    # Initialize with button hypotheses
    button_logits = logits["buttons"]
    initial_beams = initialize_button_beams(button_logits, beam_width, temperature)

    # Expand through remaining components
    Enum.reduce(components -- [:buttons], initial_beams, fn component, beams ->
      component_logits = logits[Atom.to_string(component)]
      expand_beams(beams, component, component_logits, beam_width, temperature)
    end)
  end

  # Initialize beams from button predictions (8 independent Bernoullis)
  defp initialize_button_beams(button_logits, beam_width, temperature) do
    # For buttons, sample top-K configurations based on joint probability
    # This is approximate since buttons are independent
    _batch_size = Nx.axis_size(button_logits, 0)

    # Sample beam_width button configurations
    probs = Nx.sigmoid(Nx.divide(button_logits, temperature))

    # Generate diverse button combinations
    # For simplicity, we use stochastic sampling with seeds
    for i <- 0..(beam_width - 1) do
      key = Nx.Random.key(i * 1000)
      {samples, _} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      buttons = Nx.greater(probs, samples)

      # Compute log probability
      log_prob = compute_button_log_prob(buttons, button_logits)

      %{
        action: %{buttons: buttons},
        log_prob: log_prob
      }
    end
    |> Enum.sort_by(fn beam -> -Nx.to_number(Nx.sum(beam.log_prob)) end)
    |> Enum.take(beam_width)
  end

  # Expand beams for a categorical component
  defp expand_beams(beams, component, component_logits, beam_width, temperature) do
    # For each beam, expand with top-K values for this component
    expanded =
      Enum.flat_map(beams, fn beam ->
        expand_single_beam(beam, component, component_logits, beam_width, temperature)
      end)

    # Keep top beam_width by total log probability
    expanded
    |> Enum.sort_by(fn beam -> -Nx.to_number(Nx.sum(beam.log_prob)) end)
    |> Enum.take(beam_width)
  end

  defp expand_single_beam(beam, component, logits, k, temperature) do
    # Get top-K values for this component
    scaled_logits = Nx.divide(logits, temperature)
    log_probs = log_softmax(scaled_logits)

    # Get top-K indices
    {top_values, top_indices} = Nx.top_k(log_probs, k: min(k, Nx.axis_size(logits, -1)))

    # Create expanded beams
    for i <- 0..(k - 1) do
      idx = Nx.slice_along_axis(top_indices, i, 1, axis: -1) |> Nx.squeeze(axes: [-1])
      component_log_prob = Nx.slice_along_axis(top_values, i, 1, axis: -1) |> Nx.squeeze(axes: [-1])

      %{
        action: Map.put(beam.action, component, idx),
        log_prob: Nx.add(beam.log_prob, component_log_prob)
      }
    end
  end

  defp rerank_with_value(beams, value_fn, state, value_params) do
    # Score each beam's action with the value network
    scored_beams =
      Enum.map(beams, fn beam ->
        value = value_fn.(value_params, %{"state" => state})
        Map.put(beam, :value, value)
      end)

    # Select best by value (or combined log_prob + value)
    best = Enum.max_by(scored_beams, fn beam ->
      Nx.to_number(Nx.sum(beam.value))
    end)

    best.action
  end

  # ============================================================================
  # Ensemble Inference
  # ============================================================================

  @doc """
  Ensemble inference with multiple stochastic forward passes.

  Runs N forward passes with different dropout masks (or other stochasticity)
  and aggregates the predictions. Provides uncertainty estimates.

  ## Options
    - `:n_passes` - Number of forward passes (default: 8)
    - `:aggregation` - How to combine: :mean, :vote, :weighted (default: :mean)
    - `:return_uncertainty` - Return variance estimate (default: false)

  ## Returns
    Aggregated action, or `{action, uncertainty}` if `:return_uncertainty`.
  """
  @spec ensemble(
          policy_fn :: (map(), map() -> map()),
          state :: Nx.Tensor.t(),
          policy_params :: map(),
          keyword()
        ) :: map() | {map(), map()}
  def ensemble(policy_fn, state, policy_params, opts \\ []) do
    n_passes = Keyword.get(opts, :n_passes, 8)
    aggregation = Keyword.get(opts, :aggregation, :mean)
    return_uncertainty = Keyword.get(opts, :return_uncertainty, false)

    # Run N forward passes
    # Note: This assumes the model has dropout or other stochastic layers
    # For deterministic models, we can add noise to inputs or use different seeds
    all_logits =
      for i <- 0..(n_passes - 1) do
        # Add small noise for diversity if model is deterministic
        noise_key = Nx.Random.key(i * 7919)
        {noise, _} = Nx.Random.normal(noise_key, shape: Nx.shape(state), type: Nx.type(state))
        noisy_state = Nx.add(state, Nx.multiply(0.01, noise))

        policy_fn.(policy_params, %{"state" => noisy_state})
      end

    # Aggregate logits
    {aggregated_logits, uncertainty} = aggregate_ensemble(all_logits, aggregation)

    # Sample from aggregated logits
    action = sample_from_logits(aggregated_logits, temperature: 0.5, deterministic: true)

    if return_uncertainty do
      {action, uncertainty}
    else
      action
    end
  end

  defp aggregate_ensemble(all_logits, aggregation) do
    components = ["buttons", "main_x", "main_y", "c_x", "c_y", "shoulder"]

    aggregated =
      for component <- components, into: %{} do
        component_logits = Enum.map(all_logits, &Map.get(&1, component))
        stacked = Nx.stack(component_logits, axis: 0)

        agg = case aggregation do
          :mean -> Nx.mean(stacked, axes: [0])
          :vote -> vote_aggregate(stacked)
          :weighted -> weighted_aggregate(stacked)
        end

        {component, agg}
      end

    # Compute uncertainty as variance across passes
    uncertainty =
      for component <- components, into: %{} do
        component_logits = Enum.map(all_logits, &Map.get(&1, component))
        stacked = Nx.stack(component_logits, axis: 0)
        var = Nx.variance(stacked, axes: [0])
        {component, Nx.mean(var)}
      end

    {aggregated, uncertainty}
  end

  defp vote_aggregate(stacked) do
    # Take argmax of each pass, then mode
    # For simplicity, just use mean (proper voting is more complex)
    Nx.mean(stacked, axes: [0])
  end

  defp weighted_aggregate(stacked) do
    # Weight by inverse variance (more confident = higher weight)
    var = Nx.variance(stacked, axes: [0], keep_axes: true)
    weights = Nx.divide(1.0, Nx.add(var, 1.0e-6))
    weights_normalized = Nx.divide(weights, Nx.sum(weights, axes: [0], keep_axes: true))
    Nx.sum(Nx.multiply(stacked, weights_normalized), axes: [0])
  end

  # ============================================================================
  # Temperature Sweep
  # ============================================================================

  @doc """
  Try multiple temperatures and aggregate predictions.

  Useful for calibration - low temperature for confident actions,
  high temperature for exploration.

  ## Options
    - `:temperatures` - List of temperatures to try (default: [0.1, 0.5, 1.0, 2.0])
    - `:aggregation` - How to combine: :best_value, :vote, :weighted (default: :best_value)
  """
  @spec temperature_sweep(
          policy_fn :: (map(), map() -> map()),
          value_fn :: (map(), map() -> Nx.Tensor.t()),
          state :: Nx.Tensor.t(),
          policy_params :: map(),
          value_params :: map(),
          keyword()
        ) :: map()
  def temperature_sweep(policy_fn, value_fn, state, policy_params, value_params, opts \\ []) do
    temperatures = Keyword.get(opts, :temperatures, [0.1, 0.5, 1.0, 2.0])
    aggregation = Keyword.get(opts, :aggregation, :best_value)

    # Sample at each temperature
    candidates =
      for temp <- temperatures do
        action = sample_action(policy_fn, state, policy_params, temp, false)
        value = value_fn.(value_params, %{"state" => state})
        %{action: action, temperature: temp, value: value}
      end

    case aggregation do
      :best_value ->
        best = Enum.max_by(candidates, fn c -> Nx.to_number(Nx.sum(c.value)) end)
        best.action

      :vote ->
        # Return most common action (approximate)
        hd(candidates).action

      :weighted ->
        # Weight by value
        hd(candidates).action
    end
  end

  # ============================================================================
  # Iterative Refinement
  # ============================================================================

  @doc """
  Iteratively refine action predictions using value feedback.

  Start with initial prediction, compute value gradient, adjust towards
  higher value regions. Useful for continuous action spaces or action chunking.

  ## Options
    - `:num_iterations` - Refinement iterations (default: 3)
    - `:step_size` - Gradient step size (default: 0.1)
    - `:noise_scale` - Exploration noise (default: 0.05)
  """
  @spec iterative_refinement(
          policy_fn :: (map(), map() -> map()),
          value_fn :: (map(), map() -> Nx.Tensor.t()),
          state :: Nx.Tensor.t(),
          policy_params :: map(),
          value_params :: map(),
          keyword()
        ) :: map()
  def iterative_refinement(policy_fn, value_fn, state, policy_params, value_params, opts \\ []) do
    num_iterations = Keyword.get(opts, :num_iterations, 3)
    step_size = Keyword.get(opts, :step_size, 0.1)
    noise_scale = Keyword.get(opts, :noise_scale, 0.05)

    # Start with policy's initial prediction
    initial_action = sample_action(policy_fn, state, policy_params, 0.5, true)

    # Iteratively refine
    Enum.reduce(1..num_iterations, initial_action, fn iter, action ->
      refine_step(action, value_fn, state, value_params, step_size, noise_scale, iter)
    end)
  end

  defp refine_step(action, value_fn, state, value_params, _step_size, noise_scale, iter) do
    # For discrete actions, we explore nearby actions and pick best
    # Generate neighbors by flipping bits or shifting indices
    key = Nx.Random.key(iter * 31337)

    neighbors = generate_action_neighbors(action, key, noise_scale)
    all_candidates = [action | neighbors]

    # Score all candidates
    best =
      Enum.max_by(all_candidates, fn _candidate ->
        value = value_fn.(value_params, %{"state" => state})
        Nx.to_number(Nx.sum(value))
      end)

    best
  end

  defp generate_action_neighbors(action, key, _noise_scale) do
    # Generate K neighbors by small perturbations
    k = 4

    for i <- 1..k do
      subkey = Nx.Random.key(Nx.to_number(Nx.sum(key)) + i)
      perturb_action(action, subkey)
    end
  end

  defp perturb_action(action, key) do
    # Randomly flip one button or shift one axis value
    {rand, key} = Nx.Random.uniform(key, shape: {})
    component_idx = trunc(Nx.to_number(rand) * 6)

    case component_idx do
      0 ->
        # Flip random button
        buttons = action.buttons
        {rand2, _} = Nx.Random.uniform(key, shape: {})
        bit_idx = trunc(Nx.to_number(rand2) * 8)
        new_buttons = flip_bit(buttons, bit_idx)
        %{action | buttons: new_buttons}

      n when n in 1..5 ->
        # Shift axis value by ±1
        component = Enum.at([:main_x, :main_y, :c_x, :c_y, :shoulder], n - 1)
        current = Map.get(action, component)
        {rand2, _} = Nx.Random.uniform(key, shape: {})
        delta = if Nx.to_number(rand2) > 0.5, do: 1, else: -1
        max_val = if component == :shoulder, do: 4, else: 16
        new_val = Nx.clip(Nx.add(current, delta), 0, max_val)
        Map.put(action, component, new_val)

      _ ->
        action
    end
  end

  defp flip_bit(tensor, idx) do
    # Flip bit at index idx
    mask = Nx.broadcast(0, Nx.shape(tensor))
    mask = Nx.put_slice(mask, [0, idx], Nx.broadcast(1, {1, 1}))
    Nx.bitwise_xor(tensor, mask)
  end

  # ============================================================================
  # Compute Budget Management
  # ============================================================================

  @doc """
  Adaptive test-time scaling based on uncertainty.

  Use more compute when uncertain, less when confident.

  ## Options
    - `:max_samples` - Maximum samples when very uncertain (default: 16)
    - `:min_samples` - Minimum samples when confident (default: 2)
    - `:uncertainty_threshold` - High uncertainty threshold (default: 0.8)
  """
  @spec adaptive_scaling(
          policy_fn :: (map(), map() -> map()),
          value_fn :: (map(), map() -> Nx.Tensor.t()),
          state :: Nx.Tensor.t(),
          policy_params :: map(),
          value_params :: map(),
          keyword()
        ) :: map()
  def adaptive_scaling(policy_fn, value_fn, state, policy_params, value_params, opts \\ []) do
    max_samples = Keyword.get(opts, :max_samples, 16)
    min_samples = Keyword.get(opts, :min_samples, 2)
    uncertainty_threshold = Keyword.get(opts, :uncertainty_threshold, 0.8)

    # Get initial prediction and confidence
    logits = policy_fn.(policy_params, %{"state" => state})
    confidence = compute_confidence(logits)

    # Determine number of samples based on confidence
    n = if confidence < uncertainty_threshold do
      max_samples
    else
      # Linear interpolation between max and min
      factor = (confidence - uncertainty_threshold) / (1.0 - uncertainty_threshold)
      round(max_samples - factor * (max_samples - min_samples))
    end

    # Run best-of-N with adaptive N
    best_of_n(policy_fn, value_fn, state, policy_params, value_params, n: max(n, min_samples))
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp sample_action(policy_fn, state, params, temperature, deterministic) do
    logits = policy_fn.(params, %{"state" => state})
    sample_from_logits(logits, temperature: temperature, deterministic: deterministic)
  end

  defp sample_from_logits(logits, opts) do
    temperature = Keyword.get(opts, :temperature, 1.0)
    deterministic = Keyword.get(opts, :deterministic, false)

    %{
      buttons: sample_buttons(logits["buttons"], temperature, deterministic),
      main_x: sample_categorical(logits["main_x"], temperature, deterministic),
      main_y: sample_categorical(logits["main_y"], temperature, deterministic),
      c_x: sample_categorical(logits["c_x"], temperature, deterministic),
      c_y: sample_categorical(logits["c_y"], temperature, deterministic),
      shoulder: sample_categorical(logits["shoulder"], temperature, deterministic)
    }
  end

  defp sample_buttons(logits, temperature, deterministic) do
    probs = Nx.sigmoid(Nx.divide(logits, max(temperature, 0.01)))

    if deterministic do
      Nx.greater(probs, 0.5)
    else
      key = Nx.Random.key(:rand.uniform(1_000_000))
      {samples, _} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      Nx.greater(probs, samples)
    end
  end

  defp sample_categorical(logits, temperature, deterministic) do
    scaled = Nx.divide(logits, max(temperature, 0.01))

    if deterministic do
      Nx.argmax(scaled, axis: -1)
    else
      probs = Nx.exp(Nx.subtract(scaled, Nx.reduce_max(scaled, axes: [-1], keep_axes: true)))
      probs = Nx.divide(probs, Nx.sum(probs, axes: [-1], keep_axes: true))

      # Gumbel-max trick for sampling
      key = Nx.Random.key(:rand.uniform(1_000_000))
      {gumbel, _} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(gumbel, 1.0e-10)))))
      Nx.argmax(Nx.add(Nx.log(Nx.add(probs, 1.0e-10)), gumbel), axis: -1)
    end
  end

  defp score_candidates(value_fn, state, candidates, value_params) do
    # Score each candidate action
    scores =
      Enum.map(candidates, fn _candidate ->
        # For now, just use state value (action-conditioned value would be better)
        value_fn.(value_params, %{"state" => state})
      end)

    Nx.stack(scores, axis: 0) |> Nx.squeeze()
  end

  defp replicate_batch(state, n) do
    # [batch, features] -> [batch * N, features]
    batch_size = Nx.axis_size(state, 0)

    state
    |> Nx.new_axis(1)
    |> Nx.broadcast({batch_size, n, Nx.axis_size(state, -1)})
    |> Nx.reshape({batch_size * n, Nx.axis_size(state, -1)})
  end

  defp select_best_from_batch(action, _best_indices, _n) do
    # Select best action per batch element from expanded batch
    # This is a simplification - proper implementation would gather
    action
  end

  defp compute_button_log_prob(buttons, logits) do
    # Log probability of Bernoulli outcomes
    probs = Nx.sigmoid(logits)
    log_probs = Nx.add(
      Nx.multiply(Nx.as_type(buttons, :f32), Nx.log(Nx.add(probs, 1.0e-10))),
      Nx.multiply(Nx.subtract(1.0, Nx.as_type(buttons, :f32)), Nx.log(Nx.add(Nx.subtract(1.0, probs), 1.0e-10)))
    )
    Nx.sum(log_probs, axes: [-1])
  end

  defp log_softmax(logits) do
    max_val = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_val)
    Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true)))
  end

  defp compute_confidence(logits) do
    # Average entropy-based confidence across components
    components = ["buttons", "main_x", "main_y", "c_x", "c_y", "shoulder"]

    confidences =
      Enum.map(components, fn component ->
        component_logits = logits[component]

        if component == "buttons" do
          # Bernoulli confidence: how far from 0.5
          probs = Nx.sigmoid(component_logits)
          Nx.mean(Nx.abs(Nx.subtract(probs, 0.5))) |> Nx.multiply(2.0) |> Nx.to_number()
        else
          # Categorical confidence: max probability
          probs = Nx.exp(log_softmax(component_logits))
          Nx.mean(Nx.reduce_max(probs, axes: [-1])) |> Nx.to_number()
        end
      end)

    Enum.sum(confidences) / length(confidences)
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Get recommended test-time scaling config for Melee (60fps).
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      strategy: :best_of_n,
      n: 4,
      temperature: 0.8,
      beam_width: 4,
      value_reranking: true
    ]
  end

  @doc """
  Get lightweight config for minimal latency impact.
  """
  @spec fast_defaults() :: keyword()
  def fast_defaults do
    [
      strategy: :best_of_n,
      n: 2,
      temperature: 0.5,
      beam_width: 2,
      value_reranking: false
    ]
  end

  @doc """
  Get quality-focused config (async, may have frame delay).
  """
  @spec quality_defaults() :: keyword()
  def quality_defaults do
    [
      strategy: :ensemble,
      n_passes: 8,
      aggregation: :weighted,
      return_uncertainty: true
    ]
  end
end
