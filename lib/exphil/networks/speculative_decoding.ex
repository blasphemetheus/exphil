defmodule ExPhil.Networks.SpeculativeDecoding do
  @moduledoc """
  Speculative Decoding for fast inference with quality verification.

  ## Overview

  Speculative decoding uses two models:
  1. **Draft model**: Small, fast model that proposes K candidate actions
  2. **Target model**: Large, accurate model that verifies candidates in parallel

  The key insight: verifying K actions in parallel with a large model is often
  faster than generating K actions sequentially with the same large model.

  ```
  Draft Model (MLP)          Target Model (Mamba)
       │                            │
       ▼                            │
  [a1, a2, a3, a4]                  │
       │                            │
       └──────────► Parallel ◄──────┘
                    Verify
                       │
                       ▼
              Accept [a1, a2] ✓
              Reject [a3, a4] ✗
                       │
                       ▼
              Return accepted + 1 new
  ```

  ## For Melee

  This is particularly powerful for real-time gameplay:
  - Draft model (MLP, ~9ms) proposes 4 actions quickly
  - Target model (Mamba, ~24ms) verifies all 4 in parallel
  - If 50% of speculated actions are accepted, we get ~2.5 actions in ~60ms

  ## Usage

      decoder = SpeculativeDecoding.create(
        draft_fn: &draft_model_predict/2,
        target_fn: &target_model_batch_predict/2,
        lookahead: 4
      )

      {actions, acceptance_count, decoder} =
        SpeculativeDecoding.generate(decoder, draft_params, target_params, state)
  """

  @default_lookahead 4
  @default_temperature 1.0

  @doc """
  Create a speculative decoder from draft and target models.

  ## Options

    - `:draft_fn` - Function `(params, state) -> logits` (required)
    - `:target_fn` - Function `(params, states) -> batch_logits` (required)
    - `:lookahead` - Number of tokens to speculate (default: 4)
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:acceptance_threshold` - Min probability ratio to accept (default: 0.8)

  ## Returns

    A map containing the decoder configuration and statistics.
  """
  @spec create(keyword()) :: map()
  def create(opts) do
    draft_fn = Keyword.fetch!(opts, :draft_fn)
    target_fn = Keyword.fetch!(opts, :target_fn)
    lookahead = Keyword.get(opts, :lookahead, @default_lookahead)
    temperature = Keyword.get(opts, :temperature, @default_temperature)
    acceptance_threshold = Keyword.get(opts, :acceptance_threshold, 0.8)

    %{
      draft_fn: draft_fn,
      target_fn: target_fn,
      lookahead: lookahead,
      temperature: temperature,
      acceptance_threshold: acceptance_threshold,
      stats: %{
        total_proposed: 0,
        total_accepted: 0,
        acceptance_rate: 0.0
      }
    }
  end

  @doc """
  Generate actions using speculative decoding.

  This is the main entry point for inference. It:
  1. Uses the draft model to generate K candidate actions
  2. Verifies candidates against the target model in parallel
  3. Accepts the longest prefix where target agrees with draft
  4. Returns accepted actions plus one target-sampled action

  ## Returns

    `{actions, acceptance_count, updated_decoder}`
  """
  @spec generate(map(), map(), map(), Nx.Tensor.t(), keyword()) ::
          {Nx.Tensor.t(), non_neg_integer(), map()}
  def generate(decoder, draft_params, target_params, state, opts \\ []) do
    %{
      draft_fn: draft_fn,
      target_fn: target_fn,
      lookahead: lookahead,
      temperature: temperature,
      acceptance_threshold: threshold
    } = decoder

    prng_key = Keyword.get(opts, :prng_key, Nx.Random.key(System.system_time()))

    # Step 1: Generate K candidate actions from draft model
    {candidates, draft_probs, prng_key} =
      generate_candidates_eager(draft_fn, draft_params, state, lookahead, temperature, prng_key)

    # Step 2: Build states for parallel target evaluation
    candidate_states = build_candidate_states(state, candidates, lookahead)

    # Step 3: Get target model probabilities for all candidates in parallel
    target_logits = target_fn.(target_params, candidate_states)
    target_probs = compute_probs_eager(target_logits, temperature)

    # Step 4: Determine acceptance via rejection sampling
    {accepted_actions, acceptance_count} =
      verify_candidates_eager(candidates, draft_probs, target_probs, threshold)

    # Step 5: Sample one new action from target
    {new_action, _prng_key} =
      if acceptance_count < lookahead do
        # Sample from target at rejection point
        rejected_probs = Nx.slice_along_axis(target_probs, acceptance_count, 1, axis: 0)
        sample_action_eager(Nx.squeeze(rejected_probs), prng_key)
      else
        # All accepted, sample continuation from target
        final_probs = Nx.slice_along_axis(target_probs, lookahead - 1, 1, axis: 0)
        sample_action_eager(Nx.squeeze(final_probs), prng_key)
      end

    # Combine accepted + new action
    final_actions =
      if acceptance_count > 0 do
        accepted = Nx.slice_along_axis(accepted_actions, 0, acceptance_count, axis: 0)
        Nx.concatenate([accepted, Nx.new_axis(new_action, 0)], axis: 0)
      else
        Nx.new_axis(new_action, 0)
      end

    # Update statistics
    updated_stats = update_stats(decoder.stats, lookahead, acceptance_count)
    updated_decoder = %{decoder | stats: updated_stats}

    {final_actions, acceptance_count + 1, updated_decoder}
  end

  @doc """
  Get current acceptance statistics.
  """
  @spec get_stats(map()) :: map()
  def get_stats(decoder) do
    decoder.stats
  end

  @doc """
  Reset statistics tracking.
  """
  @spec reset_stats(map()) :: map()
  def reset_stats(decoder) do
    %{decoder | stats: %{total_proposed: 0, total_accepted: 0, acceptance_rate: 0.0}}
  end

  @doc """
  Estimate speedup from speculative decoding.

  ## Arguments

    - `acceptance_rate` - Average acceptance rate (0.0 to 1.0)
    - `lookahead` - Number of speculated tokens
    - `draft_speed` - Draft model inference time (ms)
    - `target_speed` - Target model inference time (ms)

  ## Returns

    Estimated speedup factor (>1 means faster than pure target).

  ## Example

      # MLP draft (9ms), Mamba target (24ms), 50% acceptance, lookahead 4
      speedup = SpeculativeDecoding.estimate_speedup(0.5, 4, 9.0, 24.0)
      # Returns ~1.3x speedup
  """
  @spec estimate_speedup(float(), pos_integer(), number(), number()) :: float()
  def estimate_speedup(acceptance_rate, lookahead, draft_speed, target_speed) do
    # Without speculative: target_speed per token
    # With speculative:
    #   - Draft generates K tokens: K * draft_speed
    #   - Target verifies in parallel: 1 * target_speed
    #   - Expected tokens: K * acceptance_rate + 1

    expected_tokens = lookahead * acceptance_rate + 1
    spec_time = lookahead * draft_speed + target_speed
    naive_time = expected_tokens * target_speed

    naive_time / spec_time
  end

  @doc """
  Build a speculative decoder for Melee gameplay.

  Uses MLP as draft model and Mamba as target model by default.

  ## Options

    - `:draft_checkpoint` - Path to draft model checkpoint
    - `:target_checkpoint` - Path to target model checkpoint
    - `:lookahead` - Speculation depth (default: 4)
  """
  @spec build_melee_decoder(keyword()) :: map()
  def build_melee_decoder(opts) do
    draft_checkpoint = Keyword.fetch!(opts, :draft_checkpoint)
    target_checkpoint = Keyword.fetch!(opts, :target_checkpoint)
    lookahead = Keyword.get(opts, :lookahead, @default_lookahead)

    # Load models
    draft_params = load_checkpoint(draft_checkpoint)
    target_params = load_checkpoint(target_checkpoint)

    # Build prediction functions (simplified - real impl would build Axon models)
    draft_fn = fn _params, state ->
      # Placeholder - real implementation would use Policy.build
      Nx.broadcast(0.0, {Nx.axis_size(state, 0), 13})
    end

    target_fn = fn _params, states ->
      # Placeholder - real implementation would use Mamba.build
      Nx.broadcast(0.0, {Nx.axis_size(states, 0), 13})
    end

    decoder =
      create(
        draft_fn: draft_fn,
        target_fn: target_fn,
        lookahead: lookahead
      )

    %{
      decoder: decoder,
      draft_params: draft_params,
      target_params: target_params
    }
  end

  @doc """
  Recommended configuration for real-time Melee inference.
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      lookahead: 4,
      temperature: 1.0,
      acceptance_threshold: 0.8
    ]
  end

  # ============================================================================
  # Private Implementation (Eager Elixir functions)
  # ============================================================================

  defp generate_candidates_eager(draft_fn, params, state, lookahead, temperature, prng_key) do
    # Generate K candidates autoregressively from draft model
    {candidates, probs, final_key} =
      Enum.reduce(0..(lookahead - 1), {[], [], prng_key}, fn _i, {cands, probs_acc, key} ->
        # Get logits from draft model
        logits = draft_fn.(params, state)

        # Sample action
        {action, new_key} = sample_from_logits_eager(logits, temperature, key)

        # Get probability of sampled action
        action_prob = get_action_probability_eager(logits, action, temperature)

        {[action | cands], [action_prob | probs_acc], new_key}
      end)

    # Convert lists to tensors
    candidates_tensor =
      candidates
      |> Enum.reverse()
      |> Nx.stack()

    probs_tensor =
      probs
      |> Enum.reverse()
      |> Nx.stack()

    {candidates_tensor, probs_tensor, final_key}
  end

  defp sample_from_logits_eager(logits, temperature, key) do
    # Apply temperature and softmax
    scaled_logits = Nx.divide(logits, temperature)
    max_logit = Nx.reduce_max(scaled_logits, axes: [-1], keep_axes: true)
    exp_logits = Nx.exp(Nx.subtract(scaled_logits, max_logit))
    probs = Nx.divide(exp_logits, Nx.sum(exp_logits, axes: [-1], keep_axes: true))

    # Flatten for sampling
    flat_probs = Nx.flatten(probs)
    num_actions = Nx.size(flat_probs)

    # Sample using cumulative distribution
    {uniform, new_key} = Nx.Random.uniform(key, shape: {})
    cumsum = Nx.cumulative_sum(flat_probs)
    sampled_idx = Nx.sum(Nx.less(cumsum, uniform)) |> Nx.to_number() |> min(num_actions - 1)

    action = Nx.tensor(sampled_idx)
    {action, new_key}
  end

  defp get_action_probability_eager(logits, action, temperature) do
    scaled_logits = Nx.divide(logits, temperature)
    max_logit = Nx.reduce_max(scaled_logits, axes: [-1], keep_axes: true)
    exp_logits = Nx.exp(Nx.subtract(scaled_logits, max_logit))
    probs = Nx.divide(exp_logits, Nx.sum(exp_logits, axes: [-1], keep_axes: true))

    # Get probability at action index
    action_idx = Nx.to_number(action)
    flat_probs = Nx.flatten(probs)
    Nx.slice_along_axis(flat_probs, action_idx, 1) |> Nx.squeeze()
  end

  defp compute_probs_eager(logits, temperature) do
    scaled_logits = Nx.divide(logits, temperature)
    max_logit = Nx.reduce_max(scaled_logits, axes: [-1], keep_axes: true)
    exp_logits = Nx.exp(Nx.subtract(scaled_logits, max_logit))
    Nx.divide(exp_logits, Nx.sum(exp_logits, axes: [-1], keep_axes: true))
  end

  defp verify_candidates_eager(candidates, draft_probs, target_probs, threshold) do
    # Accept candidates while target_prob / draft_prob >= threshold
    lookahead = Nx.axis_size(candidates, 0)

    # Check each position
    acceptance_count =
      Enum.reduce_while(0..(lookahead - 1), 0, fn idx, count ->
        draft_p = Nx.slice_along_axis(draft_probs, idx, 1) |> Nx.squeeze() |> Nx.to_number()
        target_p = Nx.slice_along_axis(target_probs, idx, 1) |> Nx.squeeze() |> Nx.to_number()

        ratio = target_p / max(draft_p, 1.0e-8)

        if ratio >= threshold do
          {:cont, count + 1}
        else
          {:halt, count}
        end
      end)

    {candidates, acceptance_count}
  end

  defp sample_action_eager(probs, key) do
    flat_probs = Nx.flatten(probs)
    num_actions = Nx.size(flat_probs)

    # Sample using cumulative distribution
    {uniform, new_key} = Nx.Random.uniform(key, shape: {})
    cumsum = Nx.cumulative_sum(flat_probs)
    sampled_idx = Nx.sum(Nx.less(cumsum, uniform)) |> Nx.to_number() |> min(num_actions - 1)

    action = Nx.tensor(sampled_idx)
    {action, new_key}
  end

  defp build_candidate_states(state, _candidates, lookahead) do
    # Build sequence of states for parallel target evaluation
    # For simplicity, replicate the state K times
    # Real implementation would integrate candidate actions into each state
    state_shape = Nx.shape(state)

    # Replicate state for each candidate position
    case tuple_size(state_shape) do
      2 ->
        # [batch, embed_size] -> [lookahead * batch, embed_size]
        Nx.tile(state, [lookahead, 1])

      3 ->
        # [batch, seq_len, embed_size] -> [lookahead * batch, seq_len, embed_size]
        Nx.tile(state, [lookahead, 1, 1])

      _ ->
        # Fallback
        state
    end
  end

  defp update_stats(stats, proposed, accepted) do
    total_proposed = stats.total_proposed + proposed
    total_accepted = stats.total_accepted + accepted

    acceptance_rate =
      if total_proposed > 0 do
        total_accepted / total_proposed
      else
        0.0
      end

    %{
      total_proposed: total_proposed,
      total_accepted: total_accepted,
      acceptance_rate: acceptance_rate
    }
  end

  defp load_checkpoint(path) do
    case File.read(path) do
      {:ok, binary} ->
        :erlang.binary_to_term(binary)

      {:error, _reason} ->
        # Return empty params for testing
        %{}
    end
  end
end
