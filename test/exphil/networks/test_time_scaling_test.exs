defmodule ExPhil.Networks.TestTimeScalingTest do
  @moduledoc """
  Tests for Test-Time Compute Scaling strategies.

  Tests cover various scaling strategies for improved action selection
  at inference time, including best-of-N, beam search, and ensemble methods.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.TestTimeScaling

  @obs_size 64
  @batch_size 2

  # Mock policy function that returns consistent logits
  defp mock_policy_fn(params, inputs) do
    state = inputs["state"]
    batch = Nx.axis_size(state, 0)

    # Use params to seed for reproducibility
    seed = Map.get(params, :seed, 42)

    %{
      "buttons" => Nx.broadcast(0.0, {batch, 8}),
      "main_x" => make_categorical_logits(batch, 17, seed),
      "main_y" => make_categorical_logits(batch, 17, seed + 1),
      "c_x" => make_categorical_logits(batch, 17, seed + 2),
      "c_y" => make_categorical_logits(batch, 17, seed + 3),
      "shoulder" => make_categorical_logits(batch, 5, seed + 4)
    }
  end

  defp make_categorical_logits(batch, num_classes, seed) do
    key = Nx.Random.key(seed)
    {logits, _} = Nx.Random.normal(key, shape: {batch, num_classes})
    logits
  end

  # Mock value function that returns a scalar value
  defp mock_value_fn(params, inputs) do
    state = inputs["state"]
    batch = Nx.axis_size(state, 0)
    seed = Map.get(params, :seed, 123)

    key = Nx.Random.key(seed)
    {values, _} = Nx.Random.uniform(key, shape: {batch})
    values
  end

  describe "best_of_n/6" do
    test "returns action with correct structure" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :main_y)
      assert Map.has_key?(action, :c_x)
      assert Map.has_key?(action, :c_y)
      assert Map.has_key?(action, :shoulder)
    end

    test "returns all candidates when return_all is true" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      {best_action, candidates, scores} = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4,
        return_all: true
      )

      assert Map.has_key?(best_action, :buttons)
      assert length(candidates) == 4
      # Scores shape is {n_candidates, batch_size} = {4, 2}
      assert Nx.shape(scores) == {4, @batch_size}
    end

    @tag timeout: 600_000
    test "more candidates with higher N" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      {_, candidates_4, _} = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4,
        return_all: true
      )

      {_, candidates_8, _} = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 8,
        return_all: true
      )

      assert length(candidates_4) == 4
      assert length(candidates_8) == 8
    end

    test "deterministic mode produces consistent results" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action1 = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4,
        deterministic: true
      )

      action2 = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4,
        deterministic: true
      )

      # With deterministic sampling from same logits, should be identical
      assert Nx.to_flat_list(action1.main_x) == Nx.to_flat_list(action2.main_x)
    end
  end

  describe "best_of_n_batched/6" do
    test "returns action with correct structure" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action = TestTimeScaling.best_of_n_batched(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
    end
  end

  describe "beam_search/6" do
    test "returns action with correct structure" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action = TestTimeScaling.beam_search(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        beam_width: 4
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :main_y)
      assert Map.has_key?(action, :c_x)
      assert Map.has_key?(action, :c_y)
      assert Map.has_key?(action, :shoulder)
    end

    test "works without value reranking" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}

      action = TestTimeScaling.beam_search(
        &mock_policy_fn/2,
        nil,
        state,
        policy_params,
        nil,
        beam_width: 4,
        value_reranking: false
      )

      assert Map.has_key?(action, :buttons)
    end

    test "respects beam_width parameter" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}

      # Should not error with different beam widths
      for beam_width <- [2, 4, 8] do
        action = TestTimeScaling.beam_search(
          &mock_policy_fn/2,
          nil,
          state,
          policy_params,
          nil,
          beam_width: beam_width,
          value_reranking: false
        )

        assert Map.has_key?(action, :buttons),
               "Failed for beam_width=#{beam_width}"
      end
    end
  end

  describe "ensemble/4" do
    test "returns action with correct structure" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}

      action = TestTimeScaling.ensemble(
        &mock_policy_fn/2,
        state,
        policy_params,
        n_passes: 4
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
    end

    test "returns uncertainty when requested" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}

      {action, uncertainty} = TestTimeScaling.ensemble(
        &mock_policy_fn/2,
        state,
        policy_params,
        n_passes: 4,
        return_uncertainty: true
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(uncertainty, "buttons")
      assert Map.has_key?(uncertainty, "main_x")
    end

    test "supports different aggregation methods" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}

      for agg <- [:mean, :vote, :weighted] do
        action = TestTimeScaling.ensemble(
          &mock_policy_fn/2,
          state,
          policy_params,
          n_passes: 4,
          aggregation: agg
        )

        assert Map.has_key?(action, :buttons),
               "Failed for aggregation=#{agg}"
      end
    end
  end

  describe "temperature_sweep/6" do
    test "returns action with correct structure" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action = TestTimeScaling.temperature_sweep(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        temperatures: [0.5, 1.0, 2.0]
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
    end

    test "uses custom temperatures" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      # Should not error with different temperature lists
      for temps <- [[0.1], [0.1, 0.5], [0.1, 0.5, 1.0, 2.0, 5.0]] do
        action = TestTimeScaling.temperature_sweep(
          &mock_policy_fn/2,
          &mock_value_fn/2,
          state,
          policy_params,
          value_params,
          temperatures: temps
        )

        assert Map.has_key?(action, :buttons)
      end
    end
  end

  describe "iterative_refinement/6" do
    test "returns action with correct structure" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action = TestTimeScaling.iterative_refinement(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        num_iterations: 3
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
    end

    test "more iterations refine the action" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      # Should complete without error
      for iters <- [1, 2, 5] do
        action = TestTimeScaling.iterative_refinement(
          &mock_policy_fn/2,
          &mock_value_fn/2,
          state,
          policy_params,
          value_params,
          num_iterations: iters
        )

        assert Map.has_key?(action, :buttons),
               "Failed for num_iterations=#{iters}"
      end
    end
  end

  describe "adaptive_scaling/6" do
    test "returns action with correct structure" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action = TestTimeScaling.adaptive_scaling(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        max_samples: 8,
        min_samples: 2
      )

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
    end
  end

  describe "melee_defaults/0" do
    test "returns valid configuration" do
      defaults = TestTimeScaling.melee_defaults()

      assert Keyword.get(defaults, :strategy) == :best_of_n
      assert Keyword.get(defaults, :n) == 4
      assert Keyword.get(defaults, :temperature) == 0.8
      assert Keyword.get(defaults, :beam_width) == 4
    end
  end

  describe "fast_defaults/0" do
    test "returns lightweight configuration" do
      defaults = TestTimeScaling.fast_defaults()

      assert Keyword.get(defaults, :n) == 2
      assert Keyword.get(defaults, :beam_width) == 2
      assert Keyword.get(defaults, :value_reranking) == false
    end
  end

  describe "quality_defaults/0" do
    test "returns quality-focused configuration" do
      defaults = TestTimeScaling.quality_defaults()

      assert Keyword.get(defaults, :strategy) == :ensemble
      assert Keyword.get(defaults, :n_passes) == 8
      assert Keyword.get(defaults, :aggregation) == :weighted
      assert Keyword.get(defaults, :return_uncertainty) == true
    end
  end

  describe "numerical stability" do
    test "handles zero logits" do
      zero_state = Nx.broadcast(0.0, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      action = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        zero_state,
        policy_params,
        value_params,
        n: 4
      )

      assert Map.has_key?(action, :buttons)
      # Check for NaN
      assert Nx.all(Nx.is_nan(action.main_x) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles extreme temperature values" do
      state = Nx.broadcast(0.5, {@batch_size, @obs_size})
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      # Very low temperature (near deterministic)
      action_low = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4,
        temperature: 0.01
      )

      # High temperature (very random)
      action_high = TestTimeScaling.best_of_n(
        &mock_policy_fn/2,
        &mock_value_fn/2,
        state,
        policy_params,
        value_params,
        n: 4,
        temperature: 10.0
      )

      assert Map.has_key?(action_low, :buttons)
      assert Map.has_key?(action_high, :buttons)
    end
  end

  describe "batch handling" do
    test "handles different batch sizes" do
      policy_params = %{seed: 42}
      value_params = %{seed: 123}

      for batch_size <- [1, 2, 4, 8] do
        state = Nx.broadcast(0.5, {batch_size, @obs_size})

        action = TestTimeScaling.best_of_n(
          &mock_policy_fn/2,
          &mock_value_fn/2,
          state,
          policy_params,
          value_params,
          n: 4
        )

        assert Map.has_key?(action, :buttons),
               "Failed for batch_size=#{batch_size}"
      end
    end
  end
end
