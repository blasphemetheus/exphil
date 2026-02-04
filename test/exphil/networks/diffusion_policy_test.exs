defmodule ExPhil.Networks.DiffusionPolicyTest do
  @moduledoc """
  Tests for the Diffusion Policy implementation.

  Tests cover the DDPM-based action diffusion from
  "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (Chi et al., 2023).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.DiffusionPolicy

  @obs_size 64
  @action_dim 32
  @action_horizon 4
  @batch_size 4

  describe "make_schedule/1" do
    test "creates valid noise schedule" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      assert Map.has_key?(schedule, :betas)
      assert Map.has_key?(schedule, :alphas)
      assert Map.has_key?(schedule, :alphas_cumprod)
      assert Map.has_key?(schedule, :sqrt_alphas_cumprod)
      assert Map.has_key?(schedule, :sqrt_one_minus_alphas_cumprod)

      assert Nx.shape(schedule.betas) == {100}
      assert Nx.shape(schedule.alphas_cumprod) == {100}
    end

    test "betas are in valid range" do
      schedule = DiffusionPolicy.make_schedule(
        num_steps: 100,
        beta_start: 1.0e-4,
        beta_end: 0.02
      )

      min_beta = Nx.to_number(Nx.reduce_min(schedule.betas))
      max_beta = Nx.to_number(Nx.reduce_max(schedule.betas))

      # Allow for floating point tolerance
      assert_in_delta min_beta, 1.0e-4, 1.0e-6
      assert max_beta <= 0.021
    end

    test "alphas_cumprod is monotonically decreasing" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      alphas = Nx.to_flat_list(schedule.alphas_cumprod)
      pairs = Enum.zip(alphas, tl(alphas))

      assert Enum.all?(pairs, fn {a, b} -> a >= b end)
    end

    test "alphas_cumprod starts near 1 and decays" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      first = Nx.to_number(schedule.alphas_cumprod[0])
      last = Nx.to_number(schedule.alphas_cumprod[99])

      # First should be close to 1
      assert first > 0.99
      # Last should be significantly smaller than first
      assert last < first * 0.5
    end
  end

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        DiffusionPolicy.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 64,
          num_layers: 2
        )

      {init_fn, predict_fn} = Axon.build(model)

      # Create inputs
      noisy_actions = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      timestep = Nx.tensor([10, 20, 30, 40])
      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})

      params = init_fn.(
        %{
          "noisy_actions" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
          "timestep" => Nx.template({@batch_size}, :s64),
          "observations" => Nx.template({@batch_size, @obs_size}, :f32)
        },
        Axon.ModelState.empty()
      )

      output = predict_fn.(params, %{
        "noisy_actions" => noisy_actions,
        "timestep" => timestep,
        "observations" => observations
      })

      # Output should be predicted noise with same shape as actions
      assert Nx.shape(output) == {@batch_size, @action_horizon, @action_dim}
    end

    test "handles different action dimensions" do
      for action_dim <- [16, 32, 64] do
        model =
          DiffusionPolicy.build(
            obs_size: @obs_size,
            action_dim: action_dim,
            action_horizon: @action_horizon,
            hidden_size: 32,
            num_layers: 1
          )

        {init_fn, predict_fn} = Axon.build(model)

        noisy_actions = Nx.broadcast(0.5, {@batch_size, @action_horizon, action_dim})
        timestep = Nx.tensor([10, 20, 30, 40])
        observations = Nx.broadcast(0.5, {@batch_size, @obs_size})

        params = init_fn.(
          %{
            "noisy_actions" => Nx.template({@batch_size, @action_horizon, action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :s64),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

        output = predict_fn.(params, %{
          "noisy_actions" => noisy_actions,
          "timestep" => timestep,
          "observations" => observations
        })

        assert Nx.shape(output) == {@batch_size, @action_horizon, action_dim},
               "Failed for action_dim=#{action_dim}"
      end
    end

    test "handles different action horizons" do
      for horizon <- [2, 4, 8] do
        model =
          DiffusionPolicy.build(
            obs_size: @obs_size,
            action_dim: @action_dim,
            action_horizon: horizon,
            hidden_size: 32,
            num_layers: 1
          )

        {init_fn, predict_fn} = Axon.build(model)

        noisy_actions = Nx.broadcast(0.5, {@batch_size, horizon, @action_dim})
        timestep = Nx.tensor([10, 20, 30, 40])
        observations = Nx.broadcast(0.5, {@batch_size, @obs_size})

        params = init_fn.(
          %{
            "noisy_actions" => Nx.template({@batch_size, horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :s64),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

        output = predict_fn.(params, %{
          "noisy_actions" => noisy_actions,
          "timestep" => timestep,
          "observations" => observations
        })

        assert Nx.shape(output) == {@batch_size, horizon, @action_dim},
               "Failed for action_horizon=#{horizon}"
      end
    end
  end

  describe "q_sample/4" do
    test "forward diffusion adds noise correctly" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      actions = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})
      noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      timestep = Nx.tensor([0, 0, 0, 0])

      # At t=0, should be mostly original actions
      noisy = DiffusionPolicy.q_sample(actions, timestep, noise, schedule)

      # With zero noise and t=0, should be very close to original
      diff = Nx.subtract(noisy, actions)
      max_diff = Nx.to_number(Nx.reduce_max(Nx.abs(diff)))

      assert max_diff < 0.01
    end

    test "more noise at higher timesteps" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      actions = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})
      key = Nx.Random.key(42)
      {noise, _} = Nx.Random.normal(key, shape: {@batch_size, @action_horizon, @action_dim})

      timestep_low = Nx.tensor([10, 10, 10, 10])
      timestep_high = Nx.tensor([90, 90, 90, 90])

      noisy_low = DiffusionPolicy.q_sample(actions, timestep_low, noise, schedule)
      noisy_high = DiffusionPolicy.q_sample(actions, timestep_high, noise, schedule)

      # Variance should be higher at high timesteps
      var_low = Nx.to_number(Nx.variance(noisy_low))
      var_high = Nx.to_number(Nx.variance(noisy_high))

      assert var_high > var_low
    end
  end

  describe "compute_loss/2" do
    test "returns zero for identical tensors" do
      noise = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      loss = DiffusionPolicy.compute_loss(noise, noise)

      assert Nx.to_number(loss) < 1.0e-6
    end

    test "returns positive loss for different tensors" do
      true_noise = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(0.3, {@batch_size, @action_horizon, @action_dim})

      loss = DiffusionPolicy.compute_loss(true_noise, pred_noise)

      assert Nx.to_number(loss) > 0
    end

    test "loss increases with larger differences" do
      true_noise = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})

      pred_small_diff = Nx.broadcast(0.4, {@batch_size, @action_horizon, @action_dim})
      pred_large_diff = Nx.broadcast(0.1, {@batch_size, @action_horizon, @action_dim})

      loss_small = DiffusionPolicy.compute_loss(true_noise, pred_small_diff)
      loss_large = DiffusionPolicy.compute_loss(true_noise, pred_large_diff)

      assert Nx.to_number(loss_large) > Nx.to_number(loss_small)
    end
  end

  describe "p_sample/5" do
    test "denoising step produces valid output" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      noisy_actions = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      predicted_noise = Nx.broadcast(0.1, {@batch_size, @action_horizon, @action_dim})
      timestep = Nx.tensor([50, 50, 50, 50])
      random_noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})

      denoised = DiffusionPolicy.p_sample(
        noisy_actions, predicted_noise, timestep, random_noise, schedule
      )

      assert Nx.shape(denoised) == {@batch_size, @action_horizon, @action_dim}

      # Should be finite
      assert Nx.all(Nx.is_nan(denoised) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(denoised) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "no noise added at t=0" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      noisy_actions = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      predicted_noise = Nx.broadcast(0.1, {@batch_size, @action_horizon, @action_dim})
      timestep = Nx.tensor([0, 0, 0, 0])
      random_noise = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})

      # Even with large random_noise, t=0 should not add it
      denoised1 = DiffusionPolicy.p_sample(
        noisy_actions, predicted_noise, timestep, random_noise, schedule
      )

      zero_noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      denoised2 = DiffusionPolicy.p_sample(
        noisy_actions, predicted_noise, timestep, zero_noise, schedule
      )

      # Results should be identical at t=0
      diff = Nx.subtract(denoised1, denoised2)
      max_diff = Nx.to_number(Nx.reduce_max(Nx.abs(diff)))

      assert max_diff < 1.0e-5
    end
  end

  describe "output_size/1" do
    test "returns action_horizon * action_dim" do
      assert DiffusionPolicy.output_size(action_dim: 32, action_horizon: 4) == 128
      assert DiffusionPolicy.output_size(action_dim: 64, action_horizon: 8) == 512
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        DiffusionPolicy.param_count(
          obs_size: 287,
          action_dim: 64,
          action_horizon: 8,
          hidden_size: 256,
          num_layers: 4
        )

      # Should have significant params
      assert count > 500_000
      # But not unreasonably large
      assert count < 10_000_000
    end

    test "scales with num_layers" do
      base_opts = [
        obs_size: 64,
        action_dim: 32,
        action_horizon: 4,
        hidden_size: 128
      ]

      count_2 = DiffusionPolicy.param_count(Keyword.put(base_opts, :num_layers, 2))
      count_4 = DiffusionPolicy.param_count(Keyword.put(base_opts, :num_layers, 4))

      assert count_4 > count_2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = DiffusionPolicy.melee_defaults()

      assert Keyword.get(defaults, :action_dim) == 64
      assert Keyword.get(defaults, :action_horizon) == 8
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_steps) == 100
    end
  end

  describe "fast_inference_defaults/0" do
    test "returns configuration with fewer steps" do
      defaults = DiffusionPolicy.fast_inference_defaults()

      assert Keyword.get(defaults, :num_steps) == 20
      assert Keyword.get(defaults, :action_horizon) == 4
    end
  end

  describe "numerical stability" do
    test "model produces finite outputs" do
      model =
        DiffusionPolicy.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 2
        )

      {init_fn, predict_fn} = Axon.build(model)

      key = Nx.Random.key(42)
      {noisy_actions, key} = Nx.Random.normal(key, shape: {@batch_size, @action_horizon, @action_dim})
      timestep = Nx.tensor([10, 30, 50, 70])
      {observations, _} = Nx.Random.normal(key, shape: {@batch_size, @obs_size})

      params = init_fn.(
        %{
          "noisy_actions" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
          "timestep" => Nx.template({@batch_size}, :s64),
          "observations" => Nx.template({@batch_size, @obs_size}, :f32)
        },
        Axon.ModelState.empty()
      )

      output = predict_fn.(params, %{
        "noisy_actions" => noisy_actions,
        "timestep" => timestep,
        "observations" => observations
      })

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles edge timesteps (0 and max)" do
      schedule = DiffusionPolicy.make_schedule(num_steps: 100)

      actions = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      noise = Nx.broadcast(0.1, {@batch_size, @action_horizon, @action_dim})

      # Test t=0
      timestep_zero = Nx.tensor([0, 0, 0, 0])
      noisy_zero = DiffusionPolicy.q_sample(actions, timestep_zero, noise, schedule)
      assert Nx.all(Nx.is_nan(noisy_zero) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Test t=99 (max)
      timestep_max = Nx.tensor([99, 99, 99, 99])
      noisy_max = DiffusionPolicy.q_sample(actions, timestep_max, noise, schedule)
      assert Nx.all(Nx.is_nan(noisy_max) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
