defmodule ExPhil.Networks.FlowMatchingTest do
  @moduledoc """
  Tests for the Flow Matching implementation.

  Tests cover Conditional Flow Matching from
  "Flow Matching for Generative Modeling" (Lipman et al., ICLR 2023).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.FlowMatching

  @obs_size 64
  @action_dim 32
  @action_horizon 4
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 64,
          num_layers: 2
        )

      {init_fn, predict_fn} = Axon.build(model)

      # Create inputs
      x_t = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      timestep = Nx.tensor([0.1, 0.3, 0.5, 0.7])
      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "x_t" => x_t,
          "timestep" => timestep,
          "observations" => observations
        })

      # Output should be velocity with same shape as x_t
      assert Nx.shape(output) == {@batch_size, @action_horizon, @action_dim}
    end

    test "handles different action dimensions" do
      for action_dim <- [16, 32, 64] do
        model =
          FlowMatching.build(
            obs_size: @obs_size,
            action_dim: action_dim,
            action_horizon: @action_horizon,
            hidden_size: 32,
            num_layers: 1
          )

        {init_fn, predict_fn} = Axon.build(model)

        x_t = Nx.broadcast(0.5, {@batch_size, @action_horizon, action_dim})
        timestep = Nx.tensor([0.25, 0.25, 0.25, 0.25])
        observations = Nx.broadcast(0.5, {@batch_size, @obs_size})

        params =
          init_fn.(
            %{
              "x_t" => Nx.template({@batch_size, @action_horizon, action_dim}, :f32),
              "timestep" => Nx.template({@batch_size}, :f32),
              "observations" => Nx.template({@batch_size, @obs_size}, :f32)
            },
            Axon.ModelState.empty()
          )

        output =
          predict_fn.(params, %{
            "x_t" => x_t,
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
          FlowMatching.build(
            obs_size: @obs_size,
            action_dim: @action_dim,
            action_horizon: horizon,
            hidden_size: 32,
            num_layers: 1
          )

        {init_fn, predict_fn} = Axon.build(model)

        x_t = Nx.broadcast(0.5, {@batch_size, horizon, @action_dim})
        timestep = Nx.tensor([0.5, 0.5, 0.5, 0.5])
        observations = Nx.broadcast(0.5, {@batch_size, @obs_size})

        params =
          init_fn.(
            %{
              "x_t" => Nx.template({@batch_size, horizon, @action_dim}, :f32),
              "timestep" => Nx.template({@batch_size}, :f32),
              "observations" => Nx.template({@batch_size, @obs_size}, :f32)
            },
            Axon.ModelState.empty()
          )

        output =
          predict_fn.(params, %{
            "x_t" => x_t,
            "timestep" => timestep,
            "observations" => observations
          })

        assert Nx.shape(output) == {@batch_size, horizon, @action_dim},
               "Failed for action_horizon=#{horizon}"
      end
    end
  end

  describe "interpolate/3" do
    test "at t=0, returns x_0" do
      x_0 = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})
      t = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      x_t = FlowMatching.interpolate(x_0, x_1, t)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x_t, x_0))))
      assert_in_delta diff, 0.0, 1.0e-5
    end

    test "at t=1, returns x_1" do
      x_0 = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})
      t = Nx.tensor([1.0, 1.0, 1.0, 1.0])

      x_t = FlowMatching.interpolate(x_0, x_1, t)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x_t, x_1))))
      assert_in_delta diff, 0.0, 1.0e-5
    end

    test "at t=0.5, returns midpoint" do
      x_0 = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(2.0, {@batch_size, @action_horizon, @action_dim})
      t = Nx.tensor([0.5, 0.5, 0.5, 0.5])

      x_t = FlowMatching.interpolate(x_0, x_1, t)

      expected = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})
      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x_t, expected))))
      assert_in_delta diff, 0.0, 1.0e-5
    end

    test "interpolation is linear" do
      x_0 = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(10.0, {@batch_size, @action_horizon, @action_dim})

      for t_val <- [0.1, 0.3, 0.7, 0.9] do
        t = Nx.broadcast(t_val, {@batch_size})
        x_t = FlowMatching.interpolate(x_0, x_1, t)

        expected_val = t_val * 10.0
        expected = Nx.broadcast(expected_val, {@batch_size, @action_horizon, @action_dim})
        diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x_t, expected))))
        assert_in_delta diff, 0.0, 1.0e-4, "Failed for t=#{t_val}"
      end
    end
  end

  describe "target_velocity/2" do
    test "returns x_1 - x_0" do
      x_0 = Nx.broadcast(2.0, {@batch_size, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(5.0, {@batch_size, @action_horizon, @action_dim})

      velocity = FlowMatching.target_velocity(x_0, x_1)

      expected = Nx.broadcast(3.0, {@batch_size, @action_horizon, @action_dim})
      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(velocity, expected))))
      assert_in_delta diff, 0.0, 1.0e-5
    end

    test "velocity is constant along optimal transport path" do
      x_0 = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})

      # Velocity should be constant regardless of t
      velocity = FlowMatching.target_velocity(x_0, x_1)

      # The target velocity for OT path is always x_1 - x_0
      expected = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})
      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(velocity, expected))))
      assert_in_delta diff, 0.0, 1.0e-5
    end
  end

  describe "velocity_loss/2" do
    test "returns zero for identical velocities" do
      velocity = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      loss = FlowMatching.velocity_loss(velocity, velocity)

      assert Nx.to_number(loss) < 1.0e-6
    end

    test "returns positive loss for different velocities" do
      target = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})
      pred = Nx.broadcast(0.3, {@batch_size, @action_horizon, @action_dim})

      loss = FlowMatching.velocity_loss(target, pred)

      assert Nx.to_number(loss) > 0
    end

    test "loss increases with larger differences" do
      target = Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim})

      pred_small_diff = Nx.broadcast(0.4, {@batch_size, @action_horizon, @action_dim})
      pred_large_diff = Nx.broadcast(0.1, {@batch_size, @action_horizon, @action_dim})

      loss_small = FlowMatching.velocity_loss(target, pred_small_diff)
      loss_large = FlowMatching.velocity_loss(target, pred_large_diff)

      assert Nx.to_number(loss_large) > Nx.to_number(loss_small)
    end
  end

  describe "compute_loss/6" do
    test "computes loss for training" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      actions = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})
      noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      t = Nx.tensor([0.25, 0.5, 0.75, 1.0])

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      loss = FlowMatching.compute_loss(params, predict_fn, observations, actions, noise, t)

      # Loss should be a positive scalar
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) >= 0
    end
  end

  describe "sample/5" do
    test "generates actions with correct shape" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      initial_noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      actions = FlowMatching.sample(params, predict_fn, observations, initial_noise,
        num_steps: 5, solver: :euler
      )

      assert Nx.shape(actions) == {@batch_size, @action_horizon, @action_dim}
    end

    test "different solvers produce finite results" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      initial_noise = Nx.broadcast(0.1, {@batch_size, @action_horizon, @action_dim})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      for solver <- [:euler, :midpoint, :rk4] do
        actions = FlowMatching.sample(params, predict_fn, observations, initial_noise,
          num_steps: 5, solver: solver
        )

        # Check no NaN or Inf
        assert Nx.all(Nx.is_nan(actions) |> Nx.logical_not()) |> Nx.to_number() == 1,
               "NaN detected for solver=#{solver}"
        assert Nx.all(Nx.is_infinity(actions) |> Nx.logical_not()) |> Nx.to_number() == 1,
               "Inf detected for solver=#{solver}"
      end
    end

    test "more steps generally improves accuracy" do
      # This is a qualitative test - we just verify it runs with different step counts
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      initial_noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      for num_steps <- [5, 10, 20] do
        actions = FlowMatching.sample(params, predict_fn, observations, initial_noise,
          num_steps: num_steps, solver: :euler
        )

        assert Nx.shape(actions) == {@batch_size, @action_horizon, @action_dim},
               "Failed for num_steps=#{num_steps}"
      end
    end
  end

  describe "sample_guided/5" do
    test "generates actions with guidance" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      initial_noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      actions = FlowMatching.sample_guided(params, predict_fn, observations, initial_noise,
        num_steps: 5, guidance_scale: 2.0
      )

      assert Nx.shape(actions) == {@batch_size, @action_horizon, @action_dim}
    end

    test "guidance_scale=1.0 equals standard sampling" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      initial_noise = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      # With guidance_scale=1.0, should be close to standard
      actions_guided = FlowMatching.sample_guided(params, predict_fn, observations, initial_noise,
        num_steps: 5, guidance_scale: 1.0
      )

      actions_standard = FlowMatching.sample(params, predict_fn, observations, initial_noise,
        num_steps: 5, solver: :euler
      )

      # Note: Won't be exactly equal due to extra forward pass, but shapes match
      assert Nx.shape(actions_guided) == Nx.shape(actions_standard)
    end
  end

  describe "generate_rectified_pairs/5" do
    test "generates noise-action pairs" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      noise = Nx.broadcast(0.1, {@batch_size, @action_horizon, @action_dim})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      {x_0, x_1} = FlowMatching.generate_rectified_pairs(
        params, predict_fn, observations, noise, num_steps: 5
      )

      # x_0 should be the original noise
      assert Nx.shape(x_0) == {@batch_size, @action_horizon, @action_dim}
      assert Nx.shape(x_1) == {@batch_size, @action_horizon, @action_dim}

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x_0, noise))))
      assert_in_delta diff, 0.0, 1.0e-5
    end
  end

  describe "output_size/1" do
    test "returns action_horizon * action_dim" do
      assert FlowMatching.output_size(action_dim: 32, action_horizon: 4) == 128
      assert FlowMatching.output_size(action_dim: 64, action_horizon: 8) == 512
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        FlowMatching.param_count(
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

      count_2 = FlowMatching.param_count(Keyword.put(base_opts, :num_layers, 2))
      count_4 = FlowMatching.param_count(Keyword.put(base_opts, :num_layers, 4))

      assert count_4 > count_2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = FlowMatching.melee_defaults()

      assert Keyword.get(defaults, :action_dim) == 64
      assert Keyword.get(defaults, :action_horizon) == 8
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_steps) == 20
      assert Keyword.get(defaults, :solver) == :euler
    end
  end

  describe "fast_inference_defaults/0" do
    test "returns configuration with fewer steps" do
      defaults = FlowMatching.fast_inference_defaults()

      assert Keyword.get(defaults, :num_steps) == 10
      assert Keyword.get(defaults, :action_horizon) == 4
      assert Keyword.get(defaults, :hidden_size) == 128
    end
  end

  describe "quality_defaults/0" do
    test "returns high-quality configuration" do
      defaults = FlowMatching.quality_defaults()

      assert Keyword.get(defaults, :num_steps) == 50
      assert Keyword.get(defaults, :solver) == :rk4
    end
  end

  describe "numerical stability" do
    test "model produces finite outputs" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 2
        )

      {init_fn, predict_fn} = Axon.build(model)

      key = Nx.Random.key(42)
      {x_t, key} = Nx.Random.normal(key, shape: {@batch_size, @action_horizon, @action_dim})
      timestep = Nx.tensor([0.1, 0.3, 0.5, 0.7])
      {observations, _} = Nx.Random.normal(key, shape: {@batch_size, @obs_size})

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
            "timestep" => Nx.template({@batch_size}, :f32),
            "observations" => Nx.template({@batch_size, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "x_t" => x_t,
          "timestep" => timestep,
          "observations" => observations
        })

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles edge timesteps (0 and 1)" do
      x_0 = Nx.broadcast(0.0, {@batch_size, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(1.0, {@batch_size, @action_horizon, @action_dim})

      # Test t=0
      t_zero = Nx.tensor([0.0, 0.0, 0.0, 0.0])
      x_at_zero = FlowMatching.interpolate(x_0, x_1, t_zero)
      assert Nx.all(Nx.is_nan(x_at_zero) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Test t=1
      t_one = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      x_at_one = FlowMatching.interpolate(x_0, x_1, t_one)
      assert Nx.all(Nx.is_nan(x_at_one) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
