defmodule ExPhil.Networks.WorldModelTest do
  @moduledoc """
  Tests for the World Model network.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.WorldModel

  @state_dim 64
  @action_dim 13
  @hidden_size 32
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model = WorldModel.build(
        state_dim: @state_dim,
        action_dim: @action_dim,
        hidden_size: @hidden_size,
        num_layers: 2
      )

      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        %{
          "state" => Nx.template({@batch_size, @state_dim}, :f32),
          "action" => Nx.template({@batch_size, @action_dim}, :f32)
        },
        Axon.ModelState.empty()
      )

      state = Nx.broadcast(0.5, {@batch_size, @state_dim})
      action = Nx.broadcast(0.5, {@batch_size, @action_dim})

      output = predict_fn.(params, %{
        "state" => state,
        "action" => action
      })

      # Check output shapes
      assert Nx.shape(output.next_state) == {@batch_size, @state_dim}
      assert Nx.shape(output.reward) == {@batch_size, 1}
      assert Nx.shape(output.done) == {@batch_size, 1}
    end

    test "builds model without reward/done heads" do
      model = WorldModel.build(
        state_dim: @state_dim,
        action_dim: @action_dim,
        hidden_size: @hidden_size,
        predict_reward: false,
        predict_done: false
      )

      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        %{
          "state" => Nx.template({@batch_size, @state_dim}, :f32),
          "action" => Nx.template({@batch_size, @action_dim}, :f32)
        },
        Axon.ModelState.empty()
      )

      state = Nx.broadcast(0.5, {@batch_size, @state_dim})
      action = Nx.broadcast(0.5, {@batch_size, @action_dim})

      output = predict_fn.(params, %{
        "state" => state,
        "action" => action
      })

      assert Map.has_key?(output, :next_state)
      refute Map.has_key?(output, :reward)
      refute Map.has_key?(output, :done)
    end

    test "residual prediction adds to input state" do
      model = WorldModel.build(
        state_dim: @state_dim,
        action_dim: @action_dim,
        hidden_size: @hidden_size,
        residual_prediction: true,
        predict_reward: false,
        predict_done: false
      )

      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        %{
          "state" => Nx.template({@batch_size, @state_dim}, :f32),
          "action" => Nx.template({@batch_size, @action_dim}, :f32)
        },
        Axon.ModelState.empty()
      )

      # Use specific state values
      state = Nx.broadcast(1.0, {@batch_size, @state_dim})
      action = Nx.broadcast(0.0, {@batch_size, @action_dim})

      output = predict_fn.(params, %{
        "state" => state,
        "action" => action
      })

      # Next state should be different from zero (residual adds to input)
      # With zero action, the delta should be small but not exactly cancel the input
      next_state_mean = Nx.mean(output.next_state) |> Nx.to_number()
      assert next_state_mean != 0.0
    end
  end

  describe "compute_loss/3" do
    test "computes combined loss" do
      predictions = %{
        next_state: Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        reward: Nx.tensor([[0.5], [0.8]]),
        done: Nx.tensor([[0.1], [0.9]])
      }

      targets = %{
        next_state: Nx.tensor([[1.1, 2.1], [3.1, 4.1]]),
        reward: Nx.tensor([[0.6], [0.7]]),
        done: Nx.tensor([[0.0], [1.0]])
      }

      loss = WorldModel.compute_loss(predictions, targets)

      # Loss should be a scalar
      assert Nx.shape(loss) == {}
      # Loss should be positive
      assert Nx.to_number(loss) > 0
    end

    test "respects custom weights" do
      predictions = %{
        next_state: Nx.tensor([[1.0, 2.0]]),
        reward: Nx.tensor([[0.5]]),
        done: Nx.tensor([[0.5]])
      }

      targets = %{
        next_state: Nx.tensor([[2.0, 3.0]]),
        reward: Nx.tensor([[1.0]]),
        done: Nx.tensor([[1.0]])
      }

      loss_default = WorldModel.compute_loss(predictions, targets)
      loss_high_reward = WorldModel.compute_loss(predictions, targets, reward_weight: 10.0)

      # Higher reward weight should increase loss
      assert Nx.to_number(loss_high_reward) > Nx.to_number(loss_default)
    end
  end

  describe "build_ensemble/1" do
    test "creates multiple models" do
      ensemble = WorldModel.build_ensemble(
        state_dim: @state_dim,
        action_dim: @action_dim,
        hidden_size: @hidden_size,
        num_models: 3
      )

      assert length(ensemble.models) == 3
      assert ensemble.num_models == 3
      assert is_function(ensemble.aggregate_fn, 2)
    end
  end

  describe "melee_defaults/0" do
    test "returns valid configuration" do
      defaults = WorldModel.melee_defaults()

      assert Keyword.get(defaults, :state_dim) == 287
      assert Keyword.get(defaults, :action_dim) == 13
      assert Keyword.get(defaults, :hidden_size) == 512
      assert Keyword.get(defaults, :predict_reward) == true
      assert Keyword.get(defaults, :residual_prediction) == true
    end

    test "builds valid model with defaults" do
      defaults = WorldModel.melee_defaults()
      model = WorldModel.build(defaults)

      {init_fn, _predict_fn} = Axon.build(model)

      params = init_fn.(
        %{
          "state" => Nx.template({1, 287}, :f32),
          "action" => Nx.template({1, 13}, :f32)
        },
        Axon.ModelState.empty()
      )

      assert %Axon.ModelState{} = params
    end
  end

  describe "param_count/1" do
    test "returns reasonable count" do
      count = WorldModel.param_count(
        state_dim: 287,
        action_dim: 13,
        hidden_size: 512,
        num_layers: 3
      )

      # Should have significant params
      assert count > 100_000
      # But not unreasonable
      assert count < 10_000_000
    end

    test "scales with hidden size" do
      count_small = WorldModel.param_count(hidden_size: 256)
      count_large = WorldModel.param_count(hidden_size: 512)

      assert count_large > count_small
    end
  end

  describe "numerical stability" do
    test "produces finite outputs" do
      model = WorldModel.build(
        state_dim: @state_dim,
        action_dim: @action_dim,
        hidden_size: @hidden_size
      )

      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        %{
          "state" => Nx.template({@batch_size, @state_dim}, :f32),
          "action" => Nx.template({@batch_size, @action_dim}, :f32)
        },
        Axon.ModelState.empty()
      )

      # Random input
      key = Nx.Random.key(42)
      {state, key} = Nx.Random.uniform(key, shape: {@batch_size, @state_dim})
      {action, _} = Nx.Random.uniform(key, shape: {@batch_size, @action_dim})

      output = predict_fn.(params, %{
        "state" => state,
        "action" => action
      })

      # Check no NaN
      assert Nx.all(Nx.is_nan(output.next_state) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output.reward) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output.done) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "done output is bounded [0, 1]" do
      model = WorldModel.build(
        state_dim: @state_dim,
        action_dim: @action_dim,
        hidden_size: @hidden_size
      )

      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        %{
          "state" => Nx.template({@batch_size, @state_dim}, :f32),
          "action" => Nx.template({@batch_size, @action_dim}, :f32)
        },
        Axon.ModelState.empty()
      )

      state = Nx.broadcast(0.5, {@batch_size, @state_dim})
      action = Nx.broadcast(0.5, {@batch_size, @action_dim})

      output = predict_fn.(params, %{
        "state" => state,
        "action" => action
      })

      # Done should be in [0, 1] due to sigmoid
      min_done = Nx.reduce_min(output.done) |> Nx.to_number()
      max_done = Nx.reduce_max(output.done) |> Nx.to_number()

      assert min_done >= 0.0
      assert max_done <= 1.0
    end
  end
end
