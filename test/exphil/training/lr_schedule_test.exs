defmodule ExPhil.Training.LRScheduleTest do
  @moduledoc """
  Tests for learning rate scheduling in the Imitation module.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.Imitation

  describe "create_optimizer/1" do
    test "creates optimizer with constant schedule" do
      config = %{
        learning_rate: 1.0e-4,
        lr_schedule: :constant,
        warmup_steps: 0,
        weight_decay: 0.01
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "creates optimizer with cosine schedule" do
      config = %{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine,
        decay_steps: 1000,
        warmup_steps: 0,
        weight_decay: 0.01
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "creates optimizer with cosine_restarts schedule" do
      config = %{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 0,
        restart_period: 100,
        restart_mult: 2,
        weight_decay: 0.01
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "creates optimizer with warmup" do
      config = %{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine,
        warmup_steps: 100,
        decay_steps: 1000,
        weight_decay: 0.01
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "creates optimizer with cosine_restarts and warmup" do
      config = %{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 50,
        restart_period: 100,
        restart_mult: 2,
        weight_decay: 0.01
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end
  end

  describe "cosine_restarts schedule behavior" do
    # Test that the schedule computes correctly by running a few training steps
    # and observing the optimizer's behavior through gradient updates

    test "cosine_restarts produces valid LR values" do
      config = %{
        learning_rate: 1.0e-3,
        lr_schedule: :cosine_restarts,
        warmup_steps: 0,
        restart_period: 10,
        # Fixed period for easier testing
        restart_mult: 1,
        weight_decay: 0.01
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)

      # Create simple params
      params = %{w: Nx.tensor([1.0, 2.0, 3.0])}
      state = init_fn.(params)

      # Run a few steps and verify params change (indicating optimizer works)
      gradients = %{w: Nx.tensor([0.1, 0.1, 0.1])}

      {updated_params, _new_state} = update_fn.(gradients, state, params)

      # Verify params were updated
      original_sum = Nx.sum(params.w) |> Nx.to_number()
      updated_sum = Nx.sum(updated_params.w) |> Nx.to_number()
      assert original_sum != updated_sum
    end

    test "cosine_restarts with growing periods produces valid LR values" do
      config = %{
        learning_rate: 1.0e-3,
        lr_schedule: :cosine_restarts,
        warmup_steps: 0,
        restart_period: 5,
        # Periods: 5, 10, 20, ...
        restart_mult: 2,
        weight_decay: 0.01
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)

      params = %{w: Nx.tensor([1.0])}
      state = init_fn.(params)
      gradients = %{w: Nx.tensor([0.1])}

      # Run multiple steps to go through restarts
      {params, _state} =
        Enum.reduce(1..20, {params, state}, fn _, {p, s} ->
          update_fn.(gradients, s, p)
        end)

      # Just verify it doesn't crash and produces valid output
      assert params.w |> Nx.squeeze() |> Nx.to_number() != 1.0
    end
  end
end
