defmodule ExPhil.Training.LRScheduleValidationTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Imitation.Optimizer

  describe "cosine_restarts schedule" do
    test "LR starts at base_lr (no warmup)" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 0,
        restart_period: 1000,
        restart_mult: 2
      })

      lr_0 = schedule.(Nx.tensor(0)) |> Nx.to_number()
      assert_in_delta lr_0, 1.0e-4, 1.0e-6
    end

    test "LR starts at base_lr with warmup_steps=1" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 1,
        restart_period: 1000,
        restart_mult: 2
      })

      # Step 0 during warmup should be near 0 (linear ramp)
      lr_0 = schedule.(Nx.tensor(0)) |> Nx.to_number()
      # Step 1 should be at base_lr (warmup complete)
      lr_1 = schedule.(Nx.tensor(1)) |> Nx.to_number()

      assert lr_0 < 1.0e-5, "LR at step 0 should be near 0 during warmup, got #{lr_0}"
      assert_in_delta lr_1, 1.0e-4, 1.0e-5
    end

    test "LR decays during period" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 0,
        restart_period: 100,
        restart_mult: 1
      })

      lr_0 = schedule.(Nx.tensor(0)) |> Nx.to_number()
      lr_50 = schedule.(Nx.tensor(50)) |> Nx.to_number()
      lr_99 = schedule.(Nx.tensor(99)) |> Nx.to_number()

      assert lr_0 > lr_50, "LR should decrease: step 0 (#{lr_0}) > step 50 (#{lr_50})"
      assert lr_50 > lr_99, "LR should decrease: step 50 (#{lr_50}) > step 99 (#{lr_99})"
    end

    test "LR restarts after period" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 0,
        restart_period: 100,
        restart_mult: 1
      })

      lr_99 = schedule.(Nx.tensor(99)) |> Nx.to_number()
      lr_100 = schedule.(Nx.tensor(100)) |> Nx.to_number()

      assert lr_100 > lr_99, "LR should restart: step 100 (#{lr_100}) > step 99 (#{lr_99})"
      assert_in_delta lr_100, 1.0e-4, 1.0e-5
    end

    test "LR never goes negative" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 10,
        restart_period: 100,
        restart_mult: 2
      })

      for step <- [0, 1, 5, 10, 50, 99, 100, 150, 300, 1000] do
        lr = schedule.(Nx.tensor(step)) |> Nx.to_number()
        assert lr >= 0, "LR should be non-negative at step #{step}, got #{lr}"
      end
    end

    test "LR never exceeds base_lr" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine_restarts,
        warmup_steps: 0,
        restart_period: 100,
        restart_mult: 2
      })

      for step <- [0, 1, 50, 99, 100, 200, 500, 1000] do
        lr = schedule.(Nx.tensor(step)) |> Nx.to_number()
        assert lr <= 1.0e-4 + 1.0e-6, "LR should not exceed base at step #{step}, got #{lr}"
      end
    end
  end

  describe "constant schedule" do
    test "LR is always base_lr" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :constant,
        warmup_steps: 0
      })

      for step <- [0, 1, 100, 10000] do
        lr = schedule.(Nx.tensor(step)) |> Nx.to_number()
        assert_in_delta lr, 1.0e-4, 1.0e-6
      end
    end
  end

  describe "cosine schedule" do
    test "decays to near zero" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :cosine,
        warmup_steps: 0,
        decay_steps: 1000
      })

      lr_0 = schedule.(Nx.tensor(0)) |> Nx.to_number()
      lr_1000 = schedule.(Nx.tensor(1000)) |> Nx.to_number()

      assert_in_delta lr_0, 1.0e-4, 1.0e-6
      assert lr_1000 < 1.0e-6, "LR should be near 0 at end, got #{lr_1000}"
    end
  end

  describe "warmup" do
    test "linear warmup from 0 to base_lr" do
      schedule = Optimizer.build_lr_schedule(%{
        learning_rate: 1.0e-4,
        lr_schedule: :constant,
        warmup_steps: 100
      })

      lr_0 = schedule.(Nx.tensor(0)) |> Nx.to_number()
      lr_50 = schedule.(Nx.tensor(50)) |> Nx.to_number()
      lr_100 = schedule.(Nx.tensor(100)) |> Nx.to_number()

      assert lr_0 < 1.0e-5, "LR at step 0 should be near 0, got #{lr_0}"
      assert_in_delta lr_50, 5.0e-5, 1.0e-5
      assert_in_delta lr_100, 1.0e-4, 1.0e-5
    end
  end
end
