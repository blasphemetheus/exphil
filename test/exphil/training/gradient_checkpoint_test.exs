defmodule ExPhil.Training.GradientCheckpointTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.GradientCheckpoint

  describe "checkpoint/2" do
    test "computes forward pass correctly" do
      # Simple function that doubles input
      fun = fn x -> Nx.multiply(x, 2) end
      input = Nx.tensor([1.0, 2.0, 3.0])

      result = GradientCheckpoint.checkpoint(fun, input)

      expected = Nx.tensor([2.0, 4.0, 6.0])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "preserves tensor shape" do
      fun = fn x -> Nx.add(x, 1) end
      input = Nx.broadcast(0.5, {4, 8})

      result = GradientCheckpoint.checkpoint(fun, input)

      assert Nx.shape(result) == {4, 8}
    end

    test "works with nested operations" do
      fun = fn x ->
        x
        |> Nx.multiply(2)
        |> Nx.add(1)
        |> Nx.pow(2)
      end
      input = Nx.tensor([1.0, 2.0])

      result = GradientCheckpoint.checkpoint(fun, input)

      # (1*2+1)^2 = 9, (2*2+1)^2 = 25
      expected = Nx.tensor([9.0, 25.0])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end
  end

  describe "checkpoint_sequence/3" do
    test "applies functions in sequence" do
      funs = [
        fn x -> Nx.add(x, 1) end,
        fn x -> Nx.multiply(x, 2) end,
        fn x -> Nx.subtract(x, 3) end
      ]
      input = Nx.tensor([5.0])

      result = GradientCheckpoint.checkpoint_sequence(funs, input)

      # (5 + 1) * 2 - 3 = 9
      expected = Nx.tensor([9.0])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "respects checkpoint_every option" do
      funs = [
        fn x -> Nx.add(x, 1) end,
        fn x -> Nx.add(x, 2) end,
        fn x -> Nx.add(x, 3) end,
        fn x -> Nx.add(x, 4) end
      ]
      input = Nx.tensor([0.0])

      result = GradientCheckpoint.checkpoint_sequence(funs, input, checkpoint_every: 2)

      # 0 + 1 + 2 + 3 + 4 = 10
      expected = Nx.tensor([10.0])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "handles single function" do
      funs = [fn x -> Nx.multiply(x, 3) end]
      input = Nx.tensor([4.0])

      result = GradientCheckpoint.checkpoint_sequence(funs, input)

      expected = Nx.tensor([12.0])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end
  end

  describe "estimate_memory_savings/3" do
    test "calculates savings for every-layer checkpointing" do
      result = GradientCheckpoint.estimate_memory_savings(10, 100, 1)

      # No savings when checkpointing every layer (still stores all)
      assert result.without_checkpoint_mb == 1000
      assert result.with_checkpoint_mb == 1000
      assert result.savings_mb == 0
    end

    test "calculates savings for checkpointing every 2 layers" do
      result = GradientCheckpoint.estimate_memory_savings(10, 100, 2)

      assert result.without_checkpoint_mb == 1000
      assert result.with_checkpoint_mb == 500
      assert result.savings_mb == 500
      assert result.savings_percent == 50.0
    end

    test "calculates savings for checkpointing every 5 layers" do
      result = GradientCheckpoint.estimate_memory_savings(10, 100, 5)

      assert result.without_checkpoint_mb == 1000
      assert result.with_checkpoint_mb == 200
      assert result.savings_mb == 800
      assert result.savings_percent == 80.0
    end
  end
end
