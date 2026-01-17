defmodule ExPhil.Networks.ValueTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Value

  # Helper to convert tensor to number, handling {1} shape
  defp tensor_to_number(tensor) do
    case Nx.shape(tensor) do
      {} -> Nx.to_number(tensor)
      {1} -> tensor |> Nx.squeeze() |> Nx.to_number()
      _ -> raise "Expected scalar or {1} tensor, got #{inspect(Nx.shape(tensor))}"
    end
  end

  describe "build/1" do
    test "creates a valid Axon model" do
      model = Value.build(embed_size: 256)

      assert %Axon{} = model
    end

    test "accepts custom hidden sizes" do
      model = Value.build(embed_size: 128, hidden_sizes: [64, 32])

      assert %Axon{} = model
    end

    test "model can be compiled and run" do
      model = Value.build(embed_size: 64, hidden_sizes: [32])

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 64}, :f32), Axon.ModelState.empty())

      # Run forward pass
      input = Nx.broadcast(0.5, {4, 64})
      output = predict_fn.(params, input)

      # Output should be [batch_size] after squeeze
      assert Nx.shape(output) == {4}
    end
  end

  describe "build_from_backbone/1" do
    test "creates value head from existing backbone" do
      backbone = Axon.input("state", shape: {nil, 64})
      |> Axon.dense(32, activation: :relu)

      model = Value.build_from_backbone(backbone)

      assert %Axon{} = model
    end
  end

  describe "value_loss/2" do
    test "computes mean squared error" do
      predicted = Nx.tensor([1.0, 2.0, 3.0])
      targets = Nx.tensor([1.0, 2.0, 3.0])

      loss = Value.value_loss(predicted, targets)

      assert Nx.to_number(loss) == 0.0
    end

    test "returns positive loss for different values" do
      predicted = Nx.tensor([1.0, 2.0, 3.0])
      targets = Nx.tensor([2.0, 3.0, 4.0])

      loss = Value.value_loss(predicted, targets)

      # MSE of [1,1,1] = 1.0
      assert_in_delta Nx.to_number(loss), 1.0, 0.001
    end

    test "handles batched inputs" do
      predicted = Nx.tensor([0.0, 0.0, 0.0, 0.0])
      targets = Nx.tensor([1.0, -1.0, 2.0, -2.0])

      loss = Value.value_loss(predicted, targets)

      # MSE = (1 + 1 + 4 + 4) / 4 = 2.5
      assert_in_delta Nx.to_number(loss), 2.5, 0.001
    end
  end

  describe "huber_loss/3" do
    test "behaves like MSE for small errors" do
      predicted = Nx.tensor([1.0, 1.0])
      targets = Nx.tensor([1.1, 0.9])

      loss = Value.huber_loss(predicted, targets, 1.0)

      # For |diff| < delta, huber = 0.5 * diff^2
      # (0.1^2 + 0.1^2) / 2 * 0.5 = 0.005
      assert_in_delta Nx.to_number(loss), 0.005, 0.001
    end

    test "behaves linearly for large errors" do
      predicted = Nx.tensor([0.0])
      targets = Nx.tensor([10.0])

      huber = Value.huber_loss(predicted, targets, 1.0)
      mse = Value.value_loss(predicted, targets)

      # Huber should be much smaller than MSE for large errors
      assert Nx.to_number(huber) < Nx.to_number(mse)
    end

    test "respects custom delta" do
      predicted = Nx.tensor([0.0])
      targets = Nx.tensor([2.0])

      loss_small_delta = Value.huber_loss(predicted, targets, 0.5)
      loss_large_delta = Value.huber_loss(predicted, targets, 5.0)

      # Larger delta means more quadratic behavior
      assert Nx.to_number(loss_large_delta) > Nx.to_number(loss_small_delta)
    end
  end

  describe "clipped_value_loss/4" do
    test "clips value changes within range" do
      new_values = Nx.tensor([1.0, 2.0, 3.0])
      old_values = Nx.tensor([1.0, 2.0, 3.0])
      targets = Nx.tensor([1.5, 2.5, 3.5])

      loss = Value.clipped_value_loss(new_values, old_values, targets)

      assert is_struct(loss, Nx.Tensor)
      assert Nx.to_number(loss) > 0
    end

    test "uses larger loss between clipped and unclipped" do
      # Large change that should be clipped
      new_values = Nx.tensor([2.0])
      old_values = Nx.tensor([0.0])
      targets = Nx.tensor([1.0])

      loss = Value.clipped_value_loss(new_values, old_values, targets, 0.1)

      assert Nx.to_number(loss) > 0
    end
  end

  describe "compute_gae/5" do
    test "computes advantages and returns" do
      rewards = Nx.tensor([1.0, 1.0, 1.0])
      values = Nx.tensor([0.0, 0.5, 0.8, 1.0])  # time_steps + 1
      dones = Nx.tensor([0.0, 0.0, 0.0])

      {advantages, returns} = Value.compute_gae(rewards, values, dones)

      assert Nx.shape(advantages) == {3}
      assert Nx.shape(returns) == {3}
    end

    test "zeros advantages on episode end" do
      rewards = Nx.tensor([1.0, 1.0, 1.0])
      values = Nx.tensor([0.0, 0.5, 0.8, 1.0])
      dones = Nx.tensor([0.0, 1.0, 0.0])  # Episode ends at step 2

      {advantages, _returns} = Value.compute_gae(rewards, values, dones)

      advantages_list = Nx.to_flat_list(advantages)

      # After episode end, GAE should reset
      # The advantage at step 1 shouldn't propagate past the done
      assert length(advantages_list) == 3
    end

    test "respects gamma parameter" do
      rewards = Nx.tensor([1.0, 1.0])
      values = Nx.tensor([0.0, 0.0, 0.0])
      dones = Nx.tensor([0.0, 0.0])

      {adv_high_gamma, _} = Value.compute_gae(rewards, values, dones, 0.99, 0.95)
      {adv_low_gamma, _} = Value.compute_gae(rewards, values, dones, 0.5, 0.95)

      # Higher gamma = more future reward consideration
      first_adv_high = tensor_to_number(Nx.slice(adv_high_gamma, [0], [1]))
      first_adv_low = tensor_to_number(Nx.slice(adv_low_gamma, [0], [1]))

      assert first_adv_high > first_adv_low
    end
  end

  describe "normalize_advantages/1" do
    test "normalizes to zero mean and unit variance" do
      advantages = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      normalized = Value.normalize_advantages(advantages)

      mean = Nx.to_number(Nx.mean(normalized))
      std = Nx.to_number(Nx.standard_deviation(normalized))

      assert_in_delta mean, 0.0, 0.001
      assert_in_delta std, 1.0, 0.01
    end

    test "handles constant advantages" do
      advantages = Nx.tensor([5.0, 5.0, 5.0])

      normalized = Value.normalize_advantages(advantages)

      # Should return zeros (or near-zero with epsilon)
      values = Nx.to_flat_list(normalized)
      Enum.each(values, fn v ->
        assert_in_delta v, 0.0, 0.001
      end)
    end
  end
end
