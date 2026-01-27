defmodule ExPhil.Training.MixedPrecisionTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.MixedPrecision

  describe "init/2" do
    test "creates mixed precision state with FP32 master params" do
      params = %{
        layer1: %{kernel: Nx.iota({4, 3}, type: :f32), bias: Nx.iota({3}, type: :f32)}
      }

      mp_state = MixedPrecision.init(params, precision: :bf16)

      assert mp_state.precision == :bf16
      assert mp_state.loss_scale == 1.0

      # Master params should be FP32
      master = MixedPrecision.get_master_params(mp_state)
      assert Nx.type(master.layer1.kernel) == {:f, 32}
      assert Nx.type(master.layer1.bias) == {:f, 32}
    end

    test "converts BF16 params to FP32 master params" do
      params = %{
        kernel: Nx.iota({4, 3}, type: :bf16)
      }

      mp_state = MixedPrecision.init(params, precision: :bf16)

      master = MixedPrecision.get_master_params(mp_state)
      assert Nx.type(master.kernel) == {:f, 32}
    end
  end

  describe "get_compute_params/1" do
    test "casts master params to compute precision" do
      params = %{
        kernel: Nx.iota({4, 3}, type: :f32)
      }

      mp_state = MixedPrecision.init(params, precision: :bf16)
      compute_params = MixedPrecision.get_compute_params(mp_state)

      assert Nx.type(compute_params.kernel) == {:bf, 16}
    end

    test "handles nested params" do
      params = %{
        layer1: %{kernel: Nx.iota({4, 3}, type: :f32), bias: Nx.iota({3}, type: :f32)},
        layer2: %{kernel: Nx.iota({3, 2}, type: :f32), bias: Nx.iota({2}, type: :f32)}
      }

      mp_state = MixedPrecision.init(params, precision: :bf16)
      compute_params = MixedPrecision.get_compute_params(mp_state)

      assert Nx.type(compute_params.layer1.kernel) == {:bf, 16}
      assert Nx.type(compute_params.layer2.bias) == {:bf, 16}
    end
  end

  describe "stable_softmax/2" do
    test "computes correct softmax" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)
      probs = MixedPrecision.stable_softmax(logits)

      # Check probabilities sum to 1
      sum = Nx.sum(probs, axes: [1]) |> Nx.squeeze() |> Nx.to_number()
      assert_in_delta sum, 1.0, 1.0e-5

      # Check relative ordering
      probs_list = Nx.to_flat_list(probs)
      assert Enum.at(probs_list, 2) > Enum.at(probs_list, 1)
      assert Enum.at(probs_list, 1) > Enum.at(probs_list, 0)
    end

    test "preserves input precision" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :bf16)
      probs = MixedPrecision.stable_softmax(logits)

      assert Nx.type(probs) == {:bf, 16}
    end

    test "handles large values without overflow" do
      # These values would overflow in naive exp() with BF16
      logits = Nx.tensor([[100.0, 200.0, 300.0]], type: :bf16)
      probs = MixedPrecision.stable_softmax(logits)

      # Should not be NaN or Inf
      has_nan = Nx.any(Nx.is_nan(probs)) |> Nx.to_number()
      has_inf = Nx.any(Nx.is_infinity(probs)) |> Nx.to_number()
      assert has_nan == 0, "Should not have NaN values"
      assert has_inf == 0, "Should not have Inf values"

      # Probabilities should sum to 1
      sum = Nx.sum(probs, axes: [1]) |> Nx.squeeze() |> Nx.to_number()
      assert_in_delta sum, 1.0, 1.0e-2
    end
  end

  describe "stable_layer_norm/4" do
    test "normalizes to zero mean and unit variance" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], type: :f32)
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0], type: :f32)
      beta = Nx.tensor([0.0, 0.0, 0.0, 0.0, 0.0], type: :f32)

      result = MixedPrecision.stable_layer_norm(x, gamma, beta)

      # Check mean is approximately 0
      mean = Nx.mean(result) |> Nx.to_number()
      assert_in_delta mean, 0.0, 1.0e-5

      # Check variance is approximately 1
      var = Nx.variance(result) |> Nx.to_number()
      assert_in_delta var, 1.0, 1.0e-4
    end

    test "preserves input precision" do
      x = Nx.tensor([[1.0, 2.0, 3.0]], type: :bf16)
      gamma = Nx.tensor([1.0, 1.0, 1.0], type: :f32)
      beta = Nx.tensor([0.0, 0.0, 0.0], type: :f32)

      result = MixedPrecision.stable_layer_norm(x, gamma, beta)
      assert Nx.type(result) == {:bf, 16}
    end
  end

  describe "stable_cross_entropy/3" do
    test "computes correct cross entropy with one-hot targets" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)
      targets = Nx.tensor([[0.0, 0.0, 1.0]], type: :f32)  # Class 2

      loss = MixedPrecision.stable_cross_entropy(logits, targets)
      loss_val = Nx.to_number(loss)

      # Expected: -log(softmax(3)) ≈ -log(0.665) ≈ 0.407
      assert_in_delta loss_val, 0.407, 0.01
    end

    test "handles BF16 inputs" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :bf16)
      targets = Nx.tensor([[0.0, 0.0, 1.0]], type: :bf16)

      loss = MixedPrecision.stable_cross_entropy(logits, targets)

      # Should not be NaN
      is_nan = Nx.is_nan(loss) |> Nx.to_number()
      assert is_nan == 0, "Loss should not be NaN"
    end
  end

  describe "set_master_params/2" do
    test "updates master params" do
      initial_params = %{kernel: Nx.broadcast(1.0, {3, 3})}
      mp_state = MixedPrecision.init(initial_params, precision: :bf16)

      new_params = %{kernel: Nx.broadcast(2.0, {3, 3})}
      updated_state = MixedPrecision.set_master_params(mp_state, new_params)

      master = MixedPrecision.get_master_params(updated_state)
      assert Nx.to_number(Nx.mean(master.kernel)) == 2.0
    end
  end
end
