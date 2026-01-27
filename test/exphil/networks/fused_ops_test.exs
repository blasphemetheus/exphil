defmodule ExPhil.Networks.FusedOpsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.FusedOps

  describe "dense_activation/4" do
    test "produces correct output with relu" do
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32)
      weight = Nx.broadcast(0.1, {4, 2})
      bias = Nx.tensor([0.5, -0.5], type: :f32)

      result = FusedOps.dense_activation(input, weight, bias, :relu)

      # Expected: relu(input @ weight + bias)
      expected =
        input
        |> Nx.dot(weight)
        |> Nx.add(bias)
        |> Nx.max(0)

      assert_all_close(result, expected)
    end

    test "produces correct output with silu" do
      input = Nx.tensor([[1.0, 2.0]], type: :f32)
      weight = Nx.broadcast(0.5, {2, 3})
      bias = Nx.broadcast(0.1, {3})

      result = FusedOps.dense_activation(input, weight, bias, :silu)

      # Expected: silu(input @ weight + bias)
      linear = Nx.add(Nx.dot(input, weight), bias)
      expected = Nx.multiply(linear, Nx.sigmoid(linear))

      assert_all_close(result, expected)
    end

    test "produces correct output with gelu" do
      input = Nx.tensor([[1.0, -1.0, 0.5]], type: :f32)
      weight = Nx.eye(3, type: :f32)
      bias = Nx.broadcast(0.0, {3})

      result = FusedOps.dense_activation(input, weight, bias, :gelu)

      # GELU should be asymmetric (positive values > abs(negative values) for same magnitude)
      result_list = Nx.to_flat_list(result)
      assert Enum.at(result_list, 0) > abs(Enum.at(result_list, 1))
    end

    test "handles 3D input (batch, seq, features)" do
      input = Nx.iota({2, 3, 4}, type: :f32) |> Nx.divide(10)
      weight = Nx.broadcast(0.1, {4, 8})
      bias = Nx.broadcast(0.0, {8})

      result = FusedOps.dense_activation(input, weight, bias, :relu)

      assert Nx.shape(result) == {2, 3, 8}
    end

    test "works with bf16 input" do
      input = Nx.tensor([[1.0, 2.0]], type: :bf16)
      weight = Nx.broadcast(0.5, {2, 3}) |> Nx.as_type(:bf16)
      bias = Nx.broadcast(0.1, {3}) |> Nx.as_type(:bf16)

      result = FusedOps.dense_activation(input, weight, bias, :relu)

      assert Nx.type(result) == {:bf, 16}
    end
  end

  describe "dense_activation_no_bias/3" do
    test "produces correct output" do
      input = Nx.tensor([[1.0, 2.0]], type: :f32)
      weight = Nx.broadcast(0.5, {2, 3})

      result = FusedOps.dense_activation_no_bias(input, weight, :relu)

      expected = input |> Nx.dot(weight) |> Nx.max(0)
      assert_all_close(result, expected)
    end
  end

  describe "layernorm_activation/5" do
    test "normalizes correctly with identity activation" do
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], type: :f32)
      gamma = Nx.broadcast(1.0, {5})
      beta = Nx.broadcast(0.0, {5})

      result = FusedOps.layernorm_activation(input, gamma, beta, :identity)

      # Check mean is ~0
      mean = Nx.mean(result) |> Nx.to_number()
      assert_in_delta mean, 0.0, 1.0e-5

      # Check variance is ~1
      var = Nx.variance(result) |> Nx.to_number()
      assert_in_delta var, 1.0, 1.0e-4
    end

    test "applies scale and shift" do
      input = Nx.tensor([[0.0, 1.0, 2.0]], type: :f32)
      gamma = Nx.tensor([2.0, 2.0, 2.0], type: :f32)
      beta = Nx.tensor([1.0, 1.0, 1.0], type: :f32)

      result = FusedOps.layernorm_activation(input, gamma, beta, :identity)

      # After scale=2 and shift=1, mean should be 1.0
      mean = Nx.mean(result) |> Nx.to_number()
      assert_in_delta mean, 1.0, 1.0e-4
    end

    test "applies activation after normalization" do
      input = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)
      gamma = Nx.broadcast(1.0, {3})
      beta = Nx.tensor([-10.0, 0.0, 10.0], type: :f32)  # Force some negative values

      result = FusedOps.layernorm_activation(input, gamma, beta, :relu)

      # All values should be >= 0 after ReLU
      min_val = Nx.reduce_min(result) |> Nx.to_number()
      assert min_val >= 0.0
    end

    test "preserves bf16 type" do
      input = Nx.tensor([[1.0, 2.0, 3.0]], type: :bf16)
      gamma = Nx.broadcast(1.0, {3}) |> Nx.as_type(:f32)
      beta = Nx.broadcast(0.0, {3}) |> Nx.as_type(:f32)

      result = FusedOps.layernorm_activation(input, gamma, beta, :relu)

      assert Nx.type(result) == {:bf, 16}
    end
  end

  describe "fused_ffn/6" do
    test "produces correct output shape" do
      input = Nx.iota({2, 4}, type: :f32) |> Nx.divide(10)
      w1 = Nx.broadcast(0.1, {4, 16})  # Expand to 4x
      b1 = Nx.broadcast(0.0, {16})
      w2 = Nx.broadcast(0.1, {16, 4})  # Project back
      b2 = Nx.broadcast(0.0, {4})

      result = FusedOps.fused_ffn(input, w1, b1, w2, b2, :gelu)

      assert Nx.shape(result) == {2, 4}
    end

    test "matches unfused computation" do
      input = Nx.tensor([[1.0, 2.0]], type: :f32)
      w1 = Nx.broadcast(0.5, {2, 4})
      b1 = Nx.broadcast(0.1, {4})
      w2 = Nx.broadcast(0.25, {4, 2})
      b2 = Nx.broadcast(-0.1, {2})

      result = FusedOps.fused_ffn(input, w1, b1, w2, b2, :relu)

      # Compute unfused
      hidden = input |> Nx.dot(w1) |> Nx.add(b1) |> Nx.max(0)
      expected = hidden |> Nx.dot(w2) |> Nx.add(b2)

      assert_all_close(result, expected)
    end
  end

  describe "gated_linear_unit/4" do
    test "computes gate * up correctly" do
      input = Nx.tensor([[1.0, 2.0]], type: :f32)
      w_gate = Nx.broadcast(0.5, {2, 3})
      w_up = Nx.broadcast(0.3, {2, 3})

      result = FusedOps.gated_linear_unit(input, w_gate, w_up, :silu)

      # Compute expected
      gate = input |> Nx.dot(w_gate)
      gate_activated = Nx.multiply(gate, Nx.sigmoid(gate))  # SiLU
      up = Nx.dot(input, w_up)
      expected = Nx.multiply(gate_activated, up)

      assert_all_close(result, expected)
    end

    test "swiglu is correct" do
      input = Nx.tensor([[1.0, 2.0]], type: :f32)
      w_gate = Nx.broadcast(0.5, {2, 3})
      w_up = Nx.broadcast(0.3, {2, 3})

      result = FusedOps.swiglu(input, w_gate, w_up)
      expected = FusedOps.gated_linear_unit(input, w_gate, w_up, :silu)

      assert_all_close(result, expected)
    end
  end

  describe "fused_softmax/2" do
    test "produces valid probability distribution" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)

      result = FusedOps.fused_softmax(logits)

      # Check probabilities sum to 1
      sum = Nx.sum(result, axes: [1]) |> Nx.squeeze() |> Nx.to_number()
      assert_in_delta sum, 1.0, 1.0e-5

      # Check all values are in [0, 1]
      min_val = Nx.reduce_min(result) |> Nx.to_number()
      max_val = Nx.reduce_max(result) |> Nx.to_number()
      assert min_val >= 0.0
      assert max_val <= 1.0
    end

    test "handles large values without overflow" do
      logits = Nx.tensor([[100.0, 200.0, 300.0]], type: :f32)

      result = FusedOps.fused_softmax(logits)

      # Should not have NaN or Inf
      has_nan = Nx.any(Nx.is_nan(result)) |> Nx.to_number()
      has_inf = Nx.any(Nx.is_infinity(result)) |> Nx.to_number()
      assert has_nan == 0
      assert has_inf == 0

      # Should still sum to 1
      sum = Nx.sum(result, axes: [1]) |> Nx.squeeze() |> Nx.to_number()
      assert_in_delta sum, 1.0, 1.0e-4
    end

    test "preserves bf16 type" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :bf16)

      result = FusedOps.fused_softmax(logits)

      assert Nx.type(result) == {:bf, 16}
    end
  end

  describe "fused_log_softmax/2" do
    test "produces correct log probabilities" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)

      result = FusedOps.fused_log_softmax(logits)

      # Compare with log(softmax(x))
      expected = logits |> FusedOps.fused_softmax() |> Nx.log()

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    test "is more stable than log(softmax)" do
      # Large logits that would cause issues with naive log(softmax)
      logits = Nx.tensor([[100.0, 101.0, 102.0]], type: :f32)

      result = FusedOps.fused_log_softmax(logits)

      # Should not have NaN
      has_nan = Nx.any(Nx.is_nan(result)) |> Nx.to_number()
      assert has_nan == 0

      # Log probs should be negative (since probs < 1)
      max_val = Nx.reduce_max(result) |> Nx.to_number()
      assert max_val <= 0.0
    end
  end

  describe "fused_ssm_discretize/3" do
    test "computes A_bar and B_bar correctly" do
      dt = Nx.tensor([[0.1, 0.2]], type: :f32)
      a = Nx.tensor([[-1.0, -2.0]], type: :f32)
      b = Nx.tensor([[1.0, 0.5]], type: :f32)

      {a_bar, b_bar} = FusedOps.fused_ssm_discretize(dt, a, b)

      # A_bar = exp(dt * A)
      expected_a_bar = Nx.exp(Nx.multiply(dt, a))
      assert_all_close(a_bar, expected_a_bar)

      # B_bar = dt * B
      expected_b_bar = Nx.multiply(dt, b)
      assert_all_close(b_bar, expected_b_bar)
    end
  end

  describe "apply_activation/2" do
    test "relu" do
      x = Nx.tensor([-1.0, 0.0, 1.0], type: :f32)
      result = FusedOps.apply_activation(x, :relu)
      expected = Nx.tensor([0.0, 0.0, 1.0], type: :f32)
      assert_all_close(result, expected)
    end

    test "silu" do
      x = Nx.tensor([0.0, 1.0, 2.0], type: :f32)
      result = FusedOps.apply_activation(x, :silu)
      expected = Nx.multiply(x, Nx.sigmoid(x))
      assert_all_close(result, expected)
    end

    test "sigmoid" do
      x = Nx.tensor([0.0], type: :f32)
      result = FusedOps.apply_activation(x, :sigmoid)
      expected = Nx.tensor([0.5], type: :f32)
      assert_all_close(result, expected)
    end

    test "identity returns input unchanged" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
      result = FusedOps.apply_activation(x, :identity)
      assert_all_close(result, x)
    end
  end

  describe "supported_activation?/1" do
    test "returns true for supported activations" do
      assert FusedOps.supported_activation?(:relu)
      assert FusedOps.supported_activation?(:silu)
      assert FusedOps.supported_activation?(:gelu)
      assert FusedOps.supported_activation?(:sigmoid)
    end

    test "returns false for unsupported activations" do
      refute FusedOps.supported_activation?(:leaky_relu)
      refute FusedOps.supported_activation?(:elu)
      refute FusedOps.supported_activation?(:custom)
    end
  end

  # Helper functions

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)

    diff = Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert diff < atol,
           "Tensors not close: max difference = #{diff}, tolerance = #{atol}"
  end
end
