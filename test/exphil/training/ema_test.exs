defmodule ExPhil.Training.EMATest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.EMA

  describe "new/2" do
    test "creates EMA with default decay" do
      params = %{layer1: %{weights: Nx.tensor([1.0, 2.0, 3.0])}}
      ema = EMA.new(params)

      assert ema.decay == 0.999
      assert ema.step == 0
      assert EMA.get_params(ema).layer1.weights |> Nx.to_list() == [1.0, 2.0, 3.0]
    end

    test "creates EMA with custom decay" do
      params = %{w: Nx.tensor([1.0])}
      ema = EMA.new(params, decay: 0.99)

      assert ema.decay == 0.99
    end

    test "deep copies params to avoid aliasing" do
      original = Nx.tensor([1.0, 2.0])
      params = %{w: original}
      ema = EMA.new(params)

      # Modify original should not affect EMA
      # (In Nx, tensors are immutable, but this tests the deep copy intent)
      assert EMA.get_params(ema).w |> Nx.to_list() == [1.0, 2.0]
    end
  end

  describe "update/2" do
    test "updates EMA weights with formula: ema = decay * ema + (1-decay) * new" do
      # Start with [10.0]
      params = %{w: Nx.tensor([10.0])}
      ema = EMA.new(params, decay: 0.9)

      # Update with [0.0]
      # Expected: 0.9 * 10.0 + 0.1 * 0.0 = 9.0
      new_params = %{w: Nx.tensor([0.0])}
      ema = EMA.update(ema, new_params)

      result = EMA.get_params(ema).w |> Nx.squeeze() |> Nx.to_number()
      assert_in_delta result, 9.0, 0.001
      assert ema.step == 1
    end

    test "multiple updates converge towards new values" do
      params = %{w: Nx.tensor([100.0])}
      ema = EMA.new(params, decay: 0.9)

      # Repeatedly update with 0.0
      new_params = %{w: Nx.tensor([0.0])}
      ema = Enum.reduce(1..50, ema, fn _, acc -> EMA.update(acc, new_params) end)

      # After many updates, should be close to 0
      result = EMA.get_params(ema).w |> Nx.squeeze() |> Nx.to_number()
      assert result < 1.0
      assert ema.step == 50
    end

    test "handles nested params structure" do
      params = %{
        encoder: %{
          layer1: Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
          layer2: Nx.tensor([5.0, 6.0])
        },
        decoder: %{
          output: Nx.tensor([7.0])
        }
      }
      ema = EMA.new(params, decay: 0.5)

      new_params = %{
        encoder: %{
          layer1: Nx.tensor([[0.0, 0.0], [0.0, 0.0]]),
          layer2: Nx.tensor([0.0, 0.0])
        },
        decoder: %{
          output: Nx.tensor([0.0])
        }
      }

      ema = EMA.update(ema, new_params)

      # With decay=0.5: ema = 0.5 * old + 0.5 * new
      result = EMA.get_params(ema)
      assert result.encoder.layer1 |> Nx.to_flat_list() == [0.5, 1.0, 1.5, 2.0]
      assert result.encoder.layer2 |> Nx.to_list() == [2.5, 3.0]
      assert result.decoder.output |> Nx.squeeze() |> Nx.to_number() == 3.5
    end
  end

  describe "get_params/1" do
    test "returns current EMA parameters" do
      params = %{a: Nx.tensor([1.0]), b: Nx.tensor([2.0])}
      ema = EMA.new(params)

      result = EMA.get_params(ema)
      assert result.a |> Nx.squeeze() |> Nx.to_number() == 1.0
      assert result.b |> Nx.squeeze() |> Nx.to_number() == 2.0
    end
  end

  describe "get_step/1" do
    test "returns current step count" do
      params = %{w: Nx.tensor([1.0])}
      ema = EMA.new(params)

      assert EMA.get_step(ema) == 0

      ema = EMA.update(ema, params)
      assert EMA.get_step(ema) == 1

      ema = EMA.update(ema, params)
      assert EMA.get_step(ema) == 2
    end
  end

  describe "serialize/1 and deserialize/1" do
    test "round-trips EMA state" do
      params = %{
        layer: %{
          weights: Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
          bias: Nx.tensor([5.0, 6.0])
        }
      }
      ema = EMA.new(params, decay: 0.95)
      ema = EMA.update(ema, params)

      binary = EMA.serialize(ema)
      restored = EMA.deserialize(binary)

      assert restored.decay == 0.95
      assert restored.step == 1

      restored_params = EMA.get_params(restored)
      assert restored_params.layer.weights |> Nx.to_flat_list() == [1.0, 2.0, 3.0, 4.0]
      assert restored_params.layer.bias |> Nx.to_list() == [5.0, 6.0]
    end
  end

  describe "to_model_params/1" do
    test "returns params suitable for model use" do
      params = %{w: Nx.tensor([1.0, 2.0])}
      ema = EMA.new(params)

      model_params = EMA.to_model_params(ema)

      # Should be on BinaryBackend for serialization
      assert model_params.w |> Nx.to_list() == [1.0, 2.0]
    end
  end

  describe "update_with_bias_correction/2" do
    test "applies bias correction for early steps" do
      params = %{w: Nx.tensor([0.0])}
      ema = EMA.new(params, decay: 0.99)

      # Update with 100.0
      # Without correction: 0.99 * 0 + 0.01 * 100 = 1.0
      # With correction at step 1: 1.0 / (1 - 0.99^1) = 1.0 / 0.01 = 100.0
      new_params = %{w: Nx.tensor([100.0])}
      ema = EMA.update_with_bias_correction(ema, new_params)

      result = EMA.get_params(ema).w |> Nx.squeeze() |> Nx.to_number()
      # Bias correction makes early estimates closer to true mean
      assert_in_delta result, 100.0, 0.1
    end
  end
end
