defmodule ExPhil.Networks.PolicyTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy

  @embed_size 128
  @batch_size 4

  # ============================================================================
  # Model Building Tests
  # ============================================================================

  describe "build/1" do
    test "builds a valid Axon model" do
      model = Policy.build(embed_size: @embed_size)

      assert %Axon{} = model
    end

    test "requires embed_size option" do
      assert_raise KeyError, fn ->
        Policy.build([])
      end
    end

    test "accepts custom hidden sizes" do
      model = Policy.build(embed_size: @embed_size, hidden_sizes: [256, 128])

      assert %Axon{} = model
    end

    test "accepts custom axis and shoulder buckets" do
      model = Policy.build(
        embed_size: @embed_size,
        axis_buckets: 8,
        shoulder_buckets: 2
      )

      assert %Axon{} = model
    end

    test "model can be initialized" do
      model = Policy.build(embed_size: @embed_size)
      {init_fn, _predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())

      assert %Axon.ModelState{} = params
    end

    test "model forward pass produces correct output shapes" do
      model = Policy.build(embed_size: @embed_size, axis_buckets: 16, shoulder_buckets: 4)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())
      input = Nx.Random.uniform(Nx.Random.key(42), shape: {@batch_size, @embed_size}) |> elem(0)

      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      # Check output shapes
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end
  end

  describe "build_backbone/4" do
    test "builds MLP backbone" do
      input = Axon.input("state", shape: {nil, @embed_size})
      backbone = Policy.build_backbone(input, [256, 128], :relu, 0.1)

      assert %Axon{} = backbone
    end

    test "handles empty hidden sizes" do
      input = Axon.input("state", shape: {nil, @embed_size})
      backbone = Policy.build_backbone(input, [], :relu, 0.0)

      # Should just return the input
      assert %Axon{} = backbone
    end

    test "supports different activations" do
      input = Axon.input("state", shape: {nil, @embed_size})

      for activation <- [:relu, :tanh, :gelu, :selu] do
        backbone = Policy.build_backbone(input, [64], activation, 0.0)
        assert %Axon{} = backbone
      end
    end
  end

  describe "build_controller_head/3" do
    test "builds controller head with default buckets" do
      input = Axon.input("state", shape: {nil, 256})
      backbone = Axon.dense(input, 256, name: "backbone")

      head = Policy.build_controller_head(backbone, 16, 4)

      assert %Axon{} = head
    end

    test "builds controller head with custom buckets" do
      input = Axon.input("state", shape: {nil, 256})
      backbone = Axon.dense(input, 256, name: "backbone")

      head = Policy.build_controller_head(backbone, 8, 2)

      assert %Axon{} = head
    end
  end

  describe "build_autoregressive/1" do
    test "builds autoregressive model" do
      model = Policy.build_autoregressive(embed_size: @embed_size)

      assert %Axon{} = model
    end

    test "requires embed_size option" do
      assert_raise KeyError, fn ->
        Policy.build_autoregressive([])
      end
    end
  end

  # ============================================================================
  # Sampling Tests
  # ============================================================================

  describe "sample_buttons/2" do
    test "deterministic sampling uses threshold" do
      logits = Nx.tensor([[2.0, -2.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0]])

      result = Policy.sample_buttons(logits, true)

      # With sigmoid: positive logits -> > 0.5, negative -> < 0.5
      expected = Nx.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], type: :u8)
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "stochastic sampling returns valid binary tensor" do
      logits = Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      result = Policy.sample_buttons(logits, false)

      # Should be binary values
      flat = Nx.to_flat_list(result)
      assert Enum.all?(flat, &(&1 in [0, 1] or &1 in [true, false]))
    end

    test "handles batched input" do
      logits = Nx.tensor([
        [2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      ])

      result = Policy.sample_buttons(logits, true)

      assert Nx.shape(result) == {2, 8}
    end
  end

  describe "sample_categorical/3" do
    test "deterministic sampling uses argmax" do
      logits = Nx.tensor([[0.1, 0.5, 0.2, 0.9, 0.3]])

      result = Policy.sample_categorical(logits, 1.0, true)

      # Argmax should be index 3 (squeeze to get scalar)
      [value] = Nx.to_flat_list(result)
      assert value == 3
    end

    test "stochastic sampling returns valid indices" do
      logits = Nx.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
      num_classes = 5

      result = Policy.sample_categorical(logits, 1.0, false)

      # Should be a valid index
      [value] = Nx.to_flat_list(result)
      assert value >= 0
      assert value < num_classes
    end

    test "temperature affects sampling distribution" do
      # With very high temperature, distribution becomes more uniform
      # With very low temperature, becomes more peaked
      logits = Nx.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])

      # Low temperature should almost always pick index 0
      result_low = Policy.sample_categorical(logits, 0.01, true)
      [value] = Nx.to_flat_list(result_low)
      assert value == 0
    end

    test "handles batched input" do
      logits = Nx.tensor([
        [0.1, 0.9, 0.3],
        [0.9, 0.1, 0.3]
      ])

      result = Policy.sample_categorical(logits, 1.0, true)

      assert Nx.shape(result) == {2}
      # First batch should select index 1, second batch should select index 0
      [first, second] = Nx.to_flat_list(result)
      assert first == 1
      assert second == 0
    end
  end

  # ============================================================================
  # Loss Function Tests
  # ============================================================================

  describe "binary_cross_entropy/2" do
    test "returns zero for perfect predictions" do
      # Strong positive logits with target 1
      logits = Nx.tensor([[10.0, 10.0, 10.0, 10.0]])
      targets = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])

      loss = Policy.binary_cross_entropy(logits, targets)

      assert Nx.to_number(loss) < 0.001
    end

    test "returns high loss for wrong predictions" do
      # Strong positive logits with target 0
      logits = Nx.tensor([[10.0, 10.0, 10.0, 10.0]])
      targets = Nx.tensor([[0.0, 0.0, 0.0, 0.0]])

      loss = Policy.binary_cross_entropy(logits, targets)

      assert Nx.to_number(loss) > 5.0
    end

    test "is symmetric for balanced predictions" do
      logits = Nx.tensor([[0.0, 0.0, 0.0, 0.0]])
      targets_1 = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      targets_0 = Nx.tensor([[0.0, 0.0, 0.0, 0.0]])

      loss_1 = Nx.to_number(Policy.binary_cross_entropy(logits, targets_1))
      loss_0 = Nx.to_number(Policy.binary_cross_entropy(logits, targets_0))

      # Both should be log(2) ≈ 0.693
      assert_in_delta loss_1, 0.693, 0.01
      assert_in_delta loss_0, 0.693, 0.01
    end
  end

  describe "categorical_cross_entropy/2" do
    test "returns low loss for correct predictions" do
      # Strong logits for class 2
      logits = Nx.tensor([[-10.0, -10.0, 10.0, -10.0, -10.0]])
      targets = Nx.tensor([2])

      loss = Policy.categorical_cross_entropy(logits, targets)

      assert Nx.to_number(loss) < 0.001
    end

    test "returns high loss for wrong predictions" do
      # Strong logits for class 0, but target is class 2
      logits = Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]])
      targets = Nx.tensor([2])

      loss = Policy.categorical_cross_entropy(logits, targets)

      assert Nx.to_number(loss) > 10.0
    end

    test "handles uniform distribution" do
      logits = Nx.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
      targets = Nx.tensor([2])

      loss = Policy.categorical_cross_entropy(logits, targets)

      # Should be log(5) ≈ 1.609
      assert_in_delta Nx.to_number(loss), 1.609, 0.01
    end

    test "handles batched input" do
      logits = Nx.tensor([
        [10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0]
      ])
      targets = Nx.tensor([0, 1])

      loss = Policy.categorical_cross_entropy(logits, targets)

      # Both predictions are correct, loss should be low
      assert Nx.to_number(loss) < 0.001
    end
  end

  describe "imitation_loss/2" do
    test "computes combined loss for all components" do
      logits = %{
        buttons: Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        main_y: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        c_x: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        c_y: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        shoulder: Nx.tensor([[0.0, 0.0, 1.0]])
      }

      targets = %{
        buttons: Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([2]),
        main_y: Nx.tensor([2]),
        c_x: Nx.tensor([2]),
        c_y: Nx.tensor([2]),
        shoulder: Nx.tensor([2])
      }

      loss = Policy.imitation_loss(logits, targets)

      assert is_struct(loss, Nx.Tensor)
      assert Nx.to_number(loss) > 0
    end

    test "loss decreases with better predictions" do
      # Create "good" logits that match targets
      good_logits = %{
        buttons: Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]]),
        main_x: Nx.tensor([[-10.0, -10.0, 10.0, -10.0, -10.0]]),
        main_y: Nx.tensor([[-10.0, -10.0, 10.0, -10.0, -10.0]]),
        c_x: Nx.tensor([[-10.0, -10.0, 10.0, -10.0, -10.0]]),
        c_y: Nx.tensor([[-10.0, -10.0, 10.0, -10.0, -10.0]]),
        shoulder: Nx.tensor([[-10.0, -10.0, 10.0]])
      }

      # Create "bad" logits that don't match targets
      bad_logits = %{
        buttons: Nx.tensor([[-10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]),
        main_x: Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]]),
        main_y: Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]]),
        c_x: Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]]),
        c_y: Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]]),
        shoulder: Nx.tensor([[10.0, -10.0, -10.0]])
      }

      targets = %{
        buttons: Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([2]),
        main_y: Nx.tensor([2]),
        c_x: Nx.tensor([2]),
        c_y: Nx.tensor([2]),
        shoulder: Nx.tensor([2])
      }

      good_loss = Nx.to_number(Policy.imitation_loss(good_logits, targets))
      bad_loss = Nx.to_number(Policy.imitation_loss(bad_logits, targets))

      assert good_loss < bad_loss
    end
  end

  # ============================================================================
  # Utility Function Tests
  # ============================================================================

  describe "output_sizes/1" do
    test "returns correct sizes with default buckets" do
      sizes = Policy.output_sizes()

      assert sizes.buttons == 8
      assert sizes.main_x == 17
      assert sizes.main_y == 17
      assert sizes.c_x == 17
      assert sizes.c_y == 17
      assert sizes.shoulder == 5
    end

    test "returns correct sizes with custom buckets" do
      sizes = Policy.output_sizes(axis_buckets: 8, shoulder_buckets: 2)

      assert sizes.buttons == 8
      assert sizes.main_x == 9
      assert sizes.main_y == 9
      assert sizes.c_x == 9
      assert sizes.c_y == 9
      assert sizes.shoulder == 3
    end
  end

  describe "total_action_dims/1" do
    test "returns correct total with default buckets" do
      total = Policy.total_action_dims()

      # 8 buttons + 4×17 sticks + 5 shoulder = 8 + 68 + 5 = 81
      assert total == 81
    end

    test "returns correct total with custom buckets" do
      total = Policy.total_action_dims(axis_buckets: 8, shoulder_buckets: 2)

      # 8 buttons + 4×9 sticks + 3 shoulder = 8 + 36 + 3 = 47
      assert total == 47
    end
  end

  describe "to_controller_state/2" do
    test "converts samples to ControllerState struct" do
      samples = %{
        buttons: Nx.tensor([1, 0, 0, 0, 0, 0, 0, 0]),  # A pressed
        main_x: Nx.tensor(8),   # Center (8 out of 16)
        main_y: Nx.tensor(16),  # Up (16 out of 16)
        c_x: Nx.tensor(0),      # Left (0 out of 16)
        c_y: Nx.tensor(8),      # Center
        shoulder: Nx.tensor(0)  # Not pressed
      }

      controller = Policy.to_controller_state(samples, axis_buckets: 16, shoulder_buckets: 4)

      assert %ExPhil.Bridge.ControllerState{} = controller
      assert controller.button_a == true
      assert controller.button_b == false
    end

    test "handles batched samples by taking first element" do
      samples = %{
        buttons: Nx.tensor([[1, 0, 1, 0, 0, 0, 0, 0]]),
        main_x: Nx.tensor([8]),
        main_y: Nx.tensor([8]),
        c_x: Nx.tensor([8]),
        c_y: Nx.tensor([8]),
        shoulder: Nx.tensor([0])
      }

      controller = Policy.to_controller_state(samples)

      assert %ExPhil.Bridge.ControllerState{} = controller
      assert controller.button_a == true
      assert controller.button_x == true
    end
  end

  # ============================================================================
  # Integration Tests
  # ============================================================================

  describe "full pipeline" do
    test "can build, initialize, forward pass, and compute loss" do
      # Build model
      model = Policy.build(embed_size: @embed_size)
      {init_fn, predict_fn} = Axon.build(model)

      # Initialize
      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())

      # Forward pass
      input = Nx.Random.uniform(Nx.Random.key(42), shape: {@batch_size, @embed_size}) |> elem(0)
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      # Create mock targets
      key = Nx.Random.key(44)
      {button_vals, key} = Nx.Random.uniform(key, shape: {@batch_size, 8})
      {main_x_vals, key} = Nx.Random.randint(key, 0, 17, shape: {@batch_size})
      {main_y_vals, key} = Nx.Random.randint(key, 0, 17, shape: {@batch_size})
      {c_x_vals, key} = Nx.Random.randint(key, 0, 17, shape: {@batch_size})
      {c_y_vals, key} = Nx.Random.randint(key, 0, 17, shape: {@batch_size})
      {shoulder_vals, _key} = Nx.Random.randint(key, 0, 5, shape: {@batch_size})

      targets = %{
        buttons: Nx.round(button_vals),
        main_x: main_x_vals,
        main_y: main_y_vals,
        c_x: c_x_vals,
        c_y: c_y_vals,
        shoulder: shoulder_vals
      }

      # Compute loss
      loss = Policy.imitation_loss(logits, targets)

      assert is_struct(loss, Nx.Tensor)
      assert Nx.to_number(loss) > 0
    end
  end
end
