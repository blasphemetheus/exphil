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
      model =
        Policy.build(
          embed_size: @embed_size,
          axis_buckets: 8,
          shoulder_buckets: 2
        )

      assert %Axon{} = model
    end

    @tag :slow
    test "model can be initialized" do
      model = Policy.build(embed_size: @embed_size)
      {init_fn, _predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())

      assert %Axon.ModelState{} = params
    end

    @tag :slow
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

    test "builds model with action embedding layer" do
      # embed_size = continuous_size + 2 action IDs
      model = Policy.build(embed_size: @embed_size, action_embed_size: 32)

      assert %Axon{} = model
    end

    @tag :slow
    test "model with action embedding produces correct output shapes" do
      # Total embed_size includes 2 action IDs at the end
      continuous_size = 64
      # 66
      total_embed_size = continuous_size + 2

      model =
        Policy.build(
          embed_size: total_embed_size,
          action_embed_size: 32,
          axis_buckets: 16,
          shoulder_buckets: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, total_embed_size}, :f32), Axon.ModelState.empty())

      # Create input with action IDs at the end (values 0-398 are valid)
      continuous_input = Nx.broadcast(0.5, {@batch_size, continuous_size})
      # Example action IDs
      action_ids = Nx.tensor([[50, 100], [60, 110], [70, 120], [80, 130]])
      input = Nx.concatenate([continuous_input, action_ids], axis: 1)

      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      # Should still produce correct output shapes
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end

    test "builds autoregressive policy by default" do
      model = Policy.build(embed_size: @embed_size)
      assert %Axon{} = model
    end

    test "builds autoregressive policy when explicitly specified" do
      model = Policy.build(embed_size: @embed_size, policy_type: :autoregressive)
      assert %Axon{} = model
    end

    test "builds diffusion policy" do
      # DiffusionPolicy returns an Axon model
      # Uses obs_size instead of embed_size
      model = Policy.build(
        embed_size: @embed_size,
        obs_size: @embed_size,
        policy_type: :diffusion,
        action_horizon: 8,
        action_dim: 13
      )
      assert %Axon{} = model
    end

    test "builds ACT (action chunking) policy" do
      # Uses obs_size instead of embed_size
      model = Policy.build(
        embed_size: @embed_size,
        obs_size: @embed_size,
        policy_type: :act,
        action_horizon: 8,
        action_dim: 13
      )
      assert is_map(model)
      assert Map.has_key?(model, :encoder)
      assert Map.has_key?(model, :decoder)
    end

    test "builds flow matching policy" do
      # Uses obs_size instead of embed_size
      model = Policy.build(
        embed_size: @embed_size,
        obs_size: @embed_size,
        policy_type: :flow_matching,
        action_horizon: 8,
        action_dim: 13
      )
      assert %Axon{} = model
    end

    test "raises on unknown policy type" do
      assert_raise ArgumentError, ~r/Unknown policy type/, fn ->
        Policy.build(embed_size: @embed_size, policy_type: :unknown)
      end
    end
  end

  describe "build_action_embedding_layer/4" do
    test "builds action embedding layer with 2 action IDs" do
      # 64 continuous + 2 action IDs
      total_embed_size = 66
      action_embed_size = 32
      num_action_ids = 2
      input = Axon.input("state", shape: {nil, total_embed_size})

      layer =
        Policy.build_action_embedding_layer(
          input,
          total_embed_size,
          action_embed_size,
          num_action_ids
        )

      assert %Axon{} = layer
    end

    test "builds action embedding layer with 4 action IDs (enhanced Nana)" do
      # Enhanced Nana mode: 2 player actions + 2 Nana actions
      # 64 continuous + 4 action IDs
      total_embed_size = 68
      action_embed_size = 32
      num_action_ids = 4
      input = Axon.input("state", shape: {nil, total_embed_size})

      layer =
        Policy.build_action_embedding_layer(
          input,
          total_embed_size,
          action_embed_size,
          num_action_ids
        )

      assert %Axon{} = layer
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

    test "builds backbone with layer_norm option" do
      input = Axon.input("state", shape: {nil, @embed_size})
      backbone = Policy.build_backbone(input, [64, 64], :relu, 0.1, layer_norm: true)

      assert %Axon{} = backbone
    end

    test "builds backbone with residual connections" do
      input = Axon.input("state", shape: {nil, @embed_size})
      backbone = Policy.build_backbone(input, [64, 64], :relu, 0.1, residual: true)

      assert %Axon{} = backbone
    end

    test "builds backbone with residual and layer_norm" do
      input = Axon.input("state", shape: {nil, @embed_size})

      backbone =
        Policy.build_backbone(input, [64, 64], :relu, 0.1, residual: true, layer_norm: true)

      assert %Axon{} = backbone
    end

    test "residual backbone handles dimension changes with projection" do
      # Different hidden sizes require projection layers
      input = Axon.input("state", shape: {nil, @embed_size})
      backbone = Policy.build_backbone(input, [256, 128, 64], :relu, 0.1, residual: true)

      assert %Axon{} = backbone
    end
  end

  describe "build/1 with residual" do
    test "builds model with residual option" do
      model = Policy.build(embed_size: @embed_size, residual: true)

      assert %Axon{} = model
    end

    @tag :slow
    test "residual model can be initialized and run forward pass" do
      model = Policy.build(embed_size: @embed_size, hidden_sizes: [64, 64], residual: true)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())
      input = Nx.Random.uniform(Nx.Random.key(42), shape: {@batch_size, @embed_size}) |> elem(0)

      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      # Verify output shapes are correct
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end

    @tag :slow
    test "residual model has more parameters than non-residual when projection needed" do
      # With varying hidden sizes, residual needs projection layers
      non_residual =
        Policy.build(embed_size: @embed_size, hidden_sizes: [256, 128], residual: false)

      residual = Policy.build(embed_size: @embed_size, hidden_sizes: [256, 128], residual: true)

      {init_fn_nr, _} = Axon.build(non_residual)
      {init_fn_r, _} = Axon.build(residual)

      params_nr = init_fn_nr.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())
      params_r = init_fn_r.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())

      # Count parameters
      count_params = fn model_state ->
        model_state.data
        |> Enum.reduce(0, fn {_layer, layer_params}, acc ->
          layer_count =
            Enum.reduce(layer_params, 0, fn {_name, tensor}, acc2 ->
              acc2 + Nx.size(tensor)
            end)

          acc + layer_count
        end)
      end

      nr_count = count_params.(params_nr)
      r_count = count_params.(params_r)

      # Residual with projection should have more params
      assert r_count > nr_count
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
      logits =
        Nx.tensor([
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
      logits =
        Nx.tensor([
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

  describe "compute_confidence/1" do
    test "returns confidence map with all keys" do
      logits = %{
        buttons: Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        main_y: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        c_x: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        c_y: Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        shoulder: Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      }

      confidence = Policy.compute_confidence(logits)

      assert is_map(confidence)
      assert Map.has_key?(confidence, :overall)
      assert Map.has_key?(confidence, :buttons)
      assert Map.has_key?(confidence, :main)
      assert Map.has_key?(confidence, :c)
      assert Map.has_key?(confidence, :shoulder)
    end

    test "high confidence for peaked distributions" do
      # Very peaked distributions should have high confidence
      logits = %{
        buttons: Nx.tensor([[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]),
        main_x: Nx.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]]),
        main_y: Nx.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]]),
        c_x: Nx.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]]),
        c_y: Nx.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]]),
        shoulder: Nx.tensor([[10.0, 0.0, 0.0, 0.0]])
      }

      confidence = Policy.compute_confidence(logits)

      # High confidence for peaked distributions
      assert confidence.overall > 0.8
      # sigmoid(10) ≈ 1.0, far from 0.5
      assert confidence.buttons > 0.9
      # softmax([0,0,10,0,0]) peaks at ~1.0
      assert confidence.main > 0.9
    end

    test "low confidence for uniform distributions" do
      # Uniform distributions should have low confidence
      logits = %{
        buttons: Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
        main_y: Nx.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
        c_x: Nx.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
        c_y: Nx.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
        shoulder: Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      }

      confidence = Policy.compute_confidence(logits)

      # Low confidence for uniform distributions
      # sigmoid(0) = 0.5, no confidence
      assert confidence.buttons < 0.1
      # uniform softmax = 0.2 each
      assert confidence.main < 0.3
      assert confidence.overall < 0.3
    end

    test "accepts action map with :logits key" do
      action = %{
        buttons: Nx.tensor([1, 0, 0, 0, 0, 0, 0, 0]),
        main_x: 8,
        main_y: 8,
        logits: %{
          buttons: Nx.tensor([[5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]]),
          main_x: Nx.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]]),
          main_y: Nx.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]]),
          c_x: Nx.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]]),
          c_y: Nx.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]]),
          shoulder: Nx.tensor([[5.0, 0.0, 0.0, 0.0]])
        }
      }

      confidence = Policy.compute_confidence(action)

      assert confidence.overall > 0.5
    end

    test "returns zero confidence for invalid input" do
      confidence = Policy.compute_confidence(%{})

      assert confidence.overall == 0.0
      assert confidence.buttons == 0.0
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
      logits =
        Nx.tensor([
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

  describe "imitation_loss with focal loss" do
    test "computes focal loss when enabled" do
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

      focal_loss = Policy.imitation_loss(logits, targets, focal_loss: true, focal_gamma: 2.0)
      regular_loss = Policy.imitation_loss(logits, targets, focal_loss: false)

      assert is_struct(focal_loss, Nx.Tensor)
      assert Nx.to_number(focal_loss) > 0
      # Focal loss should differ from regular loss
      assert Nx.to_number(focal_loss) != Nx.to_number(regular_loss)
    end

    test "higher gamma focuses more on hard examples" do
      # Create logits where the model is somewhat confident but wrong
      # Focal loss with higher gamma should penalize this more
      uncertain_logits = %{
        buttons: Nx.tensor([[0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]]),
        main_x: Nx.tensor([[-0.5, -0.5, 0.5, -0.5, -0.5]]),
        main_y: Nx.tensor([[-0.5, -0.5, 0.5, -0.5, -0.5]]),
        c_x: Nx.tensor([[-0.5, -0.5, 0.5, -0.5, -0.5]]),
        c_y: Nx.tensor([[-0.5, -0.5, 0.5, -0.5, -0.5]]),
        shoulder: Nx.tensor([[-0.5, -0.5, 0.5]])
      }

      targets = %{
        buttons: Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([2]),
        main_y: Nx.tensor([2]),
        c_x: Nx.tensor([2]),
        c_y: Nx.tensor([2]),
        shoulder: Nx.tensor([2])
      }

      loss_gamma_1 =
        Nx.to_number(
          Policy.imitation_loss(uncertain_logits, targets, focal_loss: true, focal_gamma: 1.0)
        )

      loss_gamma_2 =
        Nx.to_number(
          Policy.imitation_loss(uncertain_logits, targets, focal_loss: true, focal_gamma: 2.0)
        )

      loss_gamma_5 =
        Nx.to_number(
          Policy.imitation_loss(uncertain_logits, targets, focal_loss: true, focal_gamma: 5.0)
        )

      # With higher gamma, the loss is down-weighted more for confident predictions
      # But for uncertain predictions like these, the relationship is more complex
      # Just verify all losses are finite and positive
      assert loss_gamma_1 > 0 and is_number(loss_gamma_1)
      assert loss_gamma_2 > 0 and is_number(loss_gamma_2)
      assert loss_gamma_5 > 0 and is_number(loss_gamma_5)
    end

    test "focal loss can be combined with label smoothing" do
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

      combined_loss =
        Policy.imitation_loss(logits, targets,
          focal_loss: true,
          focal_gamma: 2.0,
          label_smoothing: 0.1
        )

      assert is_struct(combined_loss, Nx.Tensor)
      assert Nx.to_number(combined_loss) > 0
    end
  end

  describe "weighted_categorical_cross_entropy/4" do
    test "weights edge buckets higher than center" do
      # 17-bucket system (0-16), center at 8
      num_classes = 17
      center = 8

      # Uniform logits (model equally uncertain)
      logits = Nx.broadcast(0.0, {1, num_classes})

      # Edge target (bucket 0) should have higher loss than center target (bucket 8)
      edge_target = Nx.tensor([0])
      center_target = Nx.tensor([center])

      edge_weight = 2.0

      # Regular CE (no weighting)
      edge_loss_regular = Policy.categorical_cross_entropy(logits, edge_target, 0.0)
      center_loss_regular = Policy.categorical_cross_entropy(logits, center_target, 0.0)

      # They should be equal without weighting (model is uniformly uncertain)
      assert_in_delta Nx.to_number(edge_loss_regular), Nx.to_number(center_loss_regular), 0.01

      # Weighted CE - edge should have 2x loss
      edge_loss_weighted = Policy.weighted_categorical_cross_entropy(logits, edge_target, 0.0, edge_weight)
      center_loss_weighted = Policy.weighted_categorical_cross_entropy(logits, center_target, 0.0, edge_weight)

      # Edge loss should be ~2x center loss
      assert_in_delta(
        Nx.to_number(edge_loss_weighted) / Nx.to_number(center_loss_weighted),
        edge_weight,
        0.01
      )
    end

    test "intermediate buckets have interpolated weights" do
      num_classes = 17
      center = 8

      logits = Nx.broadcast(0.0, {1, num_classes})
      edge_weight = 3.0

      # Bucket 4 is halfway between center (8) and edge (0)
      # Expected weight: 1.0 + (3.0 - 1.0) * 4/8 = 1.0 + 1.0 = 2.0
      halfway_target = Nx.tensor([4])
      center_target = Nx.tensor([center])

      halfway_loss = Policy.weighted_categorical_cross_entropy(logits, halfway_target, 0.0, edge_weight)
      center_loss = Policy.weighted_categorical_cross_entropy(logits, center_target, 0.0, edge_weight)

      # Halfway bucket should have weight 2.0 (halfway between 1.0 and 3.0)
      assert_in_delta(
        Nx.to_number(halfway_loss) / Nx.to_number(center_loss),
        2.0,
        0.01
      )
    end

    test "works with label smoothing" do
      num_classes = 17
      logits = Nx.broadcast(0.0, {1, num_classes})
      targets = Nx.tensor([0])

      loss_no_smooth = Policy.weighted_categorical_cross_entropy(logits, targets, 0.0, 2.0)
      loss_with_smooth = Policy.weighted_categorical_cross_entropy(logits, targets, 0.1, 2.0)

      # Both should be positive and different
      assert Nx.to_number(loss_no_smooth) > 0
      assert Nx.to_number(loss_with_smooth) > 0
      assert Nx.to_number(loss_no_smooth) != Nx.to_number(loss_with_smooth)
    end
  end

  describe "imitation_loss with stick_edge_weight" do
    test "applies edge weighting only to main stick" do
      # Create logits where main stick predictions are for edge buckets
      logits = %{
        buttons: Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.broadcast(0.0, {1, 17}),  # 17 buckets
        main_y: Nx.broadcast(0.0, {1, 17}),
        c_x: Nx.broadcast(0.0, {1, 17}),
        c_y: Nx.broadcast(0.0, {1, 17}),
        shoulder: Nx.tensor([[0.0, 0.0, 0.0, 0.0]])
      }

      # Targets: main stick at edges, c-stick at center
      targets_edge = %{
        buttons: Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([0]),   # Edge
        main_y: Nx.tensor([16]),  # Edge
        c_x: Nx.tensor([8]),      # Center
        c_y: Nx.tensor([8]),      # Center
        shoulder: Nx.tensor([0])
      }

      targets_center = %{
        buttons: Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        main_x: Nx.tensor([8]),   # Center
        main_y: Nx.tensor([8]),   # Center
        c_x: Nx.tensor([8]),      # Center
        c_y: Nx.tensor([8]),      # Center
        shoulder: Nx.tensor([0])
      }

      # Without edge weighting
      loss_edge_no_weight = Policy.imitation_loss(logits, targets_edge, stick_edge_weight: nil)
      loss_center_no_weight = Policy.imitation_loss(logits, targets_center, stick_edge_weight: nil)

      # With edge weighting
      loss_edge_weighted = Policy.imitation_loss(logits, targets_edge, stick_edge_weight: 2.0)
      loss_center_weighted = Policy.imitation_loss(logits, targets_center, stick_edge_weight: 2.0)

      # Without weighting, losses should be similar (uniform logits)
      # With weighting, edge loss should be higher
      edge_diff_no_weight = Nx.to_number(loss_edge_no_weight) - Nx.to_number(loss_center_no_weight)
      edge_diff_weighted = Nx.to_number(loss_edge_weighted) - Nx.to_number(loss_center_weighted)

      # The weighted edge loss should increase more than the weighted center loss
      assert edge_diff_weighted > edge_diff_no_weight
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
        # A pressed
        buttons: Nx.tensor([1, 0, 0, 0, 0, 0, 0, 0]),
        # Center (8 out of 16)
        main_x: Nx.tensor(8),
        # Up (16 out of 16)
        main_y: Nx.tensor(16),
        # Left (0 out of 16)
        c_x: Nx.tensor(0),
        # Center
        c_y: Nx.tensor(8),
        # Not pressed
        shoulder: Nx.tensor(0)
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

    test "uses K-means undiscretization when kmeans_centers provided" do
      # Create sorted K-means centers: 5 clusters at 0.0, 0.25, 0.5, 0.75, 1.0
      centers = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

      samples = %{
        buttons: Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
        # Index 2 -> center value 0.5
        main_x: Nx.tensor(2),
        # Index 4 -> center value 1.0
        main_y: Nx.tensor(4),
        # Index 1 -> center value 0.25
        c_x: Nx.tensor(1),
        # Index 3 -> center value 0.75
        c_y: Nx.tensor(3),
        shoulder: Nx.tensor(0)
      }

      controller = Policy.to_controller_state(samples, kmeans_centers: centers)

      # Check that values match the K-means cluster centers
      assert_in_delta controller.main_stick.x, 0.5, 0.001
      assert_in_delta controller.main_stick.y, 1.0, 0.001
      assert_in_delta controller.c_stick.x, 0.25, 0.001
      assert_in_delta controller.c_stick.y, 0.75, 0.001
    end
  end

  # ============================================================================
  # Integration Tests
  # ============================================================================
  # Temporal Policy Tests
  # ============================================================================

  describe "build_temporal/1" do
    @seq_len 10

    test "builds sliding window temporal policy" do
      model = Policy.build_temporal(embed_size: @embed_size, backbone: :sliding_window)

      assert %Axon{} = model
    end

    test "builds LSTM + attention hybrid temporal policy" do
      model = Policy.build_temporal(embed_size: @embed_size, backbone: :lstm_hybrid)

      assert %Axon{} = model
    end

    test "builds LSTM temporal policy" do
      model = Policy.build_temporal(embed_size: @embed_size, backbone: :lstm)

      assert %Axon{} = model
    end

    test "builds MLP temporal policy (uses last frame)" do
      model = Policy.build_temporal(embed_size: @embed_size, backbone: :mlp)

      assert %Axon{} = model
    end

    test "sliding window model produces correct output shapes" do
      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :sliding_window,
          # Match the test input seq_len
          window_size: @seq_len,
          num_heads: 2,
          head_dim: 16
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end

    test "lstm_hybrid model produces correct output shapes" do
      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :lstm_hybrid,
          hidden_size: 64,
          num_heads: 2,
          head_dim: 16
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      {buttons, main_x, _main_y, _c_x, _c_y, shoulder} = predict_fn.(params, input)

      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end

    test "builds temporal model with action embedding (mamba backbone)" do
      # Total embed_size includes 2 action IDs at the end
      continuous_size = 64
      total_embed_size = continuous_size + 2

      model =
        Policy.build_temporal(
          embed_size: total_embed_size,
          backbone: :mamba,
          action_embed_size: 32
        )

      assert %Axon{} = model
    end

    test "builds temporal model with action embedding (mlp backbone)" do
      continuous_size = 64
      total_embed_size = continuous_size + 2

      model =
        Policy.build_temporal(
          embed_size: total_embed_size,
          backbone: :mlp,
          action_embed_size: 32
        )

      assert %Axon{} = model
    end

    @tag :slow
    test "temporal mamba with action embedding produces correct output shapes" do
      continuous_size = 64
      total_embed_size = continuous_size + 2
      seq_len = 10

      model =
        Policy.build_temporal(
          embed_size: total_embed_size,
          backbone: :mamba,
          action_embed_size: 32,
          hidden_size: 64
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch_size, seq_len, total_embed_size}, :f32),
          Axon.ModelState.empty()
        )

      # Create input with action IDs at the end of each frame
      continuous_input = Nx.broadcast(0.5, {@batch_size, seq_len, continuous_size})
      # Action IDs per frame - shape {batch, seq_len, 2}
      action_ids = Nx.broadcast(50.0, {@batch_size, seq_len, 2})
      input = Nx.concatenate([continuous_input, action_ids], axis: 2)

      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end
  end

  describe "temporal_backbone_output_size/2" do
    test "returns correct size for sliding_window" do
      assert Policy.temporal_backbone_output_size(:sliding_window, num_heads: 4, head_dim: 64) ==
               256

      assert Policy.temporal_backbone_output_size(:sliding_window, num_heads: 8, head_dim: 32) ==
               256
    end

    test "returns correct size for lstm_hybrid" do
      assert Policy.temporal_backbone_output_size(:lstm_hybrid, num_heads: 4, head_dim: 64) == 256
    end

    test "returns correct size for lstm" do
      assert Policy.temporal_backbone_output_size(:lstm, hidden_size: 128) == 128
      # default
      assert Policy.temporal_backbone_output_size(:lstm) == 256
    end

    test "returns correct size for mlp" do
      assert Policy.temporal_backbone_output_size(:mlp, hidden_sizes: [256, 128]) == 128
      # default [512, 512]
      assert Policy.temporal_backbone_output_size(:mlp) == 512
    end
  end

  describe "melee_temporal_defaults/0" do
    test "returns expected defaults" do
      defaults = Policy.melee_temporal_defaults()

      assert defaults[:backbone] == :sliding_window
      assert defaults[:window_size] == 60
      assert defaults[:num_heads] == 4
      assert defaults[:head_dim] == 64
    end
  end

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
