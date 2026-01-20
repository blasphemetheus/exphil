defmodule ExPhil.NetworksTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks
  alias ExPhil.Networks.{Policy, Value, ActorCritic}

  @embed_size 512
  @batch_size 4

  # Helper to create random tensors (Nx.Random requires a key)
  defp random_tensor(shape) do
    key = Nx.Random.key(System.system_time())
    {tensor, _new_key} = Nx.Random.uniform(key, shape: shape)
    tensor
  end

  # ============================================================================
  # Policy Network Tests
  # ============================================================================

  describe "Policy.build/1" do
    test "builds a valid Axon model" do
      model = Policy.build(embed_size: @embed_size)

      assert %Axon{} = model
    end

    @tag :slow
    test "model outputs tuple of 6 components" do
      model = Policy.build(embed_size: @embed_size)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({@batch_size, @embed_size}, :f32), Axon.ModelState.empty())
      state = random_tensor({@batch_size, @embed_size})

      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, state)

      # Check shapes
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end

    @tag :slow
    test "respects custom axis_buckets" do
      model = Policy.build(embed_size: @embed_size, axis_buckets: 8)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())
      state = random_tensor({1, @embed_size})

      {_buttons, main_x, _main_y, _c_x, _c_y, _shoulder} = predict_fn.(params, state)

      # 8 buckets + 1 = 9 output classes
      assert Nx.shape(main_x) == {1, 9}
    end

    @tag :slow
    test "supports layer normalization" do
      # Build model without layer norm
      model_no_ln = Policy.build(embed_size: @embed_size, hidden_sizes: [64, 64], layer_norm: false)
      {init_fn_no_ln, _} = Axon.build(model_no_ln)
      params_no_ln = init_fn_no_ln.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())

      # Build model with layer norm
      model_with_ln = Policy.build(embed_size: @embed_size, hidden_sizes: [64, 64], layer_norm: true)
      {init_fn_with_ln, _} = Axon.build(model_with_ln)
      params_with_ln = init_fn_with_ln.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())

      # Layer norm model should have additional parameters (gamma, beta per layer)
      # backbone_ln_0 and backbone_ln_1 should exist in layer norm model's data
      assert Map.has_key?(params_with_ln.data, "backbone_ln_0")
      assert Map.has_key?(params_with_ln.data, "backbone_ln_1")
      refute Map.has_key?(params_no_ln.data, "backbone_ln_0")
      refute Map.has_key?(params_no_ln.data, "backbone_ln_1")
    end
  end

  describe "Policy.sample_buttons/2" do
    test "returns boolean tensor" do
      logits = Nx.tensor([[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0]])
      result = Policy.sample_buttons(logits, true)

      assert Nx.shape(result) == {1, 8}
      assert Nx.type(result) == {:u, 8}  # Boolean type
    end

    test "deterministic mode uses threshold" do
      # High logits -> 1, low logits -> 0
      logits = Nx.tensor([[5.0, 5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0]])
      result = Policy.sample_buttons(logits, true)

      expected = Nx.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], type: :u8)
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end
  end

  describe "Policy.sample_categorical/3" do
    test "returns indices" do
      logits = Nx.tensor([[0.0, 0.0, 100.0, 0.0, 0.0]])  # Strong preference for index 2
      result = Policy.sample_categorical(logits, 1.0, true)

      assert Nx.shape(result) == {1}
      assert Nx.to_number(Nx.squeeze(result)) == 2
    end

    test "temperature affects distribution" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]])

      # High temperature -> more uniform
      # Low temperature -> more peaked
      # Just verify it runs without error
      _low_temp = Policy.sample_categorical(logits, 0.1, false)
      _high_temp = Policy.sample_categorical(logits, 10.0, false)
    end
  end

  describe "Policy.imitation_loss/2" do
    test "computes loss for all components" do
      logits = %{
        buttons: random_tensor({@batch_size, 8}),
        main_x: random_tensor({@batch_size, 17}),
        main_y: random_tensor({@batch_size, 17}),
        c_x: random_tensor({@batch_size, 17}),
        c_y: random_tensor({@batch_size, 17}),
        shoulder: random_tensor({@batch_size, 5})
      }

      targets = %{
        buttons: Nx.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]]),
        main_x: Nx.tensor([8, 8, 8, 8]),
        main_y: Nx.tensor([8, 8, 8, 8]),
        c_x: Nx.tensor([8, 8, 8, 8]),
        c_y: Nx.tensor([8, 8, 8, 8]),
        shoulder: Nx.tensor([0, 0, 0, 0])
      }

      loss = Policy.imitation_loss(logits, targets)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  describe "Policy.to_controller_state/2" do
    test "converts samples to ControllerState" do
      samples = %{
        buttons: Nx.tensor([1, 0, 0, 0, 0, 0, 0, 0]),  # A pressed
        main_x: Nx.tensor(8),   # Center
        main_y: Nx.tensor(16),  # Full up
        c_x: Nx.tensor(8),
        c_y: Nx.tensor(8),
        shoulder: Nx.tensor(0)
      }

      controller = Policy.to_controller_state(samples)

      assert controller.button_a == true
      assert controller.button_b == false
      assert controller.main_stick.x == 0.5
      assert controller.main_stick.y == 1.0
    end
  end

  describe "Policy.output_sizes/1" do
    test "returns correct sizes for default config" do
      sizes = Policy.output_sizes()

      assert sizes.buttons == 8
      assert sizes.main_x == 17
      assert sizes.main_y == 17
      assert sizes.c_x == 17
      assert sizes.c_y == 17
      assert sizes.shoulder == 5
    end

    test "respects custom buckets" do
      sizes = Policy.output_sizes(axis_buckets: 8, shoulder_buckets: 2)

      assert sizes.main_x == 9
      assert sizes.shoulder == 3
    end
  end

  # ============================================================================
  # Value Network Tests
  # ============================================================================

  describe "Value.build/1" do
    test "builds a valid Axon model" do
      model = Value.build(embed_size: @embed_size)

      assert %Axon{} = model
    end

    @tag :slow
    test "model outputs scalar values" do
      model = Value.build(embed_size: @embed_size)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({@batch_size, @embed_size}, :f32), Axon.ModelState.empty())
      state = random_tensor({@batch_size, @embed_size})

      values = predict_fn.(params, state)

      assert Nx.shape(values) == {@batch_size}
    end
  end

  describe "Value.value_loss/2" do
    test "computes MSE loss" do
      predicted = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      targets = Nx.tensor([1.5, 2.5, 3.5, 4.5])

      loss = Value.value_loss(predicted, targets)

      # MSE of 0.5^2 = 0.25
      assert_in_delta Nx.to_number(loss), 0.25, 0.001
    end
  end

  describe "Value.huber_loss/3" do
    test "uses quadratic loss for small errors" do
      predicted = Nx.tensor([1.0])
      targets = Nx.tensor([1.1])

      loss = Value.huber_loss(predicted, targets, 1.0)

      # 0.5 * 0.1^2 = 0.005
      assert_in_delta Nx.to_number(loss), 0.005, 0.001
    end

    test "uses linear loss for large errors" do
      predicted = Nx.tensor([1.0])
      targets = Nx.tensor([3.0])  # Error of 2, > delta

      loss = Value.huber_loss(predicted, targets, 1.0)

      # delta * |error| - 0.5 * delta^2 = 1.0 * 2.0 - 0.5 = 1.5
      assert_in_delta Nx.to_number(loss), 1.5, 0.001
    end
  end

  describe "Value.compute_gae/5" do
    test "computes advantages and returns" do
      rewards = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      values = Nx.tensor([0.5, 0.5, 0.5, 0.5, 0.5])  # +1 for bootstrap
      dones = Nx.tensor([0.0, 0.0, 0.0, 1.0])

      {advantages, returns} = Value.compute_gae(rewards, values, dones, 0.99, 0.95)

      assert Nx.shape(advantages) == {4}
      assert Nx.shape(returns) == {4}

      # Returns should be positive (got rewards)
      assert Nx.to_number(Nx.mean(returns)) > 0
    end
  end

  describe "Value.normalize_advantages/1" do
    test "normalizes to zero mean and unit variance" do
      advantages = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      normalized = Value.normalize_advantages(advantages)

      mean = Nx.to_number(Nx.mean(normalized))
      std = Nx.to_number(Nx.standard_deviation(normalized))

      assert_in_delta mean, 0.0, 0.001
      assert_in_delta std, 1.0, 0.01
    end
  end

  # ============================================================================
  # ActorCritic Tests
  # ============================================================================

  describe "ActorCritic.build/1" do
    test "returns tuple of policy and value models" do
      {policy, value} = ActorCritic.build(embed_size: @embed_size)

      assert %Axon{} = policy
      assert %Axon{} = value
    end
  end

  describe "ActorCritic.build_combined/1" do
    test "builds combined model with policy and value outputs" do
      model = ActorCritic.build_combined(embed_size: @embed_size)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({@batch_size, @embed_size}, :f32), Axon.ModelState.empty())
      state = random_tensor({@batch_size, @embed_size})

      %{policy: policy_logits, value: values} = predict_fn.(params, state)

      # Policy outputs tuple
      {buttons, main_x, _main_y, _c_x, _c_y, _shoulder} = policy_logits
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}

      # Value output
      assert Nx.shape(values) == {@batch_size}
    end
  end

  describe "ActorCritic.compute_entropy/1" do
    test "computes positive entropy" do
      # Uniform-ish logits should have high entropy
      logits = {
        Nx.broadcast(0.0, {@batch_size, 8}),  # buttons
        Nx.broadcast(0.0, {@batch_size, 17}), # main_x
        Nx.broadcast(0.0, {@batch_size, 17}), # main_y
        Nx.broadcast(0.0, {@batch_size, 17}), # c_x
        Nx.broadcast(0.0, {@batch_size, 17}), # c_y
        Nx.broadcast(0.0, {@batch_size, 5})   # shoulder
      }

      entropy = ActorCritic.compute_entropy(logits)

      assert Nx.to_number(entropy) > 0
    end

    test "peaked distribution has lower entropy" do
      # Very peaked logits
      peaked_main_x = Nx.concatenate([
        Nx.broadcast(-100.0, {@batch_size, 8}),
        Nx.broadcast(100.0, {@batch_size, 1}),
        Nx.broadcast(-100.0, {@batch_size, 8})
      ], axis: 1)

      uniform_main_x = Nx.broadcast(0.0, {@batch_size, 17})

      peaked_logits = {
        Nx.broadcast(0.0, {@batch_size, 8}),
        peaked_main_x,
        Nx.broadcast(0.0, {@batch_size, 17}),
        Nx.broadcast(0.0, {@batch_size, 17}),
        Nx.broadcast(0.0, {@batch_size, 17}),
        Nx.broadcast(0.0, {@batch_size, 5})
      }

      uniform_logits = {
        Nx.broadcast(0.0, {@batch_size, 8}),
        uniform_main_x,
        Nx.broadcast(0.0, {@batch_size, 17}),
        Nx.broadcast(0.0, {@batch_size, 17}),
        Nx.broadcast(0.0, {@batch_size, 17}),
        Nx.broadcast(0.0, {@batch_size, 5})
      }

      peaked_entropy = Nx.to_number(ActorCritic.compute_entropy(peaked_logits))
      uniform_entropy = Nx.to_number(ActorCritic.compute_entropy(uniform_logits))

      assert peaked_entropy < uniform_entropy
    end
  end

  describe "ActorCritic.default_config/0" do
    test "returns valid config map" do
      config = ActorCritic.default_config()

      assert is_map(config)
      assert config.gamma == 0.99
      assert config.clip_range == 0.2
      assert config.learning_rate == 3.0e-4
    end
  end

  # ============================================================================
  # Main API Tests
  # ============================================================================

  describe "Networks.build/1" do
    test "delegates to ActorCritic.build/1" do
      {policy, value} = Networks.build(embed_size: @embed_size)

      assert %Axon{} = policy
      assert %Axon{} = value
    end
  end

  describe "Networks.build_combined/1" do
    test "delegates to ActorCritic.build_combined/1" do
      model = Networks.build_combined(embed_size: @embed_size)

      assert %Axon{} = model
    end
  end

  describe "Networks.action_dims/1" do
    test "returns total action dimensions" do
      dims = Networks.action_dims()

      # 8 buttons + 17*4 sticks + 5 shoulder = 81
      assert dims == 8 + 17 + 17 + 17 + 17 + 5
    end
  end

  describe "Networks.build_policy/1" do
    test "delegates to Policy.build/1" do
      model = Networks.build_policy(embed_size: @embed_size)

      assert %Axon{} = model
    end
  end

  describe "Networks.build_value/1" do
    test "delegates to Value.build/1" do
      model = Networks.build_value(embed_size: @embed_size)

      assert %Axon{} = model
    end
  end

  describe "Networks.sample/4" do
    @tag :slow
    test "samples actions from policy" do
      model = Policy.build(embed_size: @embed_size)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())
      state = random_tensor({1, @embed_size})

      samples = Networks.sample(params, predict_fn, state)

      assert Map.has_key?(samples, :buttons)
      assert Map.has_key?(samples, :main_x)
      assert Map.has_key?(samples, :main_y)
      assert Map.has_key?(samples, :c_x)
      assert Map.has_key?(samples, :c_y)
      assert Map.has_key?(samples, :shoulder)
    end

    @tag :slow
    test "supports deterministic mode" do
      model = Policy.build(embed_size: @embed_size)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())
      state = random_tensor({1, @embed_size})

      samples1 = Networks.sample(params, predict_fn, state, deterministic: true)
      samples2 = Networks.sample(params, predict_fn, state, deterministic: true)

      assert Nx.to_number(Nx.squeeze(samples1.main_x)) == Nx.to_number(Nx.squeeze(samples2.main_x))
    end
  end

  describe "Networks.to_controller_state/2" do
    test "converts samples to ControllerState" do
      samples = %{
        buttons: Nx.tensor([1, 0, 1, 0, 0, 0, 0, 0]),
        main_x: Nx.tensor(8),
        main_y: Nx.tensor(8),
        c_x: Nx.tensor(8),
        c_y: Nx.tensor(8),
        shoulder: Nx.tensor(0)
      }

      cs = Networks.to_controller_state(samples)

      assert %ExPhil.Bridge.ControllerState{} = cs
      assert cs.button_a == true
      assert cs.button_x == true
      assert cs.button_b == false
    end
  end

  describe "Networks.create_optimizer/1" do
    test "creates optimizer with default options" do
      optimizer = Networks.create_optimizer()

      assert is_tuple(optimizer)
    end

    test "accepts custom learning rate" do
      optimizer = Networks.create_optimizer(learning_rate: 1.0e-3)

      assert is_tuple(optimizer)
    end
  end

  describe "Networks.compute_gae/5" do
    test "delegates to Value.compute_gae/5" do
      rewards = Nx.tensor([1.0, 1.0, 1.0])
      values = Nx.tensor([0.5, 0.5, 0.5, 0.5])
      dones = Nx.tensor([0.0, 0.0, 1.0])

      {advantages, returns} = Networks.compute_gae(rewards, values, dones)

      assert Nx.shape(advantages) == {3}
      assert Nx.shape(returns) == {3}
    end
  end

  describe "Networks.default_config/0" do
    test "delegates to ActorCritic.default_config/0" do
      config = Networks.default_config()

      assert config.gamma == 0.99
      assert config.clip_range == 0.2
    end
  end

  describe "Networks.output_sizes/1" do
    test "delegates to Policy.output_sizes/1" do
      sizes = Networks.output_sizes()

      assert sizes.buttons == 8
      assert sizes.main_x == 17
    end
  end

  describe "Networks.imitation_loss/2" do
    test "delegates to Policy.imitation_loss/2" do
      logits = %{
        buttons: random_tensor({2, 8}),
        main_x: random_tensor({2, 17}),
        main_y: random_tensor({2, 17}),
        c_x: random_tensor({2, 17}),
        c_y: random_tensor({2, 17}),
        shoulder: random_tensor({2, 5})
      }

      targets = %{
        buttons: Nx.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]]),
        main_x: Nx.tensor([8, 8]),
        main_y: Nx.tensor([8, 8]),
        c_x: Nx.tensor([8, 8]),
        c_y: Nx.tensor([8, 8]),
        shoulder: Nx.tensor([0, 0])
      }

      loss = Networks.imitation_loss(logits, targets)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end
end
