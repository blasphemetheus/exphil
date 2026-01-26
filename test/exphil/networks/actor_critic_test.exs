defmodule ExPhil.Networks.ActorCriticTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.ActorCritic

  # Helper to convert tensor to number, handling {1} shape
  defp tensor_to_number(tensor) do
    case Nx.shape(tensor) do
      {} -> Nx.to_number(tensor)
      {1} -> tensor |> Nx.squeeze() |> Nx.to_number()
      _ -> raise "Expected scalar or {1} tensor, got #{inspect(Nx.shape(tensor))}"
    end
  end

  describe "build/1" do
    test "returns tuple of policy and value models" do
      {policy, value} = ActorCritic.build(embed_size: 64)

      assert %Axon{} = policy
      assert %Axon{} = value
    end
  end

  describe "build_combined/1" do
    test "creates combined actor-critic model" do
      model = ActorCritic.build_combined(embed_size: 64, hidden_sizes: [32])

      assert %Axon{} = model
    end

    @tag :slow
    test "outputs policy logits and value" do
      model = ActorCritic.build_combined(embed_size: 64, hidden_sizes: [32])

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 64}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {4, 64})
      output = predict_fn.(params, input)

      assert %{policy: policy_logits, value: value} = output

      # Policy logits should be a tuple of 6 tensors
      {buttons, main_x, main_y, c_x, c_y, shoulder} = policy_logits

      # 8 buttons
      assert Nx.shape(buttons) == {4, 8}
      # 16 buckets + 1
      assert Nx.shape(main_x) == {4, 17}
      assert Nx.shape(main_y) == {4, 17}
      assert Nx.shape(c_x) == {4, 17}
      assert Nx.shape(c_y) == {4, 17}
      # 4 buckets + 1
      assert Nx.shape(shoulder) == {4, 5}

      # Value should be [batch]
      assert Nx.shape(value) == {4}
    end

    test "accepts custom axis_buckets" do
      model =
        ActorCritic.build_combined(
          embed_size: 64,
          hidden_sizes: [32],
          axis_buckets: 8,
          shoulder_buckets: 2
        )

      assert %Axon{} = model
    end
  end

  describe "compute_log_probs/2" do
    test "computes log probabilities for actions" do
      logits = {
        # buttons
        Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        # main_x
        Nx.tensor([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        # main_y
        Nx.tensor([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        # c_x
        Nx.tensor([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        # c_y
        Nx.tensor([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        # shoulder
        Nx.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
      }

      actions = %{
        buttons: Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),
        main_x: Nx.tensor([8]),
        main_y: Nx.tensor([8]),
        c_x: Nx.tensor([8]),
        c_y: Nx.tensor([8]),
        shoulder: Nx.tensor([2])
      }

      log_probs = ActorCritic.compute_log_probs(logits, actions)

      assert Nx.shape(log_probs) == {1}
      # Log probs should be negative (or zero for perfect match)
      assert tensor_to_number(log_probs) <= 0
    end

    test "higher logit for selected action gives higher log prob" do
      # Action at position 8
      logits_low = {
        Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        # Uniform logits
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 5})
      }

      # Higher logit at position 8
      main_x_high = Nx.put_slice(Nx.broadcast(0.0, {1, 17}), [0, 8], Nx.tensor([[5.0]]))
      logits_high = put_elem(logits_low, 1, main_x_high)

      actions = %{
        buttons: Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),
        main_x: Nx.tensor([8]),
        main_y: Nx.tensor([8]),
        c_x: Nx.tensor([8]),
        c_y: Nx.tensor([8]),
        shoulder: Nx.tensor([2])
      }

      log_prob_low = ActorCritic.compute_log_probs(logits_low, actions)
      log_prob_high = ActorCritic.compute_log_probs(logits_high, actions)

      # Higher logit should give higher log probability
      assert tensor_to_number(log_prob_high) > tensor_to_number(log_prob_low)
    end
  end

  describe "compute_entropy/1" do
    test "returns positive entropy" do
      logits = {
        # Uniform buttons (max entropy)
        Nx.broadcast(0.0, {4, 8}),
        # Uniform stick
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 5})
      }

      entropy = ActorCritic.compute_entropy(logits)

      assert Nx.to_number(entropy) > 0
    end

    test "peaked distribution has lower entropy than uniform" do
      # Uniform distribution
      uniform_logits = {
        Nx.broadcast(0.0, {1, 8}),
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 17}),
        Nx.broadcast(0.0, {1, 5})
      }

      # Peaked distribution (high logit for one action)
      main_x_peaked = Nx.put_slice(Nx.broadcast(0.0, {1, 17}), [0, 8], Nx.tensor([[10.0]]))
      peaked_logits = put_elem(uniform_logits, 1, main_x_peaked)

      uniform_entropy = ActorCritic.compute_entropy(uniform_logits)
      peaked_entropy = ActorCritic.compute_entropy(peaked_logits)

      assert Nx.to_number(uniform_entropy) > Nx.to_number(peaked_entropy)
    end
  end

  describe "policy_loss/5" do
    test "computes clipped policy loss" do
      logits = {
        Nx.broadcast(0.0, {4, 8}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 5})
      }

      actions = %{
        buttons: Nx.broadcast(0, {4, 8}),
        main_x: Nx.tensor([8, 8, 8, 8]),
        main_y: Nx.tensor([8, 8, 8, 8]),
        c_x: Nx.tensor([8, 8, 8, 8]),
        c_y: Nx.tensor([8, 8, 8, 8]),
        shoulder: Nx.tensor([2, 2, 2, 2])
      }

      advantages = Nx.tensor([1.0, -1.0, 0.5, -0.5])

      {policy_loss, approx_kl, clip_fraction} =
        ActorCritic.policy_loss(logits, logits, actions, advantages, 0.2)

      assert is_struct(policy_loss, Nx.Tensor)
      assert is_struct(approx_kl, Nx.Tensor)
      assert is_struct(clip_fraction, Nx.Tensor)
    end

    test "identical old and new logits have zero KL" do
      logits = {
        Nx.broadcast(0.0, {4, 8}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 5})
      }

      actions = %{
        buttons: Nx.broadcast(0, {4, 8}),
        main_x: Nx.tensor([8, 8, 8, 8]),
        main_y: Nx.tensor([8, 8, 8, 8]),
        c_x: Nx.tensor([8, 8, 8, 8]),
        c_y: Nx.tensor([8, 8, 8, 8]),
        shoulder: Nx.tensor([2, 2, 2, 2])
      }

      advantages = Nx.tensor([1.0, 1.0, 1.0, 1.0])

      {_policy_loss, approx_kl, _clip_fraction} =
        ActorCritic.policy_loss(logits, logits, actions, advantages, 0.2)

      # When old and new logits are identical, KL should be 0
      assert_in_delta Nx.to_number(approx_kl), 0.0, 0.001
    end
  end

  describe "ppo_loss/8" do
    test "computes full PPO loss with all components" do
      logits = {
        Nx.broadcast(0.0, {4, 8}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 17}),
        Nx.broadcast(0.0, {4, 5})
      }

      actions = %{
        buttons: Nx.broadcast(0, {4, 8}),
        main_x: Nx.tensor([8, 8, 8, 8]),
        main_y: Nx.tensor([8, 8, 8, 8]),
        c_x: Nx.tensor([8, 8, 8, 8]),
        c_y: Nx.tensor([8, 8, 8, 8]),
        shoulder: Nx.tensor([2, 2, 2, 2])
      }

      advantages = Nx.tensor([1.0, -1.0, 0.5, -0.5])
      values = Nx.tensor([0.5, 0.5, 0.5, 0.5])
      returns = Nx.tensor([1.0, 0.0, 0.75, 0.25])

      losses =
        ActorCritic.ppo_loss(
          logits,
          logits,
          actions,
          advantages,
          values,
          values,
          returns
        )

      assert Map.has_key?(losses, :total)
      assert Map.has_key?(losses, :policy)
      assert Map.has_key?(losses, :value)
      assert Map.has_key?(losses, :entropy)
      assert Map.has_key?(losses, :approx_kl)
      assert Map.has_key?(losses, :clip_fraction)
    end
  end

  describe "default_config/0" do
    test "returns map with PPO hyperparameters" do
      config = ActorCritic.default_config()

      assert config.gamma == 0.99
      assert config.gae_lambda == 0.95
      assert config.clip_range == 0.2
      assert config.value_coef == 0.5
      assert config.entropy_coef == 0.01
    end
  end

  describe "create_optimizer/1" do
    test "returns optimizer with gradient clipping" do
      optimizer = ActorCritic.create_optimizer(learning_rate: 1.0e-4)

      # Optimizer should be a tuple of {init_fn, update_fn}
      assert {init_fn, update_fn} = optimizer
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end
  end
end
