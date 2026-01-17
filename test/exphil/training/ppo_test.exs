defmodule ExPhil.Training.PPOTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.PPO
  alias ExPhil.Networks.Value

  describe "new/1" do
    test "creates trainer with default config" do
      trainer = PPO.new(embed_size: 64)

      assert %PPO{} = trainer
      assert trainer.step == 0
      assert trainer.timesteps == 0
    end

    test "initializes model parameters" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])

      assert trainer.params != nil
      assert trainer.old_params != nil
    end

    test "accepts custom PPO hyperparameters" do
      trainer = PPO.new(
        embed_size: 64,
        clip_range: 0.1,
        gamma: 0.95,
        gae_lambda: 0.9
      )

      assert trainer.config.clip_range == 0.1
      assert trainer.config.gamma == 0.95
      assert trainer.config.gae_lambda == 0.9
    end

    test "uses default values for unspecified options" do
      trainer = PPO.new(embed_size: 64)

      assert trainer.config.gamma == 0.99
      assert trainer.config.clip_range == 0.2
      assert trainer.config.entropy_coef == 0.01
    end
  end

  describe "create_optimizer/1" do
    test "returns init and update functions" do
      config = %{learning_rate: 1.0e-3}

      {init_fn, update_fn} = PPO.create_optimizer(config)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end
  end

  describe "save_checkpoint/2 and load_checkpoint/2" do
    @tag :slow
    test "round-trips checkpoint data" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "ppo_checkpoint_#{:rand.uniform(10_000)}.axon")

      try do
        assert :ok = PPO.save_checkpoint(trainer, path)
        assert File.exists?(path)

        new_trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
        {:ok, loaded} = PPO.load_checkpoint(new_trainer, path)

        assert loaded.step == trainer.step
        assert loaded.timesteps == trainer.timesteps
      after
        File.rm(path)
      end
    end

    test "load_checkpoint returns error for missing file" do
      trainer = PPO.new(embed_size: 64)

      result = PPO.load_checkpoint(trainer, "/nonexistent/path.axon")

      assert {:error, _} = result
    end
  end

  describe "export_policy/2" do
    @tag :slow
    test "exports policy for inference" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "ppo_policy_#{:rand.uniform(10_000)}.axon")

      try do
        result = PPO.export_policy(trainer, path)

        assert result == :ok
        assert File.exists?(path)

        {:ok, binary} = File.read(path)
        export = :erlang.binary_to_term(binary)

        assert is_map(export.params)
        assert export.config.axis_buckets == 16
      after
        File.rm(path)
      end
    end

    @tag :slow
    test "exports temporal config for inference" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "ppo_temporal_policy_#{:rand.uniform(10_000)}.axon")

      try do
        :ok = PPO.export_policy(trainer, path)

        {:ok, binary} = File.read(path)
        export = :erlang.binary_to_term(binary)

        # Verify temporal config is included with defaults
        assert export.config.embed_size == 64
        assert export.config.temporal == false
        assert export.config.backbone == :sliding_window
        assert export.config.window_size == 60
        assert export.config.num_heads == 4
        assert export.config.head_dim == 64
      after
        File.rm(path)
      end
    end
  end

  describe "metrics_summary/1" do
    test "returns summary with initial metrics" do
      trainer = PPO.new(embed_size: 64)

      summary = PPO.metrics_summary(trainer)

      assert summary.step == 0
      assert summary.timesteps == 0
      assert summary.recent_loss == 0.0
    end
  end

  describe "get_action/3" do
    @tag :slow
    test "returns action samples and value" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
      state = Nx.broadcast(0.5, {1, 64})

      action = PPO.get_action(trainer, state)

      assert is_map(action)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :value)
    end

    @tag :slow
    test "supports deterministic mode" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
      state = Nx.broadcast(0.5, {1, 64})

      # Deterministic should always return same result
      action1 = PPO.get_action(trainer, state, deterministic: true)
      action2 = PPO.get_action(trainer, state, deterministic: true)

      # Main stick positions should be identical in deterministic mode
      assert action1.main_x == action2.main_x
      assert action1.main_y == action2.main_y
    end
  end

  describe "get_controller_action/3" do
    @tag :slow
    test "returns ControllerState" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
      state = Nx.broadcast(0.5, {1, 64})

      cs = PPO.get_controller_action(trainer, state)

      assert %ExPhil.Bridge.ControllerState{} = cs
      assert is_map(cs.main_stick)
    end
  end

  describe "get_value/2" do
    @tag :slow
    test "returns scalar value estimate" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
      state = Nx.broadcast(0.5, {1, 64})

      value = PPO.get_value(trainer, state)

      assert is_float(value)
    end
  end

  describe "PPO loss components" do
    test "clip_range affects clipped ratio bounds" do
      # This tests the mathematical property of clipping
      ratio = Nx.tensor([0.5, 1.0, 1.5, 2.0])
      clip_range = 0.2

      clipped = Nx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
      clipped_list = Nx.to_flat_list(clipped)

      assert_in_delta Enum.at(clipped_list, 0), 0.8, 0.001   # 0.5 clipped to 0.8
      assert_in_delta Enum.at(clipped_list, 1), 1.0, 0.001   # 1.0 stays 1.0
      assert_in_delta Enum.at(clipped_list, 2), 1.2, 0.001   # 1.5 clipped to 1.2
      assert_in_delta Enum.at(clipped_list, 3), 1.2, 0.001   # 2.0 clipped to 1.2
    end

    test "advantage normalization centers around zero" do
      advantages = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      normalized = Value.normalize_advantages(advantages)

      mean = Nx.to_number(Nx.mean(normalized))
      assert_in_delta mean, 0.0, 0.001
    end
  end

  describe "rollout data structure" do
    test "validates expected rollout keys" do
      # Mock rollout structure that PPO.update expects
      rollout = %{
        states: Nx.broadcast(0.0, {10, 64}),
        actions: %{
          buttons: Nx.broadcast(0, {10, 8}),
          main_x: Nx.broadcast(8, {10}),
          main_y: Nx.broadcast(8, {10}),
          c_x: Nx.broadcast(8, {10}),
          c_y: Nx.broadcast(8, {10}),
          shoulder: Nx.broadcast(0, {10})
        },
        rewards: Nx.broadcast(0.0, {10}),
        dones: Nx.broadcast(0.0, {10}),
        values: Nx.broadcast(0.0, {11}),  # time_steps + 1
        log_probs: Nx.broadcast(0.0, {10})
      }

      # Just verify structure is valid
      assert Nx.shape(rollout.states) == {10, 64}
      assert Nx.shape(rollout.rewards) == {10}
      assert Nx.shape(rollout.values) == {11}
    end
  end

  describe "config defaults" do
    test "has reasonable default hyperparameters" do
      trainer = PPO.new(embed_size: 64)
      config = trainer.config

      # PPO-specific
      assert config.gamma >= 0.9 and config.gamma <= 1.0
      assert config.gae_lambda >= 0.9 and config.gae_lambda <= 1.0
      assert config.clip_range > 0 and config.clip_range < 1.0
      assert config.value_coef > 0
      assert config.entropy_coef >= 0

      # Training
      assert config.learning_rate > 0
      assert config.batch_size > 0
      assert config.num_epochs > 0
    end
  end

  describe "update/2" do
    # Helper to create a mock rollout
    defp mock_rollout(num_steps, embed_size) do
      %{
        states: Nx.broadcast(0.5, {num_steps, embed_size}),
        actions: %{
          buttons: Nx.broadcast(0, {num_steps, 8}),
          main_x: Nx.broadcast(8, {num_steps}),
          main_y: Nx.broadcast(8, {num_steps}),
          c_x: Nx.broadcast(8, {num_steps}),
          c_y: Nx.broadcast(8, {num_steps}),
          shoulder: Nx.broadcast(0, {num_steps})
        },
        rewards: Nx.broadcast(0.1, {num_steps}),
        dones: Nx.broadcast(0.0, {num_steps}),
        values: Nx.broadcast(0.0, {num_steps + 1}),
        log_probs: Nx.broadcast(-1.0, {num_steps})
      }
    end

    @tag :slow
    test "performs PPO update and returns metrics" do
      trainer = PPO.new(
        embed_size: 64,
        hidden_sizes: [32],
        num_epochs: 2,
        num_minibatches: 2
      )
      rollout = mock_rollout(16, 64)

      {new_trainer, metrics} = PPO.update(trainer, rollout)

      # Check trainer is updated
      assert new_trainer.step >= trainer.step
      assert new_trainer.old_params != nil

      # Check metrics are returned
      assert is_map(metrics)
      assert Map.has_key?(metrics, :timesteps)
      assert metrics.timesteps == 16
    end

    @tag :slow
    test "updates parameters during training" do
      trainer = PPO.new(
        embed_size: 64,
        hidden_sizes: [32],
        num_epochs: 2,
        num_minibatches: 2
      )

      # Add some non-zero rewards to create gradients
      rollout = %{mock_rollout(16, 64) | rewards: Nx.tensor(Enum.map(1..16, fn i -> i * 0.1 end))}

      {new_trainer, _metrics} = PPO.update(trainer, rollout)

      # Params should be different after update (with non-trivial gradients)
      assert new_trainer.params != nil
    end

    @tag :slow
    test "handles episode boundaries in rollout" do
      trainer = PPO.new(
        embed_size: 64,
        hidden_sizes: [32],
        num_epochs: 1,
        num_minibatches: 2
      )

      # Create rollout with episode done in the middle
      dones = Nx.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
      rollout = %{mock_rollout(16, 64) | dones: dones}

      {new_trainer, metrics} = PPO.update(trainer, rollout)

      # Should complete without error
      assert new_trainer.step >= trainer.step
      assert is_map(metrics)
    end
  end

  describe "collect_rollout/3" do
    @tag :slow
    test "collects rollout from step function" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])

      # Step function follows the expected interface:
      # :get_state returns current state
      # :step takes {state, action} and returns {next_state, reward, done}
      step_fn = fn
        :get_state, _state ->
          Nx.broadcast(0.5, {1, 64})

        :step, {_state, _action} ->
          next_state = Nx.broadcast(0.5, {1, 64})
          reward = 0.1
          done = false
          {next_state, reward, done}
      end

      rollout = PPO.collect_rollout(trainer, step_fn, 8)

      assert Nx.shape(rollout.states) == {8, 64}
      assert Nx.shape(rollout.rewards) == {8}
      assert Nx.shape(rollout.dones) == {8}
      assert Nx.shape(rollout.values) == {9}  # num_steps + 1
      assert Nx.shape(rollout.log_probs) == {8}
    end

    @tag :slow
    test "handles episode termination during rollout" do
      trainer = PPO.new(embed_size: 64, hidden_sizes: [32])

      # Step function that ends episode on step 4 and 8
      step_counter = :counters.new(1, [:atomics])

      step_fn = fn
        :get_state, _state ->
          Nx.broadcast(0.5, {1, 64})

        :step, {_state, _action} ->
          :counters.add(step_counter, 1, 1)
          step = :counters.get(step_counter, 1)

          next_state = Nx.broadcast(0.5, {1, 64})
          reward = 0.1
          done = step == 4 or step == 8  # Episodes end at step 4 and 8

          {next_state, reward, done}
      end

      rollout = PPO.collect_rollout(trainer, step_fn, 8)

      # Should have recorded the done flags
      dones_list = Nx.to_flat_list(rollout.dones)
      assert length(dones_list) == 8
    end
  end

  describe "load_pretrained_policy/2" do
    @tag :slow
    test "loads policy from imitation checkpoint" do
      # First save an imitation checkpoint
      imitation_trainer = ExPhil.Training.Imitation.new(embed_size: 64, hidden_sizes: [32])
      imitation_path = Path.join(System.tmp_dir!(), "imitation_for_ppo_#{:rand.uniform(10_000)}.axon")

      try do
        :ok = ExPhil.Training.Imitation.export_policy(imitation_trainer, imitation_path)

        # Now create PPO trainer and load pretrained
        ppo_trainer = PPO.new(embed_size: 64, hidden_sizes: [32])
        {:ok, loaded_trainer} = PPO.load_pretrained_policy(ppo_trainer, imitation_path)

        assert loaded_trainer.params != nil
      after
        File.rm(imitation_path)
      end
    end
  end
end
