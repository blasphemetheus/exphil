defmodule ExPhil.Training.SelfPlay.SelfPlayEnvTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.SelfPlay.SelfPlayEnv

  # Tests take 1-3 seconds each due to model initialization
  @moduletag :slow
  @moduletag :self_play

  # Create a minimal mock policy for testing
  defp mock_policy do
    # Simple MLP that outputs the right shapes for policy
    model = Axon.input("state", shape: {nil, 1991})
    |> Axon.dense(64, activation: :relu)
    |> then(fn x ->
      buttons = Axon.dense(x, 8, name: "buttons")
      main_x = Axon.dense(x, 17, name: "main_x")
      main_y = Axon.dense(x, 17, name: "main_y")
      c_x = Axon.dense(x, 17, name: "c_x")
      c_y = Axon.dense(x, 17, name: "c_y")
      shoulder = Axon.dense(x, 5, name: "shoulder")
      value = Axon.dense(x, 1, name: "value")

      Axon.container({
        {buttons, main_x, main_y, c_x, c_y, shoulder},
        value
      })
    end)

    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1991}, :f32), %{})

    {model, params}
  end

  describe "new/1" do
    test "creates environment with mock game type" do
      policy = mock_policy()

      assert {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: :cpu,
        game_type: :mock
      )

      assert env.game_type == :mock
      assert env.p2_policy == :cpu
      assert env.frame_count == 0
    end

    test "creates environment with default config" do
      policy = mock_policy()

      assert {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: :cpu
      )

      # Default is :mock
      assert env.game_type == :mock
    end

    test "accepts custom reward config" do
      policy = mock_policy()
      custom_reward = %{damage_dealt: 2.0, damage_taken: -1.5}

      assert {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: :cpu,
        reward_config: custom_reward
      )

      assert env.reward_config == custom_reward
    end

    test "accepts p2 as policy tuple" do
      policy = mock_policy()

      assert {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: policy,
        game_type: :mock
      )

      # P2 should be compiled (has predict_fn)
      assert is_tuple(env.p2_policy)
      assert tuple_size(env.p2_policy) == 3
    end
  end

  describe "step/1" do
    setup do
      policy = mock_policy()

      {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: :cpu,
        game_type: :mock
      )

      %{env: env}
    end

    test "returns experience on step", %{env: env} do
      result = SelfPlayEnv.step(env)

      assert {:ok, new_env, experience} = result
      assert is_map(experience)
      assert Map.has_key?(experience, :state)
      assert Map.has_key?(experience, :action)
      assert Map.has_key?(experience, :reward)
      assert Map.has_key?(experience, :done)
      assert new_env.frame_count == 1
    end

    test "increments frame count", %{env: env} do
      {:ok, env1, _} = SelfPlayEnv.step(env)
      {:ok, env2, _} = SelfPlayEnv.step(env1)
      {:ok, env3, _} = SelfPlayEnv.step(env2)

      assert env3.frame_count == 3
    end
  end

  describe "collect_steps/2" do
    setup do
      policy = mock_policy()

      {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: :cpu,
        game_type: :mock
      )

      %{env: env}
    end

    test "collects N steps", %{env: env} do
      {:ok, _new_env, experiences} = SelfPlayEnv.collect_steps(env, 10)

      assert length(experiences) == 10
      assert Enum.all?(experiences, &is_map/1)
    end

    test "all experiences have required keys", %{env: env} do
      {:ok, _new_env, experiences} = SelfPlayEnv.collect_steps(env, 5)

      required_keys = [:state, :action, :reward, :done, :log_prob, :value]

      for exp <- experiences do
        for key <- required_keys do
          assert Map.has_key?(exp, key), "Missing key: #{key}"
        end
      end
    end
  end

  describe "reset/1" do
    setup do
      policy = mock_policy()

      {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: :cpu,
        game_type: :mock
      )

      # Run some steps
      {:ok, stepped_env, _} = SelfPlayEnv.collect_steps(env, 10)

      %{env: stepped_env}
    end

    test "resets frame count to zero", %{env: env} do
      assert env.frame_count == 10

      {:ok, reset_env} = SelfPlayEnv.reset(env)

      assert reset_env.frame_count == 0
    end

    test "increments episode count", %{env: env} do
      initial_episodes = env.episode_count

      {:ok, reset_env} = SelfPlayEnv.reset(env)

      assert reset_env.episode_count == initial_episodes + 1
    end
  end

  describe "update_p1_policy/2" do
    test "updates p1 policy" do
      policy1 = mock_policy()
      policy2 = mock_policy()

      {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy1,
        p2_policy: :cpu
      )

      updated_env = SelfPlayEnv.update_p1_policy(env, policy2)

      # Policy should be recompiled
      assert is_tuple(updated_env.p1_policy)
      assert tuple_size(updated_env.p1_policy) == 3
    end
  end

  describe "update_p2_policy/2" do
    test "updates p2 to cpu" do
      policy = mock_policy()

      {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: policy
      )

      updated_env = SelfPlayEnv.update_p2_policy(env, :cpu)

      assert updated_env.p2_policy == :cpu
    end

    test "updates p2 to new policy" do
      policy1 = mock_policy()
      policy2 = mock_policy()

      {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy1,
        p2_policy: :cpu
      )

      updated_env = SelfPlayEnv.update_p2_policy(env, policy2)

      assert is_tuple(updated_env.p2_policy)
      assert tuple_size(updated_env.p2_policy) == 3
    end
  end

  describe "shutdown/1" do
    test "shuts down mock environment" do
      policy = mock_policy()

      {:ok, env} = SelfPlayEnv.new(
        p1_policy: policy,
        p2_policy: :cpu,
        game_type: :mock
      )

      assert :ok = SelfPlayEnv.shutdown(env)
    end
  end

  describe "dolphin configuration" do
    test "requires dolphin_config when game_type is :dolphin" do
      policy = mock_policy()

      # This will fail at runtime when trying to connect, but config is accepted
      assert_raise KeyError, fn ->
        SelfPlayEnv.new(
          p1_policy: policy,
          p2_policy: :cpu,
          game_type: :dolphin
          # Missing dolphin_config
        )
      end
    end
  end
end
