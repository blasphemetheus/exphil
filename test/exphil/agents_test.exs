defmodule ExPhil.AgentsTest do
  use ExUnit.Case, async: false

  alias ExPhil.Agents
  alias ExPhil.Bridge.{GameState, Player}

  # Supervisor is already started by Application - no setup needed

  # Helper to create mock game state
  defp mock_game_state do
    player = %Player{
      x: 0.0,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: 1,
      action: 14,
      action_frame: 0,
      shield_strength: 60.0,
      character: 9,
      invulnerable: false,
      hitstun_frames_left: 0,
      jumps_left: 2,
      on_ground: true,
      speed_air_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: 0.0,
      nana: nil,
      controller_state: nil
    }

    opponent = %Player{
      x: 10.0,
      y: 0.0,
      percent: 30.0,
      stock: 4,
      facing: -1,
      action: 14,
      action_frame: 0,
      shield_strength: 60.0,
      character: 2,
      invulnerable: false,
      hitstun_frames_left: 0,
      jumps_left: 2,
      on_ground: true,
      speed_air_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: 0.0,
      nana: nil,
      controller_state: nil
    }

    %GameState{
      frame: 0,
      stage: 2,
      menu_state: 2,
      players: %{1 => player, 2 => opponent},
      projectiles: []
    }
  end

  describe "start_empty/2" do
    test "starts an agent without a policy" do
      assert {:ok, pid} = Agents.start_empty(:test_agent)
      assert is_pid(pid)
      assert {:ok, ^pid} = Agents.get(:test_agent)

      # Cleanup
      Agents.stop(:test_agent)
    end

    test "agent without policy returns error on get_action" do
      {:ok, _pid} = Agents.start_empty(:no_policy_agent)

      game_state = mock_game_state()
      assert {:error, :no_policy_loaded} = Agents.get_action(:no_policy_agent, game_state)

      Agents.stop(:no_policy_agent)
    end
  end

  describe "stop/1" do
    test "stops a running agent" do
      {:ok, pid} = Agents.start_empty(:stop_test)
      assert {:ok, ^pid} = Agents.get(:stop_test)

      assert :ok = Agents.stop(:stop_test)
      # Small wait for Registry to clean up after process termination
      Process.sleep(10)
      assert {:error, :not_found} = Agents.get(:stop_test)
    end

    test "returns error for non-existent agent" do
      assert {:error, :not_found} = Agents.stop(:nonexistent)
    end
  end

  describe "get/1" do
    test "returns pid for existing agent" do
      {:ok, pid} = Agents.start_empty(:get_test)
      assert {:ok, ^pid} = Agents.get(:get_test)
      Agents.stop(:get_test)
    end

    test "returns error for non-existent agent" do
      assert {:error, :not_found} = Agents.get(:not_there)
    end
  end

  describe "list/0" do
    test "returns empty list when no agents" do
      assert [] = Agents.list()
    end

    test "returns list of running agents" do
      {:ok, pid1} = Agents.start_empty(:list_test1)
      {:ok, pid2} = Agents.start_empty(:list_test2)

      agents = Agents.list()
      assert length(agents) == 2
      assert {:list_test1, pid1} in agents
      assert {:list_test2, pid2} in agents

      Agents.stop(:list_test1)
      Agents.stop(:list_test2)
    end
  end
end

defmodule ExPhil.Agents.AgentTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent
  alias ExPhil.Bridge.{GameState, Player}

  # Helper to create mock game state
  defp mock_game_state do
    player = %Player{
      x: 0.0,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: 1,
      action: 14,
      action_frame: 0,
      shield_strength: 60.0,
      character: 9,
      invulnerable: false,
      hitstun_frames_left: 0,
      jumps_left: 2,
      on_ground: true,
      speed_air_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: 0.0,
      nana: nil,
      controller_state: nil
    }

    opponent = %Player{
      x: 10.0,
      y: 0.0,
      percent: 30.0,
      stock: 4,
      facing: -1,
      action: 14,
      action_frame: 0,
      shield_strength: 60.0,
      character: 2,
      invulnerable: false,
      hitstun_frames_left: 0,
      jumps_left: 2,
      on_ground: true,
      speed_air_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: 0.0,
      nana: nil,
      controller_state: nil
    }

    %GameState{
      frame: 0,
      stage: 2,
      menu_state: 2,
      players: %{1 => player, 2 => opponent},
      projectiles: []
    }
  end

  describe "start_link/1" do
    test "starts agent without name" do
      assert {:ok, pid} = Agent.start_link([])
      assert is_pid(pid)
      GenServer.stop(pid)
    end

    test "starts agent with options" do
      opts = [
        frame_delay: 4,
        deterministic: true,
        temperature: 0.5
      ]

      assert {:ok, pid} = Agent.start_link(opts)
      config = Agent.get_config(pid)

      assert config.frame_delay == 4
      assert config.deterministic == true
      assert config.temperature == 0.5
      assert config.has_policy == false

      GenServer.stop(pid)
    end
  end

  describe "get_config/1" do
    test "returns current configuration" do
      {:ok, pid} = Agent.start_link(deterministic: true)

      config = Agent.get_config(pid)
      assert is_map(config)
      assert config.deterministic == true
      assert config.has_policy == false

      GenServer.stop(pid)
    end
  end

  describe "configure/2" do
    test "updates agent settings" do
      {:ok, pid} = Agent.start_link([])

      assert :ok = Agent.configure(pid, deterministic: true, temperature: 0.7)

      config = Agent.get_config(pid)
      assert config.deterministic == true
      assert config.temperature == 0.7

      GenServer.stop(pid)
    end
  end

  describe "get_action/3" do
    test "returns error when no policy loaded" do
      {:ok, pid} = Agent.start_link([])

      game_state = mock_game_state()
      assert {:error, :no_policy_loaded} = Agent.get_action(pid, game_state)

      GenServer.stop(pid)
    end
  end

  describe "get_controller/3" do
    test "returns error when no policy loaded" do
      {:ok, pid} = Agent.start_link([])

      game_state = mock_game_state()
      assert {:error, :no_policy_loaded} = Agent.get_controller(pid, game_state)

      GenServer.stop(pid)
    end
  end

  describe "load_policy/2" do
    test "returns error for invalid policy" do
      {:ok, pid} = Agent.start_link([])

      assert {:error, {:invalid_policy, _}} = Agent.load_policy(pid, %{invalid: true})

      GenServer.stop(pid)
    end

    test "returns error for missing file" do
      {:ok, pid} = Agent.start_link([])

      assert {:error, _} = Agent.load_policy(pid, "/nonexistent/path.axon")

      GenServer.stop(pid)
    end
  end
end
