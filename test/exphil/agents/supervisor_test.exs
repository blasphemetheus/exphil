defmodule ExPhil.Agents.SupervisorTest do
  use ExUnit.Case, async: false

  alias ExPhil.Agents.Supervisor
  alias ExPhil.Error.AgentError

  # Supervisor is already started by Application

  describe "start_link/1" do
    test "supervisor is running" do
      assert Process.alive?(Process.whereis(Supervisor))
    end
  end

  describe "start_agent/1" do
    test "starts an agent with name" do
      assert {:ok, pid} = Supervisor.start_agent(name: :test_agent_1)
      assert is_pid(pid)
      assert Process.alive?(pid)

      Supervisor.stop_agent(:test_agent_1)
    end

    test "starts agent with options" do
      opts = [
        name: :test_agent_2,
        frame_delay: 4,
        deterministic: true
      ]

      assert {:ok, pid} = Supervisor.start_agent(opts)
      assert is_pid(pid)

      Supervisor.stop_agent(:test_agent_2)
    end

    test "fails to start duplicate named agent" do
      {:ok, _pid1} = Supervisor.start_agent(name: :dup_test)

      # Second attempt should fail or return existing
      result = Supervisor.start_agent(name: :dup_test)
      assert {:error, {:already_started, _}} = result

      Supervisor.stop_agent(:dup_test)
    end
  end

  describe "stop_agent/1" do
    test "stops a running agent" do
      {:ok, pid} = Supervisor.start_agent(name: :stop_test)
      assert Process.alive?(pid)

      assert :ok = Supervisor.stop_agent(:stop_test)
      refute Process.alive?(pid)
    end

    test "returns error for non-existent agent" do
      assert {:error, %AgentError{reason: :not_found}} = Supervisor.stop_agent(:nonexistent)
    end
  end

  describe "get_agent/1" do
    test "returns pid for existing agent" do
      {:ok, pid} = Supervisor.start_agent(name: :get_test)

      assert {:ok, ^pid} = Supervisor.get_agent(:get_test)

      Supervisor.stop_agent(:get_test)
    end

    test "returns error for non-existent agent" do
      assert {:error, %AgentError{reason: :not_found}} = Supervisor.get_agent(:not_there)
    end
  end

  describe "list_agents/0" do
    test "returns empty list when no agents" do
      assert [] = Supervisor.list_agents()
    end

    test "returns all running agents" do
      {:ok, pid1} = Supervisor.start_agent(name: :list_1)
      {:ok, pid2} = Supervisor.start_agent(name: :list_2)
      {:ok, pid3} = Supervisor.start_agent(name: :list_3)

      agents = Supervisor.list_agents()

      assert length(agents) == 3
      assert {:list_1, pid1} in agents
      assert {:list_2, pid2} in agents
      assert {:list_3, pid3} in agents

      Supervisor.stop_agent(:list_1)
      Supervisor.stop_agent(:list_2)
      Supervisor.stop_agent(:list_3)
    end
  end
end
