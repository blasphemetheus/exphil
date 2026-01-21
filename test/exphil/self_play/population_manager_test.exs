defmodule ExPhil.SelfPlay.PopulationManagerTest do
  use ExUnit.Case, async: true

  alias ExPhil.SelfPlay.PopulationManager

  @moduletag :self_play

  describe "start_link/1" do
    test "starts with default options" do
      assert {:ok, pid} = PopulationManager.start_link(name: nil)
      assert Process.alive?(pid)

      stats = PopulationManager.get_stats(pid)
      assert stats.history_size == 0
      assert stats.has_current == false

      GenServer.stop(pid)
    end

    test "starts with custom history size" do
      assert {:ok, pid} = PopulationManager.start_link(name: nil, max_history_size: 5)
      GenServer.stop(pid)
    end
  end

  describe "set_current/3" do
    setup do
      {:ok, pid} = PopulationManager.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{manager: pid}
    end

    test "sets current policy", %{manager: manager} do
      model = mock_model()
      params = mock_params()

      assert :ok = PopulationManager.set_current(manager, model, params)

      {:ok, {_model, _params}} = PopulationManager.get_current(manager)

      stats = PopulationManager.get_stats(manager)
      assert stats.has_current == true
      assert stats.current_generation == 1
    end

    test "increments generation on each set", %{manager: manager} do
      model = mock_model()

      PopulationManager.set_current(manager, model, mock_params())
      PopulationManager.set_current(manager, model, mock_params())
      PopulationManager.set_current(manager, model, mock_params())

      stats = PopulationManager.get_stats(manager)
      assert stats.current_generation == 3
    end
  end

  describe "snapshot/1" do
    setup do
      {:ok, pid} = PopulationManager.start_link(name: nil, max_history_size: 5)
      on_exit(fn -> safe_stop(pid) end)

      # Set a current policy
      PopulationManager.set_current(pid, mock_model(), mock_params())

      %{manager: pid}
    end

    test "adds current to history", %{manager: manager} do
      assert :ok = PopulationManager.snapshot(manager)

      stats = PopulationManager.get_stats(manager)
      assert stats.history_size == 1
      assert stats.total_snapshots == 1
    end

    test "respects max history size", %{manager: manager} do
      # Snapshot more than max
      for _i <- 1..10 do
        PopulationManager.set_current(manager, mock_model(), mock_params())
        PopulationManager.snapshot(manager)
      end

      stats = PopulationManager.get_stats(manager)
      # Should be capped at 5
      assert stats.history_size == 5
    end

    test "fails without current policy" do
      {:ok, pid} = PopulationManager.start_link(name: nil)

      assert {:error, :no_current_policy} = PopulationManager.snapshot(pid)

      GenServer.stop(pid)
    end
  end

  describe "sample_opponent/2" do
    setup do
      {:ok, pid} = PopulationManager.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{manager: pid}
    end

    test "returns cpu when no policies", %{manager: manager} do
      {:ok, {policy_id, policy}} = PopulationManager.sample_opponent(manager)

      assert {:cpu, _level} = policy_id
      assert {:cpu, _} = policy
    end

    test "can sample current when available", %{manager: manager} do
      PopulationManager.set_current(manager, mock_model(), mock_params())

      # Force current-only sampling
      {:ok, {policy_id, _policy}} = PopulationManager.sample_opponent(manager, strategy: :current_only)

      assert policy_id == :current
    end

    test "can sample historical when available", %{manager: manager} do
      PopulationManager.set_current(manager, mock_model(), mock_params())
      PopulationManager.snapshot(manager)

      # Force historical sampling
      {:ok, {policy_id, _policy}} = PopulationManager.sample_opponent(manager, strategy: :historical_only)

      assert {:historical, _version} = policy_id
    end
  end

  describe "get_policy/2" do
    setup do
      {:ok, pid} = PopulationManager.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)

      PopulationManager.set_current(pid, mock_model(), mock_params())
      PopulationManager.snapshot(pid)

      %{manager: pid}
    end

    test "gets current policy", %{manager: manager} do
      {:ok, {model, params}} = PopulationManager.get_policy(manager, :current)

      assert model != nil
      assert params != nil
    end

    test "gets cpu policy", %{manager: manager} do
      {:ok, {:cpu, level}} = PopulationManager.get_policy(manager, :cpu)

      assert level in 5..9
    end

    test "gets historical policy", %{manager: manager} do
      {:ok, {_model, _params}} = PopulationManager.get_policy(manager, {:historical, "v1"})
    end

    test "returns error for unknown policy", %{manager: manager} do
      assert {:error, :not_found} = PopulationManager.get_policy(manager, {:historical, "v999"})
    end
  end

  describe "list_policies/1" do
    setup do
      {:ok, pid} = PopulationManager.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{manager: pid}
    end

    test "lists available policies", %{manager: manager} do
      PopulationManager.set_current(manager, mock_model(), mock_params())
      PopulationManager.snapshot(manager)
      PopulationManager.set_current(manager, mock_model(), mock_params())
      PopulationManager.snapshot(manager)

      policies = PopulationManager.list_policies(manager)

      # Should have: current, 2 historical, 5 CPU levels
      current_count = Enum.count(policies, & &1.type == :current)
      historical_count = Enum.count(policies, & &1.type == :historical)
      cpu_count = Enum.count(policies, & &1.type == :cpu)

      assert current_count == 1
      assert historical_count == 2
      assert cpu_count == 5
    end
  end

  describe "add_policy/4" do
    setup do
      {:ok, pid} = PopulationManager.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{manager: pid}
    end

    test "adds external policy to history", %{manager: manager} do
      assert :ok = PopulationManager.add_policy(manager, "external_v1", mock_model(), mock_params())

      {:ok, {_model, _params}} = PopulationManager.get_policy(manager, "external_v1")

      stats = PopulationManager.get_stats(manager)
      assert stats.history_size == 1
    end
  end

  # Helper functions

  defp mock_model do
    Axon.input("input", shape: {nil, 100})
    |> Axon.dense(64)
    |> Axon.dense(1)
  end

  defp mock_params do
    # Create minimal params structure
    %{
      "dense_0" => %{
        "kernel" => Nx.broadcast(0.1, {100, 64}),
        "bias" => Nx.broadcast(0.0, {64})
      },
      "dense_1" => %{
        "kernel" => Nx.broadcast(0.1, {64, 1}),
        "bias" => Nx.broadcast(0.0, {1})
      }
    }
  end

  defp safe_stop(pid) do
    if Process.alive?(pid) do
      GenServer.stop(pid, :normal, 1000)
    end
  rescue
    _ -> :ok
  end
end
