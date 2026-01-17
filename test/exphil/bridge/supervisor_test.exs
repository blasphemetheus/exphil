defmodule ExPhil.Bridge.SupervisorTest do
  use ExUnit.Case, async: false

  alias ExPhil.Bridge.Supervisor

  # Supervisor is already started by Application

  describe "start_link/1" do
    test "supervisor is running" do
      assert Process.alive?(Process.whereis(Supervisor))
    end
  end

  describe "get_bridge/1" do
    test "returns error for non-existent bridge" do
      assert {:error, :not_found} = Supervisor.get_bridge(:nonexistent)
    end
  end

  describe "stop_bridge/1" do
    test "returns error for non-existent bridge" do
      assert {:error, :not_found} = Supervisor.stop_bridge(:nonexistent)
    end
  end

  describe "list_bridges/0" do
    test "returns empty list when no bridges" do
      assert [] = Supervisor.list_bridges()
    end
  end

  describe "count_bridges/0" do
    test "returns 0 when no bridges" do
      assert 0 = Supervisor.count_bridges()
    end
  end

  # Note: Testing start_bridge/1 requires MeleePort which needs Python
  # Those tests are integration tests that should be tagged :integration
end
