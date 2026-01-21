defmodule ExPhil.SelfPlay.GameRunnerTest do
  use ExUnit.Case, async: false

  alias ExPhil.SelfPlay.GameRunner

  @moduletag :self_play

  setup do
    # Use start_supervised so ExUnit manages the Registry lifecycle
    {:ok, _} = start_supervised({Registry, keys: :unique, name: ExPhil.SelfPlay.GameRegistry})
    :ok
  end

  describe "start_link/1" do
    test "starts a game runner with required options" do
      game_id = "test_game_#{System.unique_integer([:positive])}"

      opts = [
        game_id: game_id,
        p1_policy_id: :cpu,
        p2_policy_id: :cpu,
        game_type: :mock
      ]

      assert {:ok, pid} = GameRunner.start_link(opts)
      assert Process.alive?(pid)

      # Cleanup
      GenServer.stop(pid)
    end

    test "fails without required game_id" do
      assert_raise KeyError, fn ->
        GameRunner.start_link(p1_policy_id: :cpu, p2_policy_id: :cpu)
      end
    end
  end

  describe "get_status/1" do
    setup do
      game_id = "status_test_#{System.unique_integer([:positive])}"

      {:ok, pid} = GameRunner.start_link(
        game_id: game_id,
        p1_policy_id: :cpu,
        p2_policy_id: :cpu,
        game_type: :mock
      )

      on_exit(fn -> safe_stop(pid) end)

      %{runner: pid, game_id: game_id}
    end

    test "returns waiting status before game starts", %{runner: runner, game_id: game_id} do
      status = GameRunner.get_status(runner)

      assert status.status == :waiting
      assert status.game_id == game_id
      assert status.frame_count == 0
      assert status.episode_count == 0
    end
  end

  describe "start_game/1" do
    setup do
      game_id = "start_test_#{System.unique_integer([:positive])}"

      {:ok, pid} = GameRunner.start_link(
        game_id: game_id,
        p1_policy_id: :cpu,
        p2_policy_id: :cpu,
        game_type: :mock
      )

      on_exit(fn -> safe_stop(pid) end)

      %{runner: pid}
    end

    test "transitions to playing status", %{runner: runner} do
      assert :ok = GameRunner.start_game(runner)

      status = GameRunner.get_status(runner)
      assert status.status == :playing
    end
  end

  describe "step/1" do
    setup do
      game_id = "step_test_#{System.unique_integer([:positive])}"

      {:ok, pid} = GameRunner.start_link(
        game_id: game_id,
        p1_policy_id: :cpu,
        p2_policy_id: :cpu,
        game_type: :mock
      )

      :ok = GameRunner.start_game(pid)

      on_exit(fn -> safe_stop(pid) end)

      %{runner: pid}
    end

    test "returns experience on step", %{runner: runner} do
      assert {:ok, experience} = GameRunner.step(runner)

      assert is_map(experience)
      assert Map.has_key?(experience, :state)
      assert Map.has_key?(experience, :action)
      assert Map.has_key?(experience, :reward)
      assert Map.has_key?(experience, :done)
    end

    test "increments frame count", %{runner: runner} do
      initial = GameRunner.get_status(runner).frame_count

      {:ok, _} = GameRunner.step(runner)
      {:ok, _} = GameRunner.step(runner)

      final = GameRunner.get_status(runner).frame_count
      assert final == initial + 2
    end
  end

  describe "collect_steps/2" do
    setup do
      game_id = "collect_test_#{System.unique_integer([:positive])}"

      {:ok, pid} = GameRunner.start_link(
        game_id: game_id,
        p1_policy_id: :cpu,
        p2_policy_id: :cpu,
        game_type: :mock
      )

      :ok = GameRunner.start_game(pid)

      on_exit(fn -> safe_stop(pid) end)

      %{runner: pid}
    end

    test "collects N steps of experience", %{runner: runner} do
      n = 10

      assert {:ok, experiences} = GameRunner.collect_steps(runner, n)
      assert length(experiences) == n
      assert Enum.all?(experiences, &is_map/1)
    end
  end

  describe "reset/1" do
    setup do
      game_id = "reset_test_#{System.unique_integer([:positive])}"

      {:ok, pid} = GameRunner.start_link(
        game_id: game_id,
        p1_policy_id: :cpu,
        p2_policy_id: :cpu,
        game_type: :mock
      )

      :ok = GameRunner.start_game(pid)

      # Run some steps
      for _ <- 1..10, do: GameRunner.step(pid)

      on_exit(fn -> safe_stop(pid) end)

      %{runner: pid}
    end

    test "resets frame count", %{runner: runner} do
      assert GameRunner.get_status(runner).frame_count > 0

      assert :ok = GameRunner.reset(runner)

      assert GameRunner.get_status(runner).frame_count == 0
      assert GameRunner.get_status(runner).status == :playing
    end
  end

  # Helper functions

  defp safe_stop(pid) do
    if Process.alive?(pid) do
      GenServer.stop(pid, :normal, 1000)
    end
  rescue
    _ -> :ok
  end
end
