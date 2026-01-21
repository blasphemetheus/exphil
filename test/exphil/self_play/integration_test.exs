defmodule ExPhil.SelfPlay.IntegrationTest do
  use ExUnit.Case

  alias ExPhil.SelfPlay.{
    Supervisor,
    PopulationManager,
    ExperienceCollector,
    GamePoolSupervisor,
    Matchmaker
  }

  @moduletag :self_play
  @moduletag :integration
  @moduletag timeout: 60_000

  describe "full self-play system" do
    setup do
      # Start the supervisor with unique names to avoid conflicts
      suffix = System.unique_integer([:positive])

      {:ok, sup} = start_supervised({Supervisor, [
        name: :"test_sup_#{suffix}",
        start_matchmaker: true
      ]})

      %{supervisor: sup}
    end

    test "supervisor starts all children" do
      # Verify all components started
      assert Supervisor.game_pool() != nil
      assert Supervisor.population_manager() != nil
      assert Supervisor.experience_collector() != nil
      assert Supervisor.matchmaker() != nil
    end

    test "can set and get policy" do
      model = mock_model()
      params = mock_params()

      assert :ok = Supervisor.set_policy(model, params)
      assert {:ok, {_model, _params}} = Supervisor.get_policy()
    end

    test "can start and stop games" do
      # Set policy first
      Supervisor.set_policy(mock_model(), mock_params())

      # Start a game
      {:ok, game_id} = Supervisor.start_game(game_type: :mock)

      games = Supervisor.list_games()
      assert length(games) == 1
      assert hd(games).id == game_id

      # Stop the game
      :ok = Supervisor.stop_game(game_id)

      games = Supervisor.list_games()
      assert length(games) == 0
    end

    test "can start multiple games" do
      Supervisor.set_policy(mock_model(), mock_params())

      results = Supervisor.start_games(3, game_type: :mock)

      # All should succeed
      assert Enum.all?(results, fn {:ok, _} -> true; _ -> false end)

      counts = Supervisor.count_games()
      assert counts.total == 3

      # Cleanup
      Supervisor.stop_all_games()
    end

    test "snapshot adds policy to history" do
      Supervisor.set_policy(mock_model(), mock_params())

      stats_before = PopulationManager.get_stats(Supervisor.population_manager())
      assert stats_before.history_size == 0

      Supervisor.snapshot_policy()

      stats_after = PopulationManager.get_stats(Supervisor.population_manager())
      assert stats_after.history_size == 1
    end

    test "games collect experience" do
      Supervisor.set_policy(mock_model(), mock_params())

      # Start a game
      {:ok, _game_id} = Supervisor.start_game(game_type: :mock)

      # Collect some steps
      experiences = Supervisor.collect_steps(10)

      assert length(experiences) == 10
      assert Enum.all?(experiences, &is_map/1)

      Supervisor.stop_all_games()
    end

    test "matchmaker tracks game results" do
      mm = Supervisor.matchmaker()

      # Register and report results
      Matchmaker.register_policy(mm, "test_policy_a")
      Matchmaker.register_policy(mm, "test_policy_b")

      Matchmaker.report_result(mm, "test_policy_a", "test_policy_b", :win)

      {:ok, a_info} = Matchmaker.get_rating(mm, "test_policy_a")
      {:ok, b_info} = Matchmaker.get_rating(mm, "test_policy_b")

      assert a_info.wins == 1
      assert b_info.losses == 1
    end

    test "end-to-end training iteration" do
      # Simulate a training iteration:
      # 1. Set policy
      # 2. Start games
      # 3. Collect experience
      # 4. Snapshot policy
      # 5. Update policy
      # 6. Repeat

      Supervisor.set_policy(mock_model(), mock_params())

      # Start games
      Supervisor.start_games(2, game_type: :mock)

      # Collect experience
      experiences = Supervisor.collect_steps(50)
      assert length(experiences) == 50

      # Snapshot current policy
      Supervisor.snapshot_policy()

      # Simulate policy update
      Supervisor.update_policy_params(mock_params())

      # Snapshot again
      Supervisor.snapshot_policy()

      stats = PopulationManager.get_stats(Supervisor.population_manager())
      assert stats.history_size == 2
      assert stats.current_generation == 2

      Supervisor.stop_all_games()
    end

    test "stats aggregation works" do
      Supervisor.set_policy(mock_model(), mock_params())
      Supervisor.start_games(2, game_type: :mock)
      Supervisor.collect_steps(20)

      stats = Supervisor.get_stats()

      assert Map.has_key?(stats, :games)
      assert Map.has_key?(stats, :collector)
      assert Map.has_key?(stats, :population)
      assert Map.has_key?(stats, :matchmaker)

      assert stats.games.total == 2

      Supervisor.stop_all_games()
    end
  end

  describe "component isolation" do
    test "experience collector works standalone" do
      {:ok, collector} = start_supervised({ExperienceCollector, [name: nil, batch_size: 5]})

      for _ <- 1..10 do
        ExperienceCollector.submit(collector, mock_experience())
      end

      :timer.sleep(50)

      {:ok, batch} = ExperienceCollector.get_batch(collector, 5)
      assert Map.has_key?(batch, :states)
    end

    test "population manager works standalone" do
      {:ok, pm} = start_supervised({PopulationManager, [name: nil]})

      PopulationManager.set_current(pm, mock_model(), mock_params())
      PopulationManager.snapshot(pm)

      {:ok, {_policy_id, _policy}} = PopulationManager.sample_opponent(pm)
    end

    test "matchmaker works standalone" do
      {:ok, mm} = start_supervised({Matchmaker, [name: nil]})

      Matchmaker.register_policy(mm, "p1", 1000)
      Matchmaker.register_policy(mm, "p2", 1000)
      Matchmaker.report_result(mm, "p1", "p2", :win)

      leaderboard = Matchmaker.get_leaderboard(mm, 2)
      assert length(leaderboard) == 2
      assert hd(leaderboard).id == "p1"  # Winner should be on top
    end

    test "game pool supervisor works standalone" do
      # Start registry first
      {:ok, _} = start_supervised({Registry, keys: :unique, name: ExPhil.SelfPlay.GameRegistry})
      {:ok, pool} = start_supervised({GamePoolSupervisor, [name: nil]})

      {:ok, game_id} = GamePoolSupervisor.start_game(pool, [
        game_id: "standalone_game",
        p1_policy_id: :cpu,
        p2_policy_id: :cpu,
        game_type: :mock
      ])

      games = GamePoolSupervisor.list_games(pool)
      assert length(games) == 1

      GamePoolSupervisor.stop_game(pool, game_id)
    end
  end

  # Helper functions

  defp mock_model do
    Axon.input("input", shape: {nil, 100})
    |> Axon.dense(64)
    |> Axon.dense(1)
  end

  defp mock_params do
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

  defp mock_experience do
    %{
      state: Nx.tensor(Enum.map(1..100, fn _ -> :rand.uniform() end)),
      action: %{
        buttons: Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
        main_x: Nx.tensor(8),
        main_y: Nx.tensor(8),
        c_x: Nx.tensor(8),
        c_y: Nx.tensor(8),
        shoulder: Nx.tensor(0)
      },
      log_prob: Nx.tensor(-0.5),
      value: Nx.tensor(0.1),
      reward: 0.01,
      done: false
    }
  end
end
