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

      # Collect experience (collect_steps collects N steps from EACH game)
      # With 2 games and 50 steps each, we get 100 total
      experiences = Supervisor.collect_steps(50)
      assert length(experiences) == 100

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

  # Game embedding is ~1991 dims (2 players @ 446 each + stage + projectiles)
  @game_embed_size 1991
  @hidden_size 64

  # Policy output sizes match the autoregressive controller head
  @num_buttons 8
  @axis_size 17      # 0-16 = 17 values
  @shoulder_size 5   # 0-4 = 5 values

  defp mock_model do
    # Build a minimal model with correct input/output dimensions
    # Input: game embedding [batch, 1991]
    # Output: {policy_logits, value} where policy_logits is tuple of 6 heads
    input = Axon.input("state", shape: {nil, @game_embed_size})

    # Simple backbone
    backbone = input
    |> Axon.dense(@hidden_size, name: "backbone_dense")
    |> Axon.relu()

    # Policy heads (independent in mock - real model is autoregressive)
    buttons = Axon.dense(backbone, @num_buttons, name: "buttons_logits")
    main_x = Axon.dense(backbone, @axis_size, name: "main_x_logits")
    main_y = Axon.dense(backbone, @axis_size, name: "main_y_logits")
    c_x = Axon.dense(backbone, @axis_size, name: "c_x_logits")
    c_y = Axon.dense(backbone, @axis_size, name: "c_y_logits")
    shoulder = Axon.dense(backbone, @shoulder_size, name: "shoulder_logits")

    # Value head
    value = Axon.dense(backbone, 1, name: "value_head")

    # Combine into expected output format
    Axon.container({
      {buttons, main_x, main_y, c_x, c_y, shoulder},
      value
    })
  end

  defp mock_params do
    # Initialize with small random-ish values
    %{
      "backbone_dense" => %{
        "kernel" => Nx.broadcast(0.01, {@game_embed_size, @hidden_size}),
        "bias" => Nx.broadcast(0.0, {@hidden_size})
      },
      "buttons_logits" => %{
        "kernel" => Nx.broadcast(0.01, {@hidden_size, @num_buttons}),
        "bias" => Nx.broadcast(0.0, {@num_buttons})
      },
      "main_x_logits" => %{
        "kernel" => Nx.broadcast(0.01, {@hidden_size, @axis_size}),
        "bias" => Nx.broadcast(0.0, {@axis_size})
      },
      "main_y_logits" => %{
        "kernel" => Nx.broadcast(0.01, {@hidden_size, @axis_size}),
        "bias" => Nx.broadcast(0.0, {@axis_size})
      },
      "c_x_logits" => %{
        "kernel" => Nx.broadcast(0.01, {@hidden_size, @axis_size}),
        "bias" => Nx.broadcast(0.0, {@axis_size})
      },
      "c_y_logits" => %{
        "kernel" => Nx.broadcast(0.01, {@hidden_size, @axis_size}),
        "bias" => Nx.broadcast(0.0, {@axis_size})
      },
      "shoulder_logits" => %{
        "kernel" => Nx.broadcast(0.01, {@hidden_size, @shoulder_size}),
        "bias" => Nx.broadcast(0.0, {@shoulder_size})
      },
      "value_head" => %{
        "kernel" => Nx.broadcast(0.01, {@hidden_size, 1}),
        "bias" => Nx.broadcast(0.0, {1})
      }
    }
  end

  defp mock_experience do
    %{
      state: Nx.tensor(Enum.map(1..@game_embed_size, fn _ -> :rand.uniform() end)),
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
