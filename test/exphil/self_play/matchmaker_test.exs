defmodule ExPhil.SelfPlay.MatchmakerTest do
  use ExUnit.Case, async: true

  alias ExPhil.SelfPlay.Matchmaker
  alias ExPhil.Error.LeagueError

  setup do
    # Start a fresh matchmaker for each test
    {:ok, mm} = Matchmaker.start_link(name: nil)
    %{mm: mm}
  end

  describe "start_link/1" do
    test "starts with custom k_factor" do
      {:ok, mm} = Matchmaker.start_link(k_factor: 40, name: nil)
      stats = Matchmaker.get_stats(mm)
      assert stats.num_policies == 0
    end
  end

  describe "register_policy/3" do
    test "registers a new policy", %{mm: mm} do
      assert :ok = Matchmaker.register_policy(mm, "test_policy")
      {:ok, info} = Matchmaker.get_rating(mm, "test_policy")

      # Default initial
      assert info.rating == 1000
      assert info.wins == 0
      assert info.losses == 0
      assert info.games_played == 0
    end

    test "registers with custom initial rating", %{mm: mm} do
      assert :ok = Matchmaker.register_policy(mm, "strong_policy", 1500)
      {:ok, info} = Matchmaker.get_rating(mm, "strong_policy")

      assert info.rating == 1500
    end

    test "rejects duplicate registration", %{mm: mm} do
      :ok = Matchmaker.register_policy(mm, "policy_1")
      assert {:error, %LeagueError{reason: :already_registered}} = Matchmaker.register_policy(mm, "policy_1")
    end

    test "accepts atom policy ids", %{mm: mm} do
      :ok = Matchmaker.register_policy(mm, :current)
      {:ok, info} = Matchmaker.get_rating(mm, :current)
      assert info.rating == 1000
    end
  end

  describe "request_match/3" do
    test "creates a match and returns match info", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      {:ok, match} = Matchmaker.request_match(mm, "game_1", player_id: "p1")

      assert match.p1 == "p1"
      assert match.p1_rating == 1000
      assert is_binary(match.match_id)
    end

    test "auto-registers player if not registered", %{mm: mm} do
      {:ok, match} = Matchmaker.request_match(mm, "game_1", player_id: "new_player")

      assert match.p1 == "new_player"
      {:ok, info} = Matchmaker.get_rating(mm, "new_player")
      assert info.rating == 1000
    end

    test "self_play strategy matches player against itself", %{mm: mm} do
      Matchmaker.register_policy(mm, :current)

      {:ok, match} =
        Matchmaker.request_match(mm, "game_1",
          player_id: :current,
          strategy: :self_play
        )

      # p1/p2 keep original type, ratings use normalized string key
      assert match.p1 == :current
      assert match.p2 == :current
    end

    test "historical strategy picks from versioned policies", %{mm: mm} do
      Matchmaker.register_policy(mm, :current)
      Matchmaker.register_policy(mm, "v1")
      Matchmaker.register_policy(mm, "v2")

      {:ok, match} =
        Matchmaker.request_match(mm, "game_1",
          player_id: :current,
          strategy: :historical
        )

      assert match.p1 == :current
      assert match.p2 in ["v1", "v2"]
    end

    test "historical falls back to self_play when no historical", %{mm: mm} do
      Matchmaker.register_policy(mm, :current)

      {:ok, match} =
        Matchmaker.request_match(mm, "game_1",
          player_id: :current,
          strategy: :historical
        )

      assert match.p2 == :current
    end
  end

  describe "report_result/4" do
    test "updates ratings after win", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      :ok = Matchmaker.report_result(mm, "p1", "p2", :win)

      {:ok, p1_info} = Matchmaker.get_rating(mm, "p1")
      {:ok, p2_info} = Matchmaker.get_rating(mm, "p2")

      assert p1_info.rating > 1000
      assert p2_info.rating < 1000
      assert p1_info.wins == 1
      assert p2_info.losses == 1
    end

    test "updates ratings after loss", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      :ok = Matchmaker.report_result(mm, "p1", "p2", :loss)

      {:ok, p1_info} = Matchmaker.get_rating(mm, "p1")
      {:ok, p2_info} = Matchmaker.get_rating(mm, "p2")

      assert p1_info.rating < 1000
      assert p2_info.rating > 1000
      assert p1_info.losses == 1
      assert p2_info.wins == 1
    end

    test "updates ratings after draw", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1", 1100)
      Matchmaker.register_policy(mm, "p2", 900)

      :ok = Matchmaker.report_result(mm, "p1", "p2", :draw)

      {:ok, p1_info} = Matchmaker.get_rating(mm, "p1")
      {:ok, p2_info} = Matchmaker.get_rating(mm, "p2")

      # Draw favors lower rated
      assert p1_info.rating < 1100
      assert p2_info.rating > 900
      assert p1_info.draws == 1
      assert p2_info.draws == 1
    end

    test "auto-registers unknown policies", %{mm: mm} do
      :ok = Matchmaker.report_result(mm, "unknown1", "unknown2", :win)

      {:ok, u1} = Matchmaker.get_rating(mm, "unknown1")
      {:ok, u2} = Matchmaker.get_rating(mm, "unknown2")

      assert u1.games_played == 1
      assert u2.games_played == 1
    end

    test "increments games_played", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      Matchmaker.report_result(mm, "p1", "p2", :win)
      Matchmaker.report_result(mm, "p1", "p2", :loss)
      Matchmaker.report_result(mm, "p1", "p2", :draw)

      {:ok, p1_info} = Matchmaker.get_rating(mm, "p1")
      assert p1_info.games_played == 3
    end
  end

  describe "get_ratings/1" do
    test "returns all ratings", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1", 1100)
      Matchmaker.register_policy(mm, "p2", 1200)

      ratings = Matchmaker.get_ratings(mm)

      assert map_size(ratings) == 2
      assert ratings["p1"].rating == 1100
      assert ratings["p2"].rating == 1200
    end
  end

  describe "get_leaderboard/2" do
    test "returns policies sorted by rating", %{mm: mm} do
      Matchmaker.register_policy(mm, "low", 900)
      Matchmaker.register_policy(mm, "mid", 1000)
      Matchmaker.register_policy(mm, "high", 1100)

      leaderboard = Matchmaker.get_leaderboard(mm, 10)

      assert length(leaderboard) == 3
      assert hd(leaderboard).id == "high"
      assert List.last(leaderboard).id == "low"
    end

    test "respects limit", %{mm: mm} do
      for i <- 1..5, do: Matchmaker.register_policy(mm, "p#{i}", 1000 + i)

      leaderboard = Matchmaker.get_leaderboard(mm, 3)
      assert length(leaderboard) == 3
    end
  end

  describe "get_stats/1" do
    test "returns matchmaker statistics", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")
      Matchmaker.report_result(mm, "p1", "p2", :win)

      stats = Matchmaker.get_stats(mm)

      assert stats.num_policies == 2
      assert stats.total_matches == 1
      assert stats.p1_wins == 1
    end
  end

  describe "get_win_rate/3" do
    test "returns win rate between two policies", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      # P1 wins 3, loses 1
      Matchmaker.report_result(mm, "p1", "p2", :win)
      Matchmaker.report_result(mm, "p1", "p2", :win)
      Matchmaker.report_result(mm, "p1", "p2", :win)
      Matchmaker.report_result(mm, "p1", "p2", :loss)

      {:ok, rate} = Matchmaker.get_win_rate(mm, "p1", "p2")
      assert_in_delta rate, 0.75, 0.01
    end

    test "returns 0.5 for unknown matchup", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      {:ok, rate} = Matchmaker.get_win_rate(mm, "p1", "p2")
      assert rate == 0.5
    end
  end

  describe "get_match_history/3" do
    test "returns match history for a policy", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      Matchmaker.report_result(mm, "p1", "p2", :win)
      Matchmaker.report_result(mm, "p1", "p2", :loss)

      history = Matchmaker.get_match_history(mm, "p1", 10)

      assert length(history) == 2
      # Most recent first
      assert hd(history).result == :loss
    end

    test "respects limit", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")

      for _ <- 1..10, do: Matchmaker.report_result(mm, "p1", "p2", :win)

      history = Matchmaker.get_match_history(mm, "p1", 3)
      assert length(history) == 3
    end
  end

  describe "reset/1" do
    test "clears all state", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "p2")
      Matchmaker.report_result(mm, "p1", "p2", :win)

      :ok = Matchmaker.reset(mm)

      ratings = Matchmaker.get_ratings(mm)
      assert map_size(ratings) == 0

      stats = Matchmaker.get_stats(mm)
      assert stats.total_matches == 0
    end
  end

  describe "skill_based matching" do
    test "matches players within elo range", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1", 1000)
      # Within default 100 range
      Matchmaker.register_policy(mm, "close", 1050)
      # Outside range
      Matchmaker.register_policy(mm, "far", 1500)

      # Run multiple times to check consistency
      opponents =
        for _ <- 1..10 do
          {:ok, match} =
            Matchmaker.request_match(mm, "game",
              player_id: "p1",
              strategy: :skill_based
            )

          match.p2
        end

      # Should prefer close opponent
      close_count = Enum.count(opponents, &(&1 == "close"))
      assert close_count > 0
    end
  end

  describe "exploiter matching" do
    test "prioritizes opponents with low win rate", %{mm: mm} do
      Matchmaker.register_policy(mm, "p1")
      Matchmaker.register_policy(mm, "easy")
      Matchmaker.register_policy(mm, "hard")

      # P1 beats easy, loses to hard
      for _ <- 1..5 do
        Matchmaker.report_result(mm, "p1", "easy", :win)
        Matchmaker.report_result(mm, "p1", "hard", :loss)
      end

      # Exploiter should prefer hard opponents
      opponents =
        for _ <- 1..20 do
          {:ok, match} =
            Matchmaker.request_match(mm, "game",
              player_id: "p1",
              strategy: :exploiter
            )

          match.p2
        end

      hard_count = Enum.count(opponents, &(&1 == "hard"))

      # Should prefer hard opponent more often (low win rate = high priority)
      assert hard_count > 5
    end
  end
end
