defmodule ExPhil.SelfPlay.MatchmakerTest do
  use ExUnit.Case, async: true

  alias ExPhil.SelfPlay.Matchmaker

  @moduletag :self_play

  describe "start_link/1" do
    test "starts with default options" do
      assert {:ok, pid} = Matchmaker.start_link(name: nil)
      assert Process.alive?(pid)

      stats = Matchmaker.get_stats(pid)
      assert stats.total_matches == 0
      assert stats.num_policies == 0

      GenServer.stop(pid)
    end

    test "starts with custom k_factor" do
      assert {:ok, pid} = Matchmaker.start_link(name: nil, k_factor: 40)
      GenServer.stop(pid)
    end
  end

  describe "register_policy/3" do
    setup do
      {:ok, pid} = Matchmaker.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{matchmaker: pid}
    end

    test "registers new policy with default rating", %{matchmaker: mm} do
      assert :ok = Matchmaker.register_policy(mm, "policy_1")

      {:ok, info} = Matchmaker.get_rating(mm, "policy_1")
      assert info.rating == 1000
      assert info.games_played == 0
    end

    test "registers with custom initial rating", %{matchmaker: mm} do
      assert :ok = Matchmaker.register_policy(mm, "policy_1", 1200)

      {:ok, info} = Matchmaker.get_rating(mm, "policy_1")
      assert info.rating == 1200
    end

    test "rejects duplicate registration", %{matchmaker: mm} do
      Matchmaker.register_policy(mm, "policy_1")
      assert {:error, :already_registered} = Matchmaker.register_policy(mm, "policy_1")
    end
  end

  describe "request_match/3" do
    setup do
      {:ok, pid} = Matchmaker.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{matchmaker: pid}
    end

    test "returns match with opponent", %{matchmaker: mm} do
      {:ok, match} = Matchmaker.request_match(mm, "game_1")

      assert match.game_id == "game_1"
      assert match.p1 != nil
      assert match.p2 != nil
      assert match.p1_rating != nil
      assert match.p2_rating != nil
    end

    test "auto-registers unknown players", %{matchmaker: mm} do
      Matchmaker.request_match(mm, "game_1", player_id: "new_player")

      {:ok, info} = Matchmaker.get_rating(mm, "new_player")
      assert info.rating == 1000
    end
  end

  describe "report_result/4" do
    setup do
      {:ok, pid} = Matchmaker.start_link(name: nil, k_factor: 32)
      on_exit(fn -> safe_stop(pid) end)

      # Register two policies
      Matchmaker.register_policy(pid, "policy_a", 1000)
      Matchmaker.register_policy(pid, "policy_b", 1000)

      %{matchmaker: pid}
    end

    test "updates ratings on win", %{matchmaker: mm} do
      :ok = Matchmaker.report_result(mm, "policy_a", "policy_b", :win)

      {:ok, a_info} = Matchmaker.get_rating(mm, "policy_a")
      {:ok, b_info} = Matchmaker.get_rating(mm, "policy_b")

      # Winner gains, loser loses
      assert a_info.rating > 1000
      assert b_info.rating < 1000
      assert a_info.wins == 1
      assert b_info.losses == 1
    end

    test "updates ratings on loss", %{matchmaker: mm} do
      :ok = Matchmaker.report_result(mm, "policy_a", "policy_b", :loss)

      {:ok, a_info} = Matchmaker.get_rating(mm, "policy_a")
      {:ok, b_info} = Matchmaker.get_rating(mm, "policy_b")

      # Loser loses, winner gains
      assert a_info.rating < 1000
      assert b_info.rating > 1000
    end

    test "handles draw", %{matchmaker: mm} do
      :ok = Matchmaker.report_result(mm, "policy_a", "policy_b", :draw)

      {:ok, a_info} = Matchmaker.get_rating(mm, "policy_a")
      {:ok, b_info} = Matchmaker.get_rating(mm, "policy_b")

      # Equal ratings, draw should not change much
      assert_in_delta a_info.rating, 1000, 1
      assert_in_delta b_info.rating, 1000, 1
      assert a_info.draws == 1
      assert b_info.draws == 1
    end

    test "increments games played", %{matchmaker: mm} do
      :ok = Matchmaker.report_result(mm, "policy_a", "policy_b", :win)

      {:ok, a_info} = Matchmaker.get_rating(mm, "policy_a")
      {:ok, b_info} = Matchmaker.get_rating(mm, "policy_b")

      assert a_info.games_played == 1
      assert b_info.games_played == 1
    end
  end

  describe "get_leaderboard/2" do
    setup do
      {:ok, pid} = Matchmaker.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)

      # Register policies with different ratings
      Matchmaker.register_policy(pid, "top", 1500)
      Matchmaker.register_policy(pid, "mid", 1200)
      Matchmaker.register_policy(pid, "low", 900)

      %{matchmaker: pid}
    end

    test "returns sorted leaderboard", %{matchmaker: mm} do
      leaderboard = Matchmaker.get_leaderboard(mm, 10)

      assert length(leaderboard) == 3
      assert hd(leaderboard).id == "top"
      assert List.last(leaderboard).id == "low"
    end

    test "respects limit", %{matchmaker: mm} do
      leaderboard = Matchmaker.get_leaderboard(mm, 2)

      assert length(leaderboard) == 2
    end
  end

  describe "get_win_rate/3" do
    setup do
      {:ok, pid} = Matchmaker.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)

      Matchmaker.register_policy(pid, "policy_a")
      Matchmaker.register_policy(pid, "policy_b")

      # Record some matches
      Matchmaker.report_result(pid, "policy_a", "policy_b", :win)
      Matchmaker.report_result(pid, "policy_a", "policy_b", :win)
      Matchmaker.report_result(pid, "policy_a", "policy_b", :loss)

      %{matchmaker: pid}
    end

    test "calculates correct win rate", %{matchmaker: mm} do
      {:ok, rate} = Matchmaker.get_win_rate(mm, "policy_a", "policy_b")

      # 2 wins out of 3 games
      assert_in_delta rate, 2/3, 0.01
    end

    test "returns 0.5 for unknown matchup", %{matchmaker: mm} do
      {:ok, rate} = Matchmaker.get_win_rate(mm, "policy_a", "unknown")
      assert rate == 0.5
    end
  end

  describe "get_match_history/3" do
    setup do
      {:ok, pid} = Matchmaker.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)

      Matchmaker.register_policy(pid, "policy_a")
      Matchmaker.register_policy(pid, "policy_b")
      Matchmaker.register_policy(pid, "policy_c")

      # Record matches
      Matchmaker.report_result(pid, "policy_a", "policy_b", :win)
      Matchmaker.report_result(pid, "policy_a", "policy_c", :loss)

      %{matchmaker: pid}
    end

    test "returns match history for policy", %{matchmaker: mm} do
      history = Matchmaker.get_match_history(mm, "policy_a")

      assert length(history) == 2
    end

    test "respects limit", %{matchmaker: mm} do
      history = Matchmaker.get_match_history(mm, "policy_a", 1)

      assert length(history) == 1
    end
  end

  describe "reset/1" do
    setup do
      {:ok, pid} = Matchmaker.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)

      Matchmaker.register_policy(pid, "policy_a")
      Matchmaker.report_result(pid, "policy_a", "policy_b", :win)

      %{matchmaker: pid}
    end

    test "clears all data", %{matchmaker: mm} do
      :ok = Matchmaker.reset(mm)

      stats = Matchmaker.get_stats(mm)
      assert stats.num_policies == 0
      assert stats.total_matches == 0
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
