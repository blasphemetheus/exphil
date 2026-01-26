defmodule ExPhil.SelfPlay.EloTest do
  use ExUnit.Case, async: true

  alias ExPhil.SelfPlay.Elo

  describe "expected_score/2" do
    test "equal ratings give 0.5 expected score" do
      assert Elo.expected_score(1500, 1500) == 0.5
    end

    test "higher rated player has higher expected score" do
      higher = Elo.expected_score(1600, 1400)
      lower = Elo.expected_score(1400, 1600)

      assert higher > 0.5
      assert lower < 0.5
      assert_in_delta higher + lower, 1.0, 0.001
    end

    test "400 point advantage gives ~0.91 expected score" do
      score = Elo.expected_score(1600, 1200)
      assert_in_delta score, 0.91, 0.01
    end

    test "200 point advantage gives ~0.76 expected score" do
      score = Elo.expected_score(1600, 1400)
      assert_in_delta score, 0.76, 0.01
    end
  end

  describe "update/4" do
    test "winner gains rating, loser loses rating" do
      {new_a, new_b} = Elo.update(1500, 1500, :win)

      assert new_a > 1500
      assert new_b < 1500
    end

    test "higher rated player gains less for beating lower rated" do
      {gain_favored, _} = Elo.update(1600, 1400, :win)
      {gain_underdog, _} = Elo.update(1400, 1600, :win)

      favored_gain = gain_favored - 1600
      underdog_gain = gain_underdog - 1400

      assert underdog_gain > favored_gain
    end

    test "draw moves ratings toward each other" do
      {new_a, new_b} = Elo.update(1600, 1400, :draw)

      # Higher rated loses points, lower rated gains
      assert new_a < 1600
      assert new_b > 1400
    end

    test "loss for higher rated causes larger rating drop" do
      {new_a, new_b} = Elo.update(1600, 1400, :loss)

      # Higher rated loses more when upset
      assert new_a < 1600
      assert new_b > 1400

      _drop = 1600 - new_a
      gain = new_b - 1400

      # Upset win is more valuable
      # More than default K/2
      assert gain > 16
    end

    test "custom K-factor affects rating change" do
      {new_a_32, _} = Elo.update(1500, 1500, :win, k_factor: 32)
      {new_a_16, _} = Elo.update(1500, 1500, :win, k_factor: 16)

      change_32 = new_a_32 - 1500
      change_16 = new_a_16 - 1500

      assert_in_delta change_32, change_16 * 2, 0.001
    end

    test "different K-factors for each player" do
      {new_a, new_b} = Elo.update(1500, 1500, :win, k_factor_a: 40, k_factor_b: 16)

      change_a = new_a - 1500
      change_b = 1500 - new_b

      # A gains more with higher K-factor
      assert change_a > change_b
    end

    test "rating changes sum to zero (zero-sum)" do
      {new_a, new_b} = Elo.update(1500, 1600, :win, k_factor: 32)

      change_a = new_a - 1500
      change_b = new_b - 1600

      assert_in_delta change_a + change_b, 0, 0.001
    end
  end

  describe "update_single/4" do
    test "matches update/4 for player A" do
      {expected_a, _} = Elo.update(1500, 1600, :win)
      actual_a = Elo.update_single(1500, 1600, :win)

      assert_in_delta expected_a, actual_a, 0.001
    end

    test "works for losses" do
      result = Elo.update_single(1500, 1400, :loss)
      assert result < 1500
    end

    test "works for draws" do
      # Against lower rated, draw loses points
      result = Elo.update_single(1600, 1400, :draw)
      assert result < 1600

      # Against higher rated, draw gains points
      result2 = Elo.update_single(1400, 1600, :draw)
      assert result2 > 1400
    end
  end

  describe "dynamic_k_factor/1" do
    test "returns 40 for new players" do
      assert Elo.dynamic_k_factor(0) == 40
      assert Elo.dynamic_k_factor(15) == 40
      assert Elo.dynamic_k_factor(29) == 40
    end

    test "returns 32 for intermediate players" do
      assert Elo.dynamic_k_factor(30) == 32
      assert Elo.dynamic_k_factor(50) == 32
      assert Elo.dynamic_k_factor(99) == 32
    end

    test "returns 24 for established players" do
      assert Elo.dynamic_k_factor(100) == 24
      assert Elo.dynamic_k_factor(500) == 24
    end
  end

  describe "initial_rating/0" do
    test "returns default initial rating" do
      assert Elo.initial_rating() == 1000
    end
  end

  describe "default_k_factor/0" do
    test "returns default K-factor" do
      assert Elo.default_k_factor() == 32
    end
  end

  describe "rating_difference_for_probability/1" do
    test "0.5 probability means 0 rating difference" do
      diff = Elo.rating_difference_for_probability(0.5)
      assert_in_delta diff, 0, 0.001
    end

    test "higher probability means positive difference (player is stronger)" do
      diff = Elo.rating_difference_for_probability(0.76)
      assert diff > 0
      assert_in_delta diff, 200, 5
    end

    test "lower probability means negative difference (player is weaker)" do
      diff = Elo.rating_difference_for_probability(0.24)
      assert diff < 0
      assert_in_delta diff, -200, 5
    end

    test "is inverse of expected_score" do
      rating_a = 1600
      rating_b = 1400
      prob = Elo.expected_score(rating_a, rating_b)
      diff = Elo.rating_difference_for_probability(prob)

      assert_in_delta diff, rating_a - rating_b, 0.1
    end
  end

  describe "games_to_reach/4" do
    test "returns 0 if already at target" do
      assert Elo.games_to_reach(1500, 1400, 1400, 0.6) == 0
      assert Elo.games_to_reach(1500, 1500, 1400, 0.6) == 0
    end

    test "estimates games needed to climb with high win rate" do
      # With 80% win rate against lower-rated opponents, climbing is fast
      games = Elo.games_to_reach(1000, 1100, 900, 0.8)

      assert games > 0
      # Should be achievable with 80% win rate against weaker
      assert games < 50
    end

    test "more games needed with lower win rate" do
      games_80 = Elo.games_to_reach(1000, 1100, 900, 0.8)
      games_70 = Elo.games_to_reach(1000, 1100, 900, 0.7)

      assert games_70 > games_80
    end

    test "caps at 10001 to prevent infinite loop" do
      # 50% win rate against same rating = no progress
      games = Elo.games_to_reach(1000, 2000, 1000, 0.5)
      assert games == 10001
    end
  end
end
