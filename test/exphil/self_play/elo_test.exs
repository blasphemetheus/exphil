defmodule ExPhil.SelfPlay.EloTest do
  use ExUnit.Case, async: true

  alias ExPhil.SelfPlay.Elo

  @moduletag :self_play

  describe "expected_score/2" do
    test "returns 0.5 for equal ratings" do
      assert_in_delta Elo.expected_score(1500, 1500), 0.5, 0.001
    end

    test "returns higher score for higher-rated player" do
      score = Elo.expected_score(1600, 1400)
      assert score > 0.5
      assert_in_delta score, 0.76, 0.01
    end

    test "returns lower score for lower-rated player" do
      score = Elo.expected_score(1400, 1600)
      assert score < 0.5
      assert_in_delta score, 0.24, 0.01
    end

    test "scores sum to 1" do
      a = Elo.expected_score(1500, 1600)
      b = Elo.expected_score(1600, 1500)
      assert_in_delta a + b, 1.0, 0.001
    end
  end

  describe "update/4" do
    test "winner gains, loser loses for equal ratings" do
      {new_a, new_b} = Elo.update(1500, 1500, :win)

      assert new_a > 1500
      assert new_b < 1500
    end

    test "rating changes are symmetric for equal initial ratings" do
      {new_a, new_b} = Elo.update(1500, 1500, :win)

      gain = new_a - 1500
      loss = 1500 - new_b
      assert_in_delta gain, loss, 0.001
    end

    test "upset (lower rated wins) causes larger change" do
      # Lower rated player wins
      {new_low, _new_high} = Elo.update(1400, 1600, :win)

      # Standard case (higher rated wins)
      {new_high2, _new_low2} = Elo.update(1600, 1400, :win)

      # Upset should cause larger rating change
      upset_gain = new_low - 1400
      normal_gain = new_high2 - 1600

      assert upset_gain > normal_gain
    end

    test "draw causes minimal change for equal ratings" do
      {new_a, new_b} = Elo.update(1500, 1500, :draw)

      assert_in_delta new_a, 1500, 0.1
      assert_in_delta new_b, 1500, 0.1
    end

    test "respects custom k_factor" do
      {new_a_k16, _} = Elo.update(1500, 1500, :win, k_factor: 16)
      {new_a_k32, _} = Elo.update(1500, 1500, :win, k_factor: 32)

      change_k16 = new_a_k16 - 1500
      change_k32 = new_a_k32 - 1500

      assert_in_delta change_k32, change_k16 * 2, 0.001
    end

    test "supports different k_factors per player" do
      {new_a, new_b} = Elo.update(1500, 1500, :win, k_factor_a: 40, k_factor_b: 16)

      change_a = new_a - 1500
      change_b = 1500 - new_b

      # A should change more than B
      assert change_a > change_b
    end
  end

  describe "update_single/4" do
    test "calculates single player update" do
      new_rating = Elo.update_single(1500, 1500, :win)
      assert new_rating > 1500
    end

    test "matches update/4 for player A" do
      {from_update, _} = Elo.update(1500, 1600, :win)
      from_single = Elo.update_single(1500, 1600, :win)

      assert_in_delta from_update, from_single, 0.001
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
    test "returns default k_factor" do
      assert Elo.default_k_factor() == 32
    end
  end

  describe "rating_difference_for_probability/1" do
    test "returns 0 for 50% probability" do
      diff = Elo.rating_difference_for_probability(0.5)
      assert_in_delta diff, 0, 0.001
    end

    test "returns positive for > 50% probability" do
      diff = Elo.rating_difference_for_probability(0.75)
      assert diff > 0
    end

    test "returns negative for < 50% probability" do
      diff = Elo.rating_difference_for_probability(0.25)
      assert diff < 0
    end

    test "is inverse of expected_score" do
      # If rating diff is 200, expected score should be...
      expected = Elo.expected_score(1700, 1500)

      # Inverse should give us back ~200
      diff = Elo.rating_difference_for_probability(expected)
      assert_in_delta diff, 200, 1
    end
  end

  describe "games_to_reach/4" do
    test "returns 0 if already at or above target" do
      assert Elo.games_to_reach(1500, 1500, 1400, 0.6) == 0
      assert Elo.games_to_reach(1600, 1500, 1400, 0.6) == 0
    end

    test "returns positive for reachable target" do
      games = Elo.games_to_reach(1000, 1100, 1000, 0.6)
      assert games > 0
    end

    test "more games needed for higher target" do
      # Use higher win rate to ensure both complete within iteration cap
      games_low = Elo.games_to_reach(1000, 1050, 1000, 0.7)
      games_high = Elo.games_to_reach(1000, 1100, 1000, 0.7)

      assert games_high > games_low
    end
  end
end
