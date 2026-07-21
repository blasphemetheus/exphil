defmodule ExPhil.Eval.EloTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.Elo

  describe "expected/2" do
    test "equal ratings expect 0.5" do
      assert Elo.expected(1000.0, 1000.0) == 0.5
    end

    test "400 points above expects ~0.909" do
      assert_in_delta Elo.expected(1400.0, 1000.0), 0.909, 0.001
    end

    test "symmetric: expectations sum to 1" do
      assert_in_delta Elo.expected(1234.0, 987.0) + Elo.expected(987.0, 1234.0), 1.0, 1.0e-9
    end
  end

  describe "update/5" do
    test "a win between equals moves both by k/2 in opposite directions" do
      ratings = Elo.update(%{}, "a", "b", 1.0, k: 32.0)
      assert_in_delta ratings["a"], 1016.0, 1.0e-6
      assert_in_delta ratings["b"], 984.0, 1.0e-6
    end

    test "a draw between equals moves nothing" do
      ratings = Elo.update(%{}, "a", "b", 0.5)
      assert_in_delta ratings["a"], 1000.0, 1.0e-6
      assert_in_delta ratings["b"], 1000.0, 1.0e-6
    end

    test "rating is conserved across any update" do
      ratings = Elo.update(%{"a" => 1350.0, "b" => 900.0}, "a", "b", 0.0, k: 24.0)
      assert_in_delta ratings["a"] + ratings["b"], 2250.0, 1.0e-6
    end

    test "an upset moves more than an expected win" do
      expected_win = Elo.update(%{"a" => 1400.0, "b" => 1000.0}, "a", "b", 1.0)
      upset = Elo.update(%{"a" => 1400.0, "b" => 1000.0}, "b", "a", 1.0)

      assert upset["b"] - 1000.0 > expected_win["a"] - 1400.0
    end
  end

  describe "round_robin/2" do
    test "n=4, 2 games/pair -> 12 games with alternating order" do
      games = Elo.round_robin(["a", "b", "c", "d"], 2)
      assert length(games) == 12
      # Each unordered pair appears once in each order
      assert {"a", "b"} in games and {"b", "a"} in games
      assert {"c", "d"} in games and {"d", "c"} in games
    end

    test "single game per pair keeps first-listed first" do
      assert Elo.round_robin(["a", "b"]) == [{"a", "b"}]
    end
  end

  describe "standings/2" do
    test "the dominant player ranks first with the right record" do
      matches = [
        {"champ", "mid", 1.0},
        {"champ", "weak", 1.0},
        {"mid", "weak", 1.0},
        {"weak", "champ", 0.0}
      ]

      [first, second, third] = Elo.standings(matches)

      assert first.player == "champ"
      assert first.wins == 3 and first.losses == 0
      assert second.player == "mid"
      assert third.player == "weak"
      assert first.rating > second.rating and second.rating > third.rating
    end

    test "draws are recorded on both sides" do
      [a, b] = Elo.standings([{"a", "b", 0.5}])
      assert a.draws == 1 and b.draws == 1
      assert a.games == 1 and b.games == 1
    end
  end
end
