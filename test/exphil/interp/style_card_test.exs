defmodule ExPhil.Interp.StyleCardTest do
  use ExUnit.Case, async: true

  alias ExPhil.Interp.StyleCard

  @wait 14
  @fair 66
  @missed_tech 183
  @hitstun_lo 75
  @dj_f 27
  @fall 29

  defp neutral_controller do
    %{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      button_a: false,
      button_b: false,
      button_x: false,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false
    }
  end

  defp press(key), do: Map.put(neutral_controller(), key, true)

  defp grounded(x), do: %{x: x, y: 0.0, on_ground: true, stock: 4}
  defp airborne(x, y), do: %{x: x, y: y, on_ground: false, stock: 4}

  describe "reaction_p50/2" do
    test "measures event -> attack-entry delta; 5f punish fails the human floor" do
      n = 200
      p2 = List.duplicate(@wait, 100) ++ List.duplicate(@missed_tech, 30) ++ List.duplicate(@wait, n - 130)
      # P1 attacks 5 frames after the missed-tech entry (index 100)
      p1 = List.duplicate(@wait, 105) ++ List.duplicate(@fair, 8) ++ List.duplicate(@wait, n - 113)

      assert StyleCard.reaction_p50(p1, p2) == 5
    end

    test "nil when the opponent is never vulnerable" do
      p = List.duplicate(@wait, 100)
      assert StyleCard.reaction_p50(p, p) == nil
    end
  end

  describe "press_interval_cv/1" do
    test "a perfect metronome has ~zero CV; jitter raises it" do
      # 20 X presses exactly 10f apart
      metro =
        Enum.flat_map(1..20, fn _ -> [press(:button_x)] ++ List.duplicate(neutral_controller(), 9) end)

      assert StyleCard.press_interval_cv(metro) < 0.05

      # Same count, alternating 5f/15f gaps
      jitter =
        Enum.flat_map(1..10, fn _ ->
          [press(:button_x)] ++ List.duplicate(neutral_controller(), 4) ++
            [press(:button_x)] ++ List.duplicate(neutral_controller(), 14)
        end)

      assert StyleCard.press_interval_cv(jitter) >= 0.2
    end

    test "nil below the evidence floor" do
      few = [press(:button_x)] ++ List.duplicate(neutral_controller(), 50)
      assert StyleCard.press_interval_cv(few) == nil
    end
  end

  describe "apm/2" do
    test "counts button onsets per minute" do
      # 10 presses over 600 frames (10s) -> 60/min
      controllers =
        Enum.flat_map(1..10, fn _ -> [press(:button_a)] ++ List.duplicate(neutral_controller(), 59) end)

      assert StyleCard.apm(controllers, 600) == 60.0
    end
  end

  describe "di_presence_pct/2" do
    test "neutral stick through hitstun = 0%; deflected = 100%" do
      actions = List.duplicate(@wait, 50) ++ List.duplicate(@hitstun_lo, 20) ++ List.duplicate(@wait, 50)

      neutral = List.duplicate(neutral_controller(), length(actions))
      assert StyleCard.di_presence_pct(actions, neutral) == 0.0

      di_c = Map.put(neutral_controller(), :main_stick, %{x: 0.9, y: 0.5})
      with_di = List.duplicate(di_c, length(actions))
      assert StyleCard.di_presence_pct(actions, with_di) == 100.0
    end

    test "nil when never in hitstun" do
      actions = List.duplicate(@wait, 100)
      assert StyleCard.di_presence_pct(actions, List.duplicate(neutral_controller(), 100)) == nil
    end
  end

  describe "recovery_stats/1" do
    test "offstage stretch with a DJ counts as an attempt" do
      players =
        List.duplicate(grounded(0.0), 50) ++
          List.duplicate(airborne(100.0, -20.0), 30) ++ List.duplicate(grounded(0.0), 50)

      actions =
        List.duplicate(@wait, 50) ++
          List.duplicate(@fall, 10) ++ [@dj_f, @dj_f] ++ List.duplicate(@fall, 18) ++
          List.duplicate(@wait, 50)

      s = StyleCard.recovery_stats(%{actions: actions, players: players})
      assert s.situations == 1
      assert s.attempts == 1
      assert s.rate == 1.0
    end

    test "offstage drift with no DJ/special is a non-attempt" do
      players =
        List.duplicate(grounded(0.0), 50) ++
          List.duplicate(airborne(100.0, -20.0), 30) ++ List.duplicate(grounded(0.0), 50)

      actions = List.duplicate(@wait, 50) ++ List.duplicate(@fall, 30) ++ List.duplicate(@wait, 50)

      s = StyleCard.recovery_stats(%{actions: actions, players: players})
      assert s.situations == 1
      assert s.attempts == 0
      assert s.rate == 0.0
    end

    test "no offstage situations -> nil rate (no evidence)" do
      players = List.duplicate(grounded(0.0), 100)
      s = StyleCard.recovery_stats(%{actions: List.duplicate(@wait, 100), players: players})
      assert s.situations == 0
      assert s.rate == nil
    end
  end

  describe "evaluate/2 (card assembly)" do
    test "returns 6 gates; no-evidence metrics pass; char pack resolves" do
      n = 700
      controllers =
        Enum.flat_map(1..10, fn _ -> [press(:button_a)] ++ List.duplicate(neutral_controller(), 69) end)

      data = %{
        p1: %{
          actions: List.duplicate(@wait, n),
          players: List.duplicate(grounded(0.0), n),
          controllers: controllers
        },
        p2: %{actions: List.duplicate(@wait, n), players: List.duplicate(grounded(30.0), n)},
        n: n
      }

      result = StyleCard.evaluate(data, char: :mewtwo)
      assert result.total == 6
      assert length(result.gates) == 6

      by_name = Map.new(result.gates, &{&1.name, &1})
      # nil-evidence gates pass
      assert by_name["reaction p50 (f)"].pass
      assert by_name["DI presence %"].pass
      assert by_name["recovery attempt rate"].pass
      assert by_name["SDs"].pass
      # ~51 inputs/min is below the human band floor -> fails
      refute by_name["inputs/min"].pass
    end
  end
end
