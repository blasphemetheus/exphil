defmodule ExPhil.Eval.DecisionMapTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.{ControllerState, Player}
  alias ExPhil.Eval.DecisionMap

  @edge 85.5656967163

  defp ctrl(x, y, buttons \\ []) do
    %ControllerState{main_stick: %{x: x, y: y}, c_stick: %{x: 0.5, y: 0.5}, l_shoulder: 0.0, r_shoulder: 0.0,
      button_a: :a in buttons, button_b: :b in buttons, button_x: :x in buttons, button_y: :y in buttons,
      button_z: false, button_l: false, button_r: false, button_d_up: false}
  end

  defp player(over) do
    Map.merge(
      %Player{character: 2, x: 100.0, y: -30.0, percent: 0.0, stock: 4, facing: -1, action: 29, action_frame: 1,
        invulnerable: false, jumps_left: 1, on_ground: false, shield_strength: 60.0, hitstun_frames_left: 0,
        speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: 0.0, speed_x_attack: 0.0, speed_y_attack: 0.0},
      Map.new(over)
    )
  end

  defp frame(own, c \\ ctrl(0.5, 0.5)), do: %{own: own, opp: player(x: 30.0, y: 0.0, on_ground: true), controller: c}

  describe "eligible?/2" do
    test "free-falling offstage with no hitstun, not helpless, not in a special, not on the ledge" do
      assert DecisionMap.eligible?(player([]), @edge)
      refute DecisionMap.eligible?(player(on_ground: true, x: 50.0, y: 0.0), @edge)
      refute DecisionMap.eligible?(player(x: 50.0, y: 10.0), @edge)
      assert DecisionMap.eligible?(player(x: 50.0, y: -10.0), @edge)
      refute DecisionMap.eligible?(player(hitstun_frames_left: 3), @edge)
      refute DecisionMap.eligible?(player(action: 35), @edge)
      refute DecisionMap.eligible?(player(action: 356), @edge)
      refute DecisionMap.eligible?(player(action: 253), @edge)
    end
  end

  describe "bucket/1" do
    test "height band × jumps left" do
      assert DecisionMap.bucket(player(y: 5.0)) == "y>0:j1+"
      assert DecisionMap.bucket(player(y: -10.0, jumps_left: 0)) == "0..-20:j0"
      assert DecisionMap.bucket(player(y: -30.0)) == "-20..-40:j1+"
      assert DecisionMap.bucket(player(y: -50.0)) == "-40..-60:j1+"
      assert DecisionMap.bucket(player(y: -70.0)) == "<-60:j1+"
    end
  end

  describe "decision/3" do
    test "jump by spent double jump or a jump press with one in hand" do
      assert DecisionMap.decision(player([]), player(jumps_left: 0, action: 27), ctrl(0.5, 0.5)) == "jump"
      assert DecisionMap.decision(player([]), player([]), ctrl(0.5, 0.5, [:x])) == "jump"
      assert DecisionMap.decision(player(jumps_left: 0), player(jumps_left: 0), ctrl(0.5, 0.5, [:y])) == nil
    end

    test "special onset by stick zone, airdodge, aerial" do
      assert DecisionMap.decision(player([]), player(action: 356), ctrl(0.5, 1.0, [:b])) == "special_up"
      assert DecisionMap.decision(player([]), player(action: 365), ctrl(1.0, 0.5, [:b])) == "special_side"
      assert DecisionMap.decision(player([]), player(action: 341), ctrl(0.5, 0.5, [:b])) == "special_neutral"
      assert DecisionMap.decision(player([]), player(action: 236), ctrl(0.5, 0.5)) == "airdodge"
      assert DecisionMap.decision(player([]), player(action: 69), ctrl(0.5, 0.0, [:a])) == "aerial"
      assert DecisionMap.decision(player(action: 69), player(action: 69), ctrl(0.5, 0.0, [:a])) == nil
    end
  end

  describe "from_game/2, summarize/1, slope/2, compare/3" do
    test "counts onsets per bucket and reads the height conditioning" do
      # ten ledge-band frames with one jump, ten low-band frames with five jumps
      shallow = for i <- 0..9, do: frame(player(y: -10.0, jumps_left: 1), ctrl(0.5, 0.5, if(i == 4, do: [:x], else: [])))
      low = for i <- 0..9, do: frame(player(y: -50.0, jumps_left: 1), ctrl(0.5, 0.5, if(rem(i, 2) == 1, do: [:x], else: [])))
      # a terminator so the last frame has a successor
      game = shallow ++ low ++ [frame(player(y: -70.0))]

      counts = DecisionMap.from_game(game, edge: @edge)
      assert counts["0..-20:j1+"]["frames"] == 10
      assert counts["0..-20:j1+"]["jump"] == 1
      assert counts["-40..-60:j1+"]["frames"] == 10
      # presses on the controller at t+1: low frames 1,3,5,7,9; the transition from
      # the last shallow frame into low frame 0 carries no press
      assert counts["-40..-60:j1+"]["jump"] == 5

      s = DecisionMap.summarize(counts)
      assert s["0..-20:j1+"].jump == 0.1
      assert s["-40..-60:j1+"].jump == 0.5
      assert DecisionMap.slope(s, :jump) == 5.0

      expert = DecisionMap.summarize(DecisionMap.merge(counts, counts))
      flat = DecisionMap.summarize(%{"-40..-60:j1+" => %{"frames" => 200, "jump" => 20}})
      [row] = DecisionMap.compare(flat, expert, decision: :jump, min_n: 100)
      assert row.bucket == "-40..-60:j1+"
      assert row.model == 0.1 and row.expert == 0.5
      assert row.z < -5
    end

    test "merge adds counts" do
      a = %{"<-60:j0" => %{"frames" => 3, "jump" => 0, "aerial" => 1}}
      assert DecisionMap.merge(a, a)["<-60:j0"]["frames"] == 6
    end
  end
end
