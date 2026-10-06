defmodule ExPhil.Eval.SilenceMapTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.{ControllerState, Player}
  alias ExPhil.Eval.SilenceMap

  @edge 85.5656967163

  defp ctrl(x, y, buttons \\ []) do
    %ControllerState{main_stick: %{x: x, y: y}, c_stick: %{x: 0.5, y: 0.5}, l_shoulder: 0.0, r_shoulder: 0.0,
      button_a: :a in buttons, button_b: :b in buttons, button_x: false, button_y: false, button_z: false,
      button_l: false, button_r: false, button_d_up: false}
  end

  defp player(over) do
    Map.merge(
      %Player{character: 2, x: 0.0, y: 0.0, percent: 0.0, stock: 4, facing: 1, action: 14, action_frame: 1,
        invulnerable: false, jumps_left: 1, on_ground: true, shield_strength: 60.0, hitstun_frames_left: 0,
        speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: 0.0, speed_x_attack: 0.0, speed_y_attack: 0.0},
      Map.new(over)
    )
  end

  defp frame(own, c), do: %{own: own, opp: player(x: 30.0), controller: c}

  describe "transition/2" do
    test "classifies the four input transitions" do
      assert SilenceMap.transition(ctrl(1.0, 0.5), ctrl(0.5, 0.5)) == :enter_silence
      assert SilenceMap.transition(ctrl(0.5, 0.5), ctrl(1.0, 0.5)) == :resume
      assert SilenceMap.transition(ctrl(0.5, 0.5), ctrl(0.5, 0.5)) == :stay
      assert SilenceMap.transition(ctrl(1.0, 0.5), ctrl(0.95, 0.55)) == :hold
      assert SilenceMap.transition(ctrl(1.0, 0.5), ctrl(0.5, 1.0)) == :change
      assert SilenceMap.transition(ctrl(1.0, 0.5), ctrl(1.0, 0.5, [:a])) == :change
    end

    test "full-left and the ±pi octant seam are the same zone" do
      assert SilenceMap.transition(ctrl(0.0, 0.5), ctrl(0.0, 0.49)) == :hold
    end
  end

  describe "state_bin/2" do
    test "physical bins" do
      assert SilenceMap.state_bin(player([]), @edge) == "grounded_center"
      assert SilenceMap.state_bin(player(x: -80.0), @edge) == "grounded_edge"
      assert SilenceMap.state_bin(player(on_ground: false, x: 20.0, y: 40.0), @edge) == "airborne_onstage"
      assert SilenceMap.state_bin(player(on_ground: false, x: -100.0, y: 10.0), @edge) == "offstage_high_j1+"
      assert SilenceMap.state_bin(player(on_ground: false, x: -100.0, y: -30.0, jumps_left: 0), @edge) == "offstage_low_j0"
      assert SilenceMap.state_bin(player(on_ground: false, x: -100.0, y: -70.0), @edge) == "offstage_deep_j1+"
      assert SilenceMap.state_bin(player(on_ground: false, x: -90.0, y: -20.0, action: 253), @edge) == "ledge_hang"
      assert SilenceMap.state_bin(player(on_ground: false, hitstun_frames_left: 5), @edge) == "hitstun"
    end
  end

  describe "from_game/2 + summarize/1 + compare/3" do
    test "counts active frames and the enter-silence hazard per state bin" do
      off = player(on_ground: false, x: -100.0, y: -30.0, jumps_left: 0)
      # held, held, released, silent, silent, resumed
      ctrls = [ctrl(1.0, 0.5), ctrl(1.0, 0.5), ctrl(1.0, 0.5), ctrl(0.5, 0.5), ctrl(0.5, 0.5), ctrl(1.0, 0.5)]
      game = Enum.map(ctrls, &frame(off, &1))
      counts = SilenceMap.from_game(game, edge: @edge)
      c = counts[{"state", "offstage_low_j0"}]
      assert c["frames"] == 5
      assert c["active"] == 3 and c["enter_silence"] == 1 and c["change"] == 0
      assert c["silent"] == 2 and c["resume"] == 1

      s = SilenceMap.summarize(counts)
      assert s["state:offstage_low_j0"].enter_silence == 0.3333
      assert s["state:offstage_low_j0"].resume == 0.5
      # situation labels ride along (offstage is a Situations label)
      assert Map.has_key?(s, "situation:offstage")
      # age family: the release happened on the 3rd frame of the hold
      assert counts[{"age", "offstage:a01-03"}]["enter_silence"] == 1
      assert counts[{"age", "offstage:a01-03"}]["active"] == 3

      expert = %{"state:offstage_low_j0" => %{active: 1000, silent: 100, enter_silence: 0.01, change: 0.1, resume: 0.5}}
      [row] = SilenceMap.compare(s, expert, min_n: 1)
      assert row.bucket == "state:offstage_low_j0"
      assert row.ratio == 33.33 and row.n == 3
    end

    test "dead and rebirth frames are not counted" do
      dead = player(action: 0)
      game = [frame(dead, ctrl(1.0, 0.5)), frame(dead, ctrl(0.5, 0.5))]
      assert SilenceMap.from_game(game, edge: @edge) == %{}
    end

    test "merge adds counts" do
      a = %{{"state", "x"} => %{"frames" => 1, "active" => 1}}
      b = %{{"state", "x"} => %{"frames" => 2, "active" => 0}, {"state", "y"} => %{"frames" => 1}}
      assert SilenceMap.merge(a, b) == %{{"state", "x"} => %{"frames" => 3, "active" => 1}, {"state", "y"} => %{"frames" => 1}}
    end
  end
end
