defmodule ExPhil.Agents.MewtwoFairExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MewtwoFairExpert
  alias ExPhil.Bridge.ControllerState

  @fixture "test/fixtures/replays/mewtwo_fair_chains.slp"

  setup_all do
    expert = MewtwoFairExpert.from_fixture(@fixture)

    {:ok, replay} = ExPhil.Data.Peppi.parse(@fixture)
    frames = ExPhil.Data.Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)

    %{expert: expert, frames: frames}
  end

  # af 200 never occurs in the fixture -> guaranteed table miss -> rules fire
  defp player(overrides) do
    Map.merge(
      %{
        action: 14.0,
        action_frame: 200.0,
        on_ground: true,
        jumps_left: 1,
        x: 0.0,
        y: 0.0,
        speed_y_self: 0.0,
        facing: 1
      },
      Map.new(overrides)
    )
  end

  describe "tables from fixture" do
    test "fine and coarse tables are populated", %{expert: expert} do
      assert map_size(expert.fine) > 50
      assert map_size(expert.coarse) > 50
    end

    test "fair frames (action 66) are covered", %{expert: expert} do
      fair_keys = Enum.filter(Map.keys(expert.coarse), fn {a, _, _} -> a == 66 end)
      assert length(fair_keys) > 10
    end

    test "fine table separates fair by jump variant", %{expert: expert} do
      # Same (action, af) with different jumps_left/height must appear as
      # distinct fine keys — that's the whole point of the physics key
      fair_fine =
        expert.fine
        |> Map.keys()
        |> Enum.filter(fn {a, _, _, _, _, _, _} -> a == 66 end)
        |> Enum.group_by(fn {_, af, _, _, _, _, _} -> af end)

      assert Enum.any?(fair_fine, fn {_af, keys} -> length(keys) > 1 end),
             "expected at least one action_frame with multiple physics variants"
    end
  end

  describe "recovery rules" do
    test "grounded off-script restarts with a Y tap against prev", %{expert: expert} do
      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true))
      assert c.button_y

      y_held = %{ControllerState.neutral() | button_y: true}
      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true), y_held)
      refute c.button_y
    end

    test "falling near the stage in an aerial attack taps L (L-cancel insurance)", %{
      expert: expert
    } do
      p = player(action: 66.0, on_ground: false, y: 4.0, speed_y_self: -2.0)

      {:ok, c} = MewtwoFairExpert.label(expert, p)
      assert c.button_l

      l_held = %{ControllerState.neutral() | button_l: true}
      {:ok, c} = MewtwoFairExpert.label(expert, p, l_held)
      refute c.button_l
    end

    test "falling near the stage OUTSIDE an attack never taps L (would airdodge)", %{
      expert: expert
    } do
      p = player(action: 29.0, on_ground: false, y: 4.0, speed_y_self: -2.0)
      {:ok, c} = MewtwoFairExpert.label(expert, p)
      refute c.button_l
    end

    test "airborne past the edge steers back toward center", %{expert: expert} do
      p = player(action: 25.0, on_ground: false, x: 75.0, y: 30.0, speed_y_self: 1.0)
      {:ok, c} = MewtwoFairExpert.label(expert, p)
      assert c.main_stick.x < 0.1
      refute c.button_l

      {:ok, c} = MewtwoFairExpert.label(expert, Map.put(p, :x, -75.0))
      assert c.main_stick.x > 0.9
    end

    test "edge safety overrides TABLE-covered states", %{expert: expert} do
      # (25, af 1, airborne) is densely covered by the fixture table — but at
      # the edge the table's center-stage answer (incl. the recorder's
      # fade-back drift) must lose to the steer (observed SDs)
      p = player(action: 25.0, action_frame: 1.0, on_ground: false, x: 80.0, y: 15.0)
      {:ok, c} = MewtwoFairExpert.label(expert, p)
      assert c.main_stick.x < 0.1
      refute c.button_y

      # L-cancel still fires while steering back in a falling aerial
      landing = player(action: 66.0, on_ground: false, x: 80.0, y: 4.0, speed_y_self: -2.0)
      {:ok, c} = MewtwoFairExpert.label(expert, landing)
      assert c.button_l
      assert c.main_stick.x < 0.1
    end

    test "rising in a jump c-sticks a fair toward facing", %{expert: expert} do
      p = player(action: 25.0, on_ground: false, y: 20.0, speed_y_self: 2.0, facing: 1)

      {:ok, c} = MewtwoFairExpert.label(expert, p)
      assert c.c_stick.x > 0.9

      {:ok, c} = MewtwoFairExpert.label(expert, Map.put(p, :facing, -1))
      assert c.c_stick.x < 0.1

      cstick_out = %{ControllerState.neutral() | c_stick: %{x: 1.0, y: 0.5}}
      {:ok, c} = MewtwoFairExpert.label(expert, p, cstick_out)
      assert_in_delta c.c_stick.x, 0.5, 0.01
    end

    test "high airborne fall rides out neutral", %{expert: expert} do
      p = player(action: 29.0, on_ground: false, y: 40.0, speed_y_self: -1.0)
      {:ok, c} = MewtwoFairExpert.label(expert, p)
      assert c == ControllerState.neutral()
    end

    test "grounded near the edge walks toward center instead of jumping", %{expert: expert} do
      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true, x: 75.0))
      refute c.button_y
      assert c.main_stick.x < 0.1

      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true, x: -75.0))
      refute c.button_y
      assert c.main_stick.x > 0.9
    end

    test "dead/respawn states are skipped", %{expert: expert} do
      assert :skip = MewtwoFairExpert.label(expert, player(action: 0.0))
      assert :skip = MewtwoFairExpert.label(expert, player(action: 12.0))
    end

    test "out of range approaches instead of jump-restarting", %{expert: expert} do
      opp = %{x: 60.0}
      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true, x: 0.0), nil, opp)
      refute c.button_y, "far away: approach, don't swing at nothing"
      assert c.main_stick.x > 0.9, "dash toward the opponent"

      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true, x: 0.0), nil, %{x: -60.0})
      assert c.main_stick.x < 0.1
    end

    test "in range keeps the jump-restart cycle", %{expert: expert} do
      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true, x: 0.0), nil, %{x: 20.0})
      assert c.button_y
    end

    test "opponent BEHIND takes the :turn_toward branch, not jump-restart", %{expert: expert} do
      # facing right (facing: 1), opponent to the left = behind, adjacent
      p = player(on_ground: true, x: 0.0, facing: 1)

      {:ok, c, tag} = MewtwoFairExpert.label_traced(expert, p, nil, %{x: -20.0})
      assert tag == :turn_toward, "pathology-#4 fix: behind must not jump-restart"
      refute c.button_y
      assert c.main_stick.x < 0.1, "steer toward the opponent (left)"

      # same geometry mirrored: facing left, opponent to the right
      {:ok, c, tag} =
        MewtwoFairExpert.label_traced(expert, player(on_ground: true, x: 0.0, facing: -1), nil, %{x: 20.0})

      assert tag == :turn_toward
      assert c.main_stick.x > 0.9, "steer toward the opponent (right)"
    end

    test "unknown opponent behaves like the pre-distance expert", %{expert: expert} do
      {:ok, c} = MewtwoFairExpert.label(expert, player(on_ground: true))
      assert c.button_y
    end
  end

  describe "fidelity on expert-quality data" do
    test "labels reproduce the fixture's own recorded buttons", %{expert: expert, frames: frames} do
      agreement = MewtwoFairExpert.button_agreement(expert, frames)
      # Measured 0.854 on the canonical fixture (human recording — noisier
      # than the TAS-like multishine fixture's 0.95+)
      assert agreement > 0.8, "button agreement only #{Float.round(agreement, 3)}"
    end
  end

  describe "early-air jump scrub (pathology #4)" do
    test "no jump-state cell before af 8 presses a jump button", %{expert: expert} do
      for {table, kind} <- [{expert.fine, :fine}, {expert.coarse, :coarse}],
          {key, controller} <- table,
          elem(key, 0) in 25..28 and elem(key, 1) < 8 do
        refute controller.button_x or controller.button_y,
               "#{kind} cell #{inspect(key)} presses jump before af 8"
      end
    end
  end
end
