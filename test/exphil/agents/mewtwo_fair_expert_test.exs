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
        |> Enum.filter(fn {a, _, _, _, _} -> a == 66 end)
        |> Enum.group_by(fn {_, af, _, _, _} -> af end)

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

    test "falling near the stage taps L (L-cancel insurance)", %{expert: expert} do
      p = player(action: 29.0, on_ground: false, y: 4.0, speed_y_self: -2.0)

      {:ok, c} = MewtwoFairExpert.label(expert, p)
      assert c.button_l

      l_held = %{ControllerState.neutral() | button_l: true}
      {:ok, c} = MewtwoFairExpert.label(expert, p, l_held)
      refute c.button_l
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
  end

  describe "fidelity on expert-quality data" do
    test "labels reproduce the fixture's own recorded buttons", %{expert: expert, frames: frames} do
      agreement = MewtwoFairExpert.button_agreement(expert, frames)
      # Measured 0.854 on the canonical fixture (human recording — noisier
      # than the TAS-like multishine fixture's 0.95+)
      assert agreement > 0.8, "button agreement only #{Float.round(agreement, 3)}"
    end
  end
end
