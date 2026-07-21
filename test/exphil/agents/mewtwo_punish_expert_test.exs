defmodule ExPhil.Agents.MewtwoPunishExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MewtwoPunishExpert, as: Expert

  setup_all do
    %{e: Expert.new()}
  end

  defp me(overrides \\ []) do
    Map.merge(%{action: 14.0, action_frame: 5.0, x: 0.0, on_ground: true}, Map.new(overrides))
  end

  defp opp(overrides) do
    Map.merge(%{action: 14.0, action_frame: 1.0, x: 40.0}, Map.new(overrides))
  end

  describe "punishable?/1" do
    test "aerial landing lag and special-fall landing are always punishable" do
      assert Expert.punishable?(opp(action: 71.0, action_frame: 1.0))
      assert Expert.punishable?(opp(action: 43.0, action_frame: 1.0))
    end

    test "ground attacks only past the dangerous early frames" do
      refute Expert.punishable?(opp(action: 60.0, action_frame: 4.0))
      assert Expert.punishable?(opp(action: 60.0, action_frame: 20.0))
    end

    test "whiffed grab after its active window" do
      refute Expert.punishable?(opp(action: 212.0, action_frame: 3.0))
      assert Expert.punishable?(opp(action: 212.0, action_frame: 12.0))
    end

    test "neutral movement is not punishable" do
      refute Expert.punishable?(opp(action: 14.0))
      refute Expert.punishable?(opp(action: 20.0))
      refute Expert.punishable?(nil)
    end
  end

  describe "label/4 gating" do
    test "skips when opponent is not in lag", %{e: e} do
      assert :skip = Expert.label(e, me(), nil, opp(action: 14.0))
      assert :skip = Expert.label(e, me(), nil, nil)
    end

    test "skips when we are airborne or mid-move", %{e: e} do
      lag = opp(action: 71.0, x: 10.0)
      assert :skip = Expert.label(e, me(on_ground: false), nil, lag)
      assert :skip = Expert.label(e, me(action: 66.0), nil, lag)
    end

    test "skips beyond max reach (lag expires before arrival)", %{e: e} do
      assert :skip = Expert.label(e, me(x: 0.0), nil, opp(action: 71.0, x: 90.0))
    end

    test "skips during a ground attack's active frames (don't run into the hitbox)", %{e: e} do
      assert :skip = Expert.label(e, me(x: 0.0), nil, opp(action: 60.0, action_frame: 4.0, x: 30.0))
    end
  end

  describe "punish behavior" do
    test "dashes in from mid-range while the lag lasts", %{e: e} do
      {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 71.0, x: 40.0))
      assert c.main_stick.x > 0.9
      refute c.button_a
      refute c.button_z

      {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 71.0, x: -40.0))
      assert c.main_stick.x < 0.1
    end

    test "dtilts in close range from a standing state", %{e: e} do
      {:ok, c} = Expert.label(e, me(action: 14.0, x: 0.0), nil, opp(action: 71.0, x: 10.0))
      assert c.main_stick.y < 0.3
      assert c.button_a
    end

    test "dtilt A-press alternates against a previously landed press", %{e: e} do
      prev = %{button_a: true}
      {:ok, c} = Expert.label(e, me(action: 14.0, x: 0.0), prev, opp(action: 71.0, x: 10.0))
      assert c.main_stick.y < 0.3
      refute c.button_a
    end

    test "dash-grabs in close range out of a dash (dtilt unavailable)", %{e: e} do
      {:ok, c} = Expert.label(e, me(action: 20.0, x: 0.0), nil, opp(action: 71.0, x: 10.0))
      assert c.button_z
      refute c.button_a
    end
  end

  describe "combo cascade integration" do
    test "combo expert prefers chase over punish over fair" do
      combo = ExPhil.Agents.MewtwoComboExpert.from_frames([], [])

      # Opponent lying down -> chase branch (not punish, though 184 is 'lag')
      {:ok, c} = ExPhil.Agents.MewtwoComboExpert.label(combo, me(x: 0.0), nil, opp(action: 184.0, x: 8.0))
      assert c.button_z

      # Opponent in aerial landing lag -> punish branch (dtilt in range)
      {:ok, c} = ExPhil.Agents.MewtwoComboExpert.label(combo, me(x: 0.0), nil, opp(action: 71.0, x: 10.0))
      assert c.button_a
      assert c.main_stick.y < 0.3
    end
  end
end
