defmodule ExPhil.Agents.MewtwoTechChaseExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MewtwoTechChaseExpert, as: Expert
  alias ExPhil.Bridge.ControllerState

  setup_all do
    %{e: Expert.new()}
  end

  defp me(overrides \\ []) do
    Map.merge(%{action: 14.0, action_frame: 5.0, x: 0.0, on_ground: true}, Map.new(overrides))
  end

  defp opp(overrides) do
    Map.merge(%{action: 14.0, action_frame: 1.0, x: 40.0}, Map.new(overrides))
  end

  test "skips when the opponent is not in a knockdown state", %{e: e} do
    assert :skip = Expert.label(e, me(), nil, opp(action: 14.0))
    assert :skip = Expert.label(e, me(), nil, opp(action: 66.0))
    assert :skip = Expert.label(e, me(), nil, nil)
  end

  test "skips when we are airborne or mid-move", %{e: e} do
    downed = opp(action: 184.0)
    assert :skip = Expert.label(e, me(on_ground: false), nil, downed)
    assert :skip = Expert.label(e, me(action: 66.0), nil, downed)
  end

  test "pursues a rolling opponent's current position", %{e: e} do
    # Roll to our right
    {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 200.0, x: 40.0))
    assert c.main_stick.x > 0.9
    refute c.button_z

    # Roll crossing to our left
    {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 201.0, x: -40.0))
    assert c.main_stick.x < 0.1
  end

  test "grabs in punish range as a lying opponent becomes actionable", %{e: e} do
    lying = opp(action: 184.0, x: 8.0)

    {:ok, c} = Expert.label(e, me(x: 0.0), nil, lying)
    assert c.button_z, "in range on a downed-wait opponent: grab"

    z_held = %{ControllerState.neutral() | button_z: true}
    {:ok, c} = Expert.label(e, me(x: 0.0), z_held, lying)
    refute c.button_z, "grab is a tap against prev"
  end

  test "holds position (no grab) while a close opponent is still locked", %{e: e} do
    # Down-bounce: not actionable, don't burn the grab
    {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 183.0, x: 8.0))
    refute c.button_z
    assert_in_delta c.main_stick.x, 0.65, 0.01

    # Early tech-in-place frames: same
    {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 199.0, action_frame: 3.0, x: 8.0))
    refute c.button_z
  end

  test "grabs a late tech-in-place in range", %{e: e} do
    {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 199.0, action_frame: 15.0, x: 8.0))
    assert c.button_z
  end

  test "chases getup rolls", %{e: e} do
    {:ok, c} = Expert.label(e, me(x: 0.0), nil, opp(action: 188.0, x: 50.0))
    assert c.main_stick.x > 0.9
  end
end
