defmodule ExPhil.Agents.MewtwoComboExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MewtwoComboExpert, as: Expert

  setup_all do
    %{e: Expert.from_fixture("test/fixtures/replays/mewtwo_fair_chains.slp")}
  end

  defp me(overrides \\ []) do
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

  test "routes to the chase expert when the opponent is down", %{e: e} do
    # Rolling opponent far away: pursue (chase behavior, not fair's approach)
    {:ok, c} = Expert.label(e, me(), nil, %{action: 200.0, action_frame: 1.0, x: 50.0})
    assert c.main_stick.x > 0.9
    refute c.button_y, "chase never jumps"

    # Lying opponent in range: grab
    {:ok, c} = Expert.label(e, me(), nil, %{action: 184.0, action_frame: 1.0, x: 8.0})
    assert c.button_z
  end

  test "routes to the fair expert when the opponent stands", %{e: e} do
    standing_far = %{action: 14.0, action_frame: 1.0, x: 60.0}
    {:ok, c} = Expert.label(e, me(), nil, standing_far)
    assert c.main_stick.x > 0.9, "fair expert approaches"
    refute c.button_z

    standing_close = %{action: 14.0, action_frame: 1.0, x: 20.0}
    {:ok, c} = Expert.label(e, me(), nil, standing_close)
    assert c.button_y, "fair expert jump-restarts in range"
  end

  test "dead states skip through both", %{e: e} do
    assert :skip = Expert.label(e, me(action: 0.0), nil, %{action: 184.0, action_frame: 1.0, x: 8.0})
  end
end
