defmodule ExPhil.Agents.EdgeTurnaroundExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.EdgeTurnaroundExpert

  setup_all do
    %{expert: EdgeTurnaroundExpert.new()}
  end

  # Danger-zone dashing by default: FD edge 85.57, margin 20 -> |x| > 65.57
  defp player(overrides) do
    Map.merge(
      %{action: 0x14, x: 80.0, on_ground: true, speed_ground_x_self: 2.0, facing: 1},
      Map.new(overrides)
    )
  end

  test "dashing toward the near edge inside the margin: hard stick reversal", %{expert: e} do
    {:ok, c} = EdgeTurnaroundExpert.label(e, player([]))
    assert c.main_stick.x == 0.0
    refute c.button_x or c.button_b

    {:ok, c} = EdgeTurnaroundExpert.label(e, player(x: -80.0, speed_ground_x_self: -2.0, facing: -1))
    assert c.main_stick.x == 1.0
  end

  test "running counts like dashing", %{expert: e} do
    assert {:ok, _} = EdgeTurnaroundExpert.label(e, player(action: 0x15))
  end

  test "center-stage dashing is skipped (outside the danger margin)", %{expert: e} do
    assert :skip = EdgeTurnaroundExpert.label(e, player(x: 30.0))
  end

  test "dashing AWAY from the edge is skipped (already correcting)", %{expert: e} do
    assert :skip = EdgeTurnaroundExpert.label(e, player(speed_ground_x_self: -2.0, facing: -1))
  end

  test "airborne and non-dash grounded states are skipped", %{expert: e} do
    assert :skip = EdgeTurnaroundExpert.label(e, player(on_ground: false))
    # WAIT (14) near the edge is standing, not the SD pattern
    assert :skip = EdgeTurnaroundExpert.label(e, player(action: 14))
  end

  test "facing is the fallback when ground speed is missing or zero", %{expert: e} do
    assert {:ok, _} = EdgeTurnaroundExpert.label(e, player(speed_ground_x_self: nil))
    assert {:ok, _} = EdgeTurnaroundExpert.label(e, player(speed_ground_x_self: 0.0))
    assert :skip = EdgeTurnaroundExpert.label(e, player(speed_ground_x_self: nil, facing: -1))
  end

  test "per-stage edge geometry moves the danger zone", %{expert: e} do
    # x=50 is safe on FD...
    assert :skip = EdgeTurnaroundExpert.label(e, player(x: 50.0))
    # ...but dangerous on Yoshi's (edge 56, margin 20 -> |x| > 36)
    ys = EdgeTurnaroundExpert.new(edge_x: 56.0)
    assert {:ok, _} = EdgeTurnaroundExpert.label(ys, player(x: 50.0))
  end
end
