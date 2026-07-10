defmodule ExPhil.Agents.FoxRecoveryExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.FoxRecoveryExpert
  alias ExPhil.Bridge.ControllerState

  setup_all do
    %{expert: FoxRecoveryExpert.new()}
  end

  defp player(overrides) do
    Map.merge(
      %{action: 29.0, x: 0.0, y: 0.0, on_ground: false, jumps_left: 0},
      Map.new(overrides)
    )
  end

  test "onstage grounded play is skipped (not recovery's business)", %{expert: e} do
    assert :skip = FoxRecoveryExpert.label(e, player(action: 14.0, on_ground: true))
    assert :skip = FoxRecoveryExpert.label(e, player(action: 20.0, on_ground: true, x: 50.0))
  end

  test "dead states are skipped", %{expert: e} do
    assert :skip = FoxRecoveryExpert.label(e, player(action: 0.0))
  end

  test "ledge hang gets up toward the stage", %{expert: e} do
    {:ok, c} = FoxRecoveryExpert.label(e, player(action: 253.0, x: 88.0, y: -10.0))
    assert c.main_stick.x < 0.1

    {:ok, c} = FoxRecoveryExpert.label(e, player(action: 253.0, x: -88.0, y: -10.0))
    assert c.main_stick.x > 0.9
  end

  test "offstage with a jump: taps X drifting inward (edge, not hold)", %{expert: e} do
    p = player(x: 100.0, y: -20.0, jumps_left: 1)

    {:ok, c} = FoxRecoveryExpert.label(e, p)
    assert c.button_x
    assert c.main_stick.x < 0.1

    x_held = %{ControllerState.neutral() | button_x: true}
    {:ok, c} = FoxRecoveryExpert.label(e, p, x_held)
    refute c.button_x, "jump must be a tap against prev"
  end

  test "offstage jumpless and low: Firefox aimed at the ledge", %{expert: e} do
    p = player(x: 110.0, y: -40.0, jumps_left: 0)

    {:ok, c} = FoxRecoveryExpert.label(e, p)
    assert c.button_b, "the entire E2 post-mortem: B was never pressed"
    assert c.main_stick.x < 0.5, "aim inward (target ledge is left of us)"
    assert c.main_stick.y > 0.5, "aim upward (we are below the ledge)"

    b_held = %{ControllerState.neutral() | button_b: true}
    {:ok, c} = FoxRecoveryExpert.label(e, p, b_held)
    refute c.button_b, "B must be a tap against prev"
    assert c.main_stick.y > 0.5, "keep aiming during the charge"
  end

  test "mid-special keeps steering at the ledge", %{expert: e} do
    {:ok, c} = FoxRecoveryExpert.label(e, player(action: 355.0, x: -120.0, y: -30.0))
    assert c.main_stick.x > 0.5
    assert c.main_stick.y > 0.5
    refute c.button_b
  end

  test "hitstun offstage holds survival DI inward", %{expert: e} do
    {:ok, c} = FoxRecoveryExpert.label(e, player(action: 87.0, x: 130.0, y: 10.0))
    assert c.main_stick.x < 0.1
    refute c.button_b
  end

  test "high over the stage drifts to center without buttons", %{expert: e} do
    {:ok, c} = FoxRecoveryExpert.label(e, player(x: 40.0, y: 60.0))
    assert c.main_stick.x < 0.1
    refute c.button_x
    refute c.button_b
  end

  test "low onstage aerials are skipped (normal play)", %{expert: e} do
    assert :skip = FoxRecoveryExpert.label(e, player(x: 20.0, y: 10.0, action: 66.0))
  end
end
