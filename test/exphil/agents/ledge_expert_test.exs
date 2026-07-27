defmodule ExPhil.Agents.LedgeExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.LedgeExpert
  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Constants

  @cliff_catch Constants.cliff_catch()
  @cliff_wait Constants.cliff_wait()

  defp player(action, overrides \\ []) do
    Map.merge(
      %{action: action, action_frame: 1, on_ground: false, x: 85.57 + 2.0, y: -12.0, percent: 0.0, jumps_left: 1},
      Map.new(overrides)
    )
  end

  defp neutral, do: ControllerState.neutral()

  test "new/1 rejects ledgedash (timing tables not implemented) and unknown strategies" do
    assert_raise ArgumentError, ~r/ledgedash/, fn -> LedgeExpert.new(strategy: :ledgedash) end
    assert_raise ArgumentError, ~r/unknown/, fn -> LedgeExpert.new(strategy: :wavedash) end
  end

  test "CliffCatch and getup animations label neutral (no control)" do
    expert = LedgeExpert.new()
    assert {:ok, c} = LedgeExpert.label(expert, player(@cliff_catch))
    assert c == neutral()

    for action <- Constants.ledge_getups() do
      assert {:ok, c} = LedgeExpert.label(expert, player(action))
      assert c == neutral()
    end
  end

  test "getup pushes the stick toward the stage from either ledge" do
    expert = LedgeExpert.new(strategy: :getup)

    assert {:ok, right} = LedgeExpert.label(expert, player(@cliff_wait, x: 87.5))
    assert right.main_stick.x == 0.0

    assert {:ok, left} = LedgeExpert.label(expert, player(@cliff_wait, x: -87.5))
    assert left.main_stick.x == 1.0
  end

  test "button strategies tap with edge alternation off prev" do
    for {strategy, button} <- [attack: :button_a, roll: :button_l, jump: :button_x] do
      expert = LedgeExpert.new(strategy: strategy)

      assert {:ok, press} = LedgeExpert.label(expert, player(@cliff_wait), nil)
      assert Map.get(press, button) == true

      assert {:ok, release} = LedgeExpert.label(expert, player(@cliff_wait), press)
      assert Map.get(release, button) == false
    end
  end

  test "drop_jump: down in CliffWait, then jump inward while falling near the edge" do
    expert = LedgeExpert.new(strategy: :drop_jump)

    assert {:ok, drop} = LedgeExpert.label(expert, player(@cliff_wait))
    assert drop.main_stick.y == 0.0

    falling = player(29, on_ground: false, x: 87.0, y: -20.0, jumps_left: 1)
    assert {:ok, jump} = LedgeExpert.label(expert, falling)
    assert jump.button_x == true
    assert jump.main_stick.x == 0.0
  end

  test "non-ledge states are skipped; slow_getup? branches at 100%" do
    expert = LedgeExpert.new()
    assert :skip = LedgeExpert.label(expert, player(14, on_ground: true, x: 0.0))

    refute LedgeExpert.slow_getup?(%{percent: 99.9})
    assert LedgeExpert.slow_getup?(%{percent: 100.0})
  end
end
