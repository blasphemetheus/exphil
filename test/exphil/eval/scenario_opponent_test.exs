defmodule ExPhil.Eval.ScenarioOpponentTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.ScenarioOpponent

  test "neutral readback uses Peppi controller fields and checks all controls" do
    fields =
      Map.new(
        [
          :button_a,
          :button_b,
          :button_x,
          :button_y,
          :button_z,
          :button_l,
          :button_r,
          :button_start,
          :button_d_up,
          :button_d_down,
          :button_d_left,
          :button_d_right
        ],
        &{&1, false}
      )
      |> Map.merge(%{
        main_stick_x: 0.5,
        main_stick_y: 0.5,
        c_stick_x: 0.5,
        c_stick_y: 0.5,
        l_trigger: 0.0,
        r_trigger: 0.0
      })

    controller = struct!(ExPhil.Data.Peppi.Controller, fields)
    assert ScenarioOpponent.recorded_neutral?(controller)
    refute ScenarioOpponent.recorded_neutral?(%{controller | button_start: true})
    refute ScenarioOpponent.recorded_neutral?(%{controller | main_stick_x: 0.7})
    refute ScenarioOpponent.recorded_neutral?(%{controller | l_trigger: 0.5})
  end

  test "neutral response ignores attacks without changing replay mode" do
    neutral = %{buttons: %{a: false}, main_stick: %{x: 0.5, y: 0.5}}
    attack = %{buttons: %{a: true}, main_stick: %{x: 1.0, y: 0.5}}
    assert ScenarioOpponent.input("neutral", {%{}, attack}, neutral) == neutral
    assert ScenarioOpponent.input("replay", {%{}, attack}, neutral) == attack
    assert ScenarioOpponent.input("replay", nil, neutral) == neutral
    assert ScenarioOpponent.input("neutral", nil, neutral) == neutral
    assert_raise ArgumentError, fn -> ScenarioOpponent.input("invalid", nil, neutral) end
  end
end
