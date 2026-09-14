defmodule ExPhil.Eval.AnalogReadbackTest do
  use ExUnit.Case, async: true
  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Eval.ScenarioInputTiming

  test "conversion matches independently recorded sticks, boundary values, triggers and clicks" do
    fixture = File.read!("test/fixtures/analog_readback.json") |> Jason.decode!(keys: :atoms)

    for group <- fixture.profiles, row <- group.rows do
      profile = String.to_existing_atom(group.profile)
      expected = ScenarioInputTiming.expected_recording(row.sent, profile)
      recorded = row.recorded

      for {actual, predicted} <- [
            {recorded.main_stick_x, expected.main_stick.x},
            {recorded.main_stick_y, expected.main_stick.y},
            {recorded.c_stick_x, expected.c_stick.x},
            {recorded.c_stick_y, expected.c_stick.y},
            {recorded.l_trigger, expected.l_shoulder},
            {recorded.r_trigger, expected.r_shoulder}
          ],
          do: assert_in_delta(actual, predicted, 0.00001)
    end
  end

  test "right-trigger readback is not accepted on the left or silently summed" do
    input = ControllerState.to_input(%{ControllerState.neutral() | button_r: true})
    recorded = ScenarioInputTiming.expected_recording(input)
    trace = [%{frame: 1, sent: input, issued: input}]
    assert ScenarioInputTiming.verify(trace, %{1 => recorded}, 0).valid
    wrong = %{recorded | l_shoulder: 1.0, r_shoulder: 0.0}
    refute ScenarioInputTiming.verify(trace, %{1 => wrong}, 0).valid

    refute ScenarioInputTiming.verify(
             trace,
             %{1 => %{recorded | main_stick: %{x: 0.65, y: 0.5}}},
             0
           ).valid
  end

  test "legacy trigger compression requires an explicit historical profile" do
    input = %{ControllerState.to_input(ControllerState.neutral()) | shoulder: 1.0}
    legacy = ScenarioInputTiming.expected_recording(input, :pipe_v1)
    trace = [%{frame: 1, sent: input, issued: input}]
    refute ScenarioInputTiming.verify(trace, %{1 => legacy}, 0).valid
    assert ScenarioInputTiming.verify(trace, %{1 => legacy}, 0, profile: :pipe_v1).valid
    assert ScenarioInputTiming.expected_recording(input).l_shoulder == 1.0
  end
end
