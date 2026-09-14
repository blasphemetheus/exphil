defmodule ExPhil.Training.RecoveryTargetsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Eval.ScenarioInputTiming
  alias ExPhil.Training.Labels

  defp controller(command) do
    buttons = [
      :button_a,
      :button_b,
      :button_x,
      :button_y,
      :button_z,
      :button_l,
      :button_r,
      :button_d_up
    ]

    [main_x, main_y, c_x, c_y, left, right] = command["analog"]

    struct!(
      ControllerState,
      Enum.zip(buttons, command["buttons"]) ++
        [
          main_stick: %{x: main_x, y: main_y},
          c_stick: %{x: c_x, y: c_y},
          l_shoulder: left,
          r_shoulder: right
        ]
    )
  end

  test "recorded teacher futures label recovery transitions at every tested delay" do
    fixture = File.read!("test/fixtures/recovery_targets.json") |> Jason.decode!()

    for recovery <- fixture["cases"] do
      samples = recovery["samples"]
      issued = Map.new(samples, &{&1["frame"], controller(&1["issued"])})
      off_loop = Enum.filter(samples, &(not &1["on_loop"])) |> MapSet.new(& &1["frame"])
      assert MapSet.size(off_loop) == 7

      frames =
        Enum.map(samples, fn sample ->
          %{game_state: %{frame: sample["frame"]}, controller: controller(sample["recorded"])}
        end)
        |> Labels.tag(:recorded)

      for delay <- 2..5 do
        targets = Labels.at_delay(frames, delay, require_tagged: true)
        assert length(targets) == length(frames) - delay
        assert Enum.count(targets, &MapSet.member?(off_loop, &1.game_state.frame)) == 7

        for target <- targets do
          frame = target.game_state.frame
          input = ControllerState.to_input(target.controller)
          trace = [%{frame: frame, sent: input, issued: input}]
          assert ScenarioInputTiming.verify(trace, %{frame => issued[frame + delay]}, 0).valid
        end

        wrong_holds =
          Enum.count(targets, fn target ->
            frame = target.game_state.frame
            MapSet.member?(off_loop, frame) and issued[frame] != issued[frame + delay]
          end)

        assert wrong_holds > 0
      end
    end
  end

  test "neutral standing starts enter the loop with faithful delayed targets" do
    fixture = File.read!("test/fixtures/neutral_start_targets.json") |> Jason.decode!()

    for start <- fixture["cases"] do
      samples = start["samples"]
      assert hd(samples)["state"] == [14, 0, true]
      assert Enum.any?(samples, & &1["on_loop"])
      issued = Map.new(samples, &{&1["frame"], controller(&1["issued"])})

      frames =
        Enum.map(samples, fn sample ->
          %{game_state: %{frame: sample["frame"]}, controller: controller(sample["recorded"])}
        end)
        |> Labels.tag(:recorded)

      for delay <- 2..5, target <- Labels.at_delay(frames, delay) do
        frame = target.game_state.frame
        input = ControllerState.to_input(target.controller)
        trace = [%{frame: frame, sent: input, issued: input}]
        assert ScenarioInputTiming.verify(trace, %{frame => issued[frame + delay]}, 0).valid
      end
    end
  end
end
