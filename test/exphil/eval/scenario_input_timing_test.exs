defmodule ExPhil.Eval.ScenarioInputTimingTest do
  use ExUnit.Case, async: true
  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Eval.ScenarioInputTiming

  defp fixture do
    neutral = ControllerState.neutral()
    pressed = %{neutral | button_b: true}
    controllers = [neutral, pressed, neutral, neutral, pressed, pressed]

    inputs =
      controllers
      |> Enum.with_index()
      |> Map.new(fn {controller, frame} ->
        {frame, controller}
      end)

    trace =
      for frame <- 0..3 do
        %{
          frame: frame,
          sent: ControllerState.to_input(inputs[frame]),
          issued: ControllerState.to_input(inputs[frame + 2])
        }
      end

    {trace, inputs}
  end

  test "checks the declared latency and excludes unsent tail decisions" do
    {trace, inputs} = fixture()
    result = ScenarioInputTiming.verify(trace, inputs, 2)
    assert result.valid
    assert result.sent == %{count: 4, matches: 4, mismatches: []}
    assert result.decisions == %{count: 2, matches: 2, mismatches: []}
    assert result.unsent_tail_decisions == 2
  end

  test "a one-frame delivery shift fails instead of being scored as policy quality" do
    {trace, inputs} = fixture()
    shifted = Map.new(inputs, fn {frame, controller} -> {frame + 1, controller} end)
    refute ScenarioInputTiming.verify(trace, shifted, 2).valid
  end

  test "empty and missing replay evidence fail closed" do
    {trace, _inputs} = fixture()
    refute ScenarioInputTiming.verify(trace, %{}, 2).valid
    refute ScenarioInputTiming.verify([], %{}, 2).valid
  end

  test "both issuance and send paths must match" do
    {trace, inputs} = fixture()
    wrong = Enum.map(trace, &Map.put(&1, :issued, ControllerState.to_input(inputs[1])))
    result = ScenarioInputTiming.verify(wrong, inputs, 2)
    assert result.sent.matches == result.sent.count
    refute result.valid
  end

  test "duplicate trace frames are not accepted as timing evidence" do
    {trace, inputs} = fixture()
    refute ScenarioInputTiming.verify([hd(trace) | trace], inputs, 2).valid
  end

  test "missing intermediate and reordered frames fail closed" do
    {trace, inputs} = fixture()
    refute ScenarioInputTiming.verify(List.delete_at(trace, 2), inputs, 2).valid
    refute ScenarioInputTiming.verify(Enum.reverse(trace), inputs, 2).valid
  end

  test "long varied input streams accept exact delivery and reject a late segment" do
    neutral = ControllerState.neutral()

    inputs =
      Map.new(0..520, fn frame ->
        controller = %{
          neutral
          | button_b: rem(frame, 3) == 0,
            button_x: rem(frame, 7) == 0,
            main_stick: %{x: rem(frame * 17, 101) / 100, y: 0.5}
        }

        {frame, controller}
      end)

    for delay <- 0..8 do
      trace =
        for frame <- 0..499 do
          %{
            frame: frame,
            sent: ControllerState.to_input(inputs[frame]),
            issued: ControllerState.to_input(inputs[frame + delay])
          }
        end

      recorded =
        Map.new(inputs, fn {frame, controller} ->
          {frame, ScenarioInputTiming.expected_recording(ControllerState.to_input(controller))}
        end)

      assert ScenarioInputTiming.verify(trace, recorded, delay).valid
      late = Enum.reduce(250..300, recorded, &Map.put(&2, &1, recorded[&1 - 1]))
      refute ScenarioInputTiming.verify(trace, late, delay).valid
    end
  end

  test "directory verification waits for a late replay file" do
    path = "test/fixtures/replays/fox_multishine_closed_d1.slp"
    {:ok, replay} = ExPhil.Data.Peppi.parse(path)

    inputs =
      replay
      |> ExPhil.Data.Peppi.to_training_frames()
      |> Map.new(&{&1.game_state.frame, &1.controller})

    trace =
      for frame <- 0..5 do
        %{
          frame: frame,
          sent: ControllerState.to_input(inputs[frame]),
          issued: ControllerState.to_input(inputs[frame + 2])
        }
      end

    directory =
      Path.join(System.tmp_dir!(), "scenario_timing_#{System.unique_integer([:positive])}")

    File.mkdir_p!(directory)
    on_exit(fn -> File.rm_rf!(directory) end)

    writer =
      Task.async(fn ->
        Process.sleep(100)
        File.cp!(path, Path.join(directory, "late.slp"))
      end)

    assert ScenarioInputTiming.verify_directory(directory, trace, 2).valid
    Task.await(writer)
  end

  test "missing directory fails closed with a diagnostic reason" do
    directory =
      Path.join(System.tmp_dir!(), "missing_timing_#{System.unique_integer([:positive])}")

    result = ScenarioInputTiming.verify_directory(directory, [], 2, 0)
    refute result.valid
    assert result.reason =~ "no replay"
  end
end
