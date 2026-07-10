defmodule ExPhil.Training.MixFramesTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.MixFrames
  alias ExPhil.Bridge.ControllerState

  @moduletag :tmp_dir

  defp frame(n) do
    %{
      game_state: %{frame: n, players: %{}},
      controller: %{ControllerState.neutral() | button_a: true},
      prev_controller: nil
    }
  end

  defp write_export(dir, name, frame_lists, delay) do
    path = Path.join(dir, name)

    payload = %{
      expert: "test_drill",
      exported_at: "2026-07-10T00:00:00Z",
      action_delay: delay,
      frame_lists: frame_lists
    }

    File.write!(path, :erlang.term_to_binary(payload, [:compressed]))
    path
  end

  test "loads and shifts per segment with stats", %{tmp_dir: dir} do
    # Two segments of 5 contiguous frames each; delay 2 drops the last 2
    # frames of each segment (shift_actions contiguity)
    seg = fn base -> Enum.map(base..(base + 4), &frame/1) end
    path = write_export(dir, "a.frames", [seg.(0), seg.(100)], 2)

    {frames, [stats]} = MixFrames.load(path, action_delay: 2)

    assert length(frames) == 6
    assert stats.expert == "test_drill"
    assert stats.frames == 6

    # Shift must not cross the segment boundary: no frame may have a target
    # sourced from the other segment (frame numbers 3,4 and 103,104 dropped)
    numbers = Enum.map(frames, & &1.game_state.frame)
    assert numbers == [0, 1, 2, 100, 101, 102]
  end

  test "globs and comma lists combine files", %{tmp_dir: dir} do
    write_export(dir, "one.frames", [[frame(0), frame(1), frame(2)]], 0)
    write_export(dir, "two.frames", [[frame(10), frame(11)]], 0)

    {frames, stats} = MixFrames.load(Path.join(dir, "*.frames"), action_delay: 0)

    assert length(frames) == 5
    assert length(stats) == 2
  end

  test "unreadable file is skipped with empty stats", %{tmp_dir: dir} do
    File.write!(Path.join(dir, "junk.frames"), "not a term")

    {frames, [stats]} = MixFrames.load(Path.join(dir, "junk.frames"), action_delay: 0)
    assert frames == []
    assert stats.frames == 0
  end
end
