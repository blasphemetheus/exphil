defmodule ExPhil.Interp.BasinRolloutEntryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Interp.BasinRollout

  @squat_wait ExPhil.Constants.squat_wait()
  @stand 14

  defp frame(n, action) do
    %{game_state: %{frame: n, players: %{1 => %{action: action}}}}
  end

  describe "find_absorbed_index/2" do
    test "finds the start of the first sustained SquatWait run" do
      frames =
        Enum.map(0..99, &frame(&1, @stand)) ++
          Enum.map(100..299, &frame(&1, @squat_wait))

      assert BasinRollout.find_absorbed_index(frames) == {:ok, 100}
    end

    test "tolerates sub-threshold gaps inside the run" do
      # 120-frame window at index 100 contains 6 stand frames -> 95% occupancy
      frames =
        Enum.map(0..99, &frame(&1, @stand)) ++
          Enum.map(100..149, &frame(&1, @squat_wait)) ++
          Enum.map(150..155, &frame(&1, @stand)) ++
          Enum.map(156..299, &frame(&1, @squat_wait))

      assert BasinRollout.find_absorbed_index(frames) == {:ok, 100}
    end

    test "returns :none without a sustained run or on short replays" do
      no_squat = Enum.map(0..299, &frame(&1, @stand))
      assert BasinRollout.find_absorbed_index(no_squat) == :none

      brief = Enum.map(0..99, &frame(&1, @squat_wait))
      assert BasinRollout.find_absorbed_index(brief) == :none

      # alternating never reaches 90% occupancy
      alternating =
        Enum.map(0..299, &frame(&1, if(rem(&1, 2) == 0, do: @squat_wait, else: @stand)))

      assert BasinRollout.find_absorbed_index(alternating) == :none
    end
  end
end
