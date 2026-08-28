defmodule ExPhil.Interp.LoopStatsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Interp.LoopStats

  defp ctrl(opts \\ []) do
    %ControllerState{
      ControllerState.neutral()
      | button_d_up: Keyword.get(opts, :d_up, false),
        main_stick: %{x: Keyword.get(opts, :x, 0.5), y: 0.5}
    }
  end

  describe "taunt_stats/1" do
    test "counts entries, not frames" do
      # one 10-frame taunt, then a second taunt later
      actions = List.duplicate(1, 5) ++ List.duplicate(264, 10) ++ [1, 1] ++ [265, 265]

      assert %{entries: 2, frames: 12} = LoopStats.taunt_stats(actions)
    end

    test "a taunt already in progress on frame 0 counts as one entry" do
      assert %{entries: 1} = LoopStats.taunt_stats([264, 264, 1])
    end

    test "no taunts is zero, not an error" do
      assert %{entries: 0, frames: 0} = LoopStats.taunt_stats([1, 2, 3])
    end
  end

  describe "dpad_stats/1" do
    test "counts rising edges, so a held press is one press" do
      controllers = [
        ctrl(),
        ctrl(d_up: true),
        ctrl(d_up: true),
        ctrl(d_up: true),
        ctrl(),
        ctrl(d_up: true)
      ]

      assert %{presses: 2, frames: 4} = LoopStats.dpad_stats(controllers)
    end

    test "tolerates nil controllers" do
      assert %{presses: 0} = LoopStats.dpad_stats([nil, nil])
    end
  end

  describe "action_streaks/2" do
    test "long_frac is the share of frames inside runs at or over the threshold" do
      # 100 frames of one action (long), then 10 alternating (short runs)
      actions = List.duplicate(14, 100) ++ Enum.map(1..10, &rem(&1, 2))

      s = LoopStats.action_streaks(actions, min_long: 60)

      assert s.max == 100
      assert s.long_runs == 1
      assert_in_delta s.long_frac, 100 / 110, 0.001
    end

    test "no long runs gives long_frac 0.0" do
      s = LoopStats.action_streaks(Enum.map(1..100, &rem(&1, 7)), min_long: 60)
      assert s.long_frac == 0.0
    end
  end

  describe "input_streaks/2" do
    test "an identical held input is one long run (the argmax signature)" do
      s = LoopStats.input_streaks(List.duplicate(ctrl(), 120), min_long: 10)

      assert s.max == 120
      assert s.long_frac == 1.0
    end

    test "an input that changes every frame never locks (the sampling signature)" do
      controllers = Enum.map(1..120, fn i -> ctrl(x: rem(i, 8) / 8) end)
      s = LoopStats.input_streaks(controllers, min_long: 10)

      assert s.max < 10
      assert s.long_frac == 0.0
    end

    test "sub-bucket analog jitter does not break a run" do
      # 0.5 and 0.51 land in the same 1/8 bucket
      controllers = Enum.map(1..40, fn i -> ctrl(x: if(rem(i, 2) == 0, do: 0.5, else: 0.51)) end)
      assert LoopStats.input_streaks(controllers, min_long: 10).long_frac == 1.0
    end
  end

  describe "action_cycles/2" do
    test "detects a back-to-back repeated cycle over the transition sequence" do
      # GRAB_WAIT (216) / GRAB_PUMMEL (217) alternating 4 times, with each
      # state held for several frames — the real pummel-loop shape.
      actions =
        Enum.flat_map(1..4, fn _ ->
          List.duplicate(216, 8) ++ List.duplicate(217, 5)
        end)

      assert %{count: 1, max_repeats: 4, episodes: [ep]} = LoopStats.action_cycles(actions)
      assert ep.pattern == [216, 217]
      assert ep.period == 2
    end

    test "ignores a cycle repeated fewer than min_repeats times" do
      actions = List.duplicate(216, 8) ++ List.duplicate(217, 5) ++ List.duplicate(216, 8)

      assert %{count: 0} = LoopStats.action_cycles(actions, min_repeats: 3)
    end

    test "prefers the shortest period, so a 2-cycle is not reported as a 4-cycle" do
      actions = Enum.flat_map(1..6, fn _ -> [1, 1, 2, 2] end)

      assert %{episodes: [ep]} = LoopStats.action_cycles(actions)
      assert ep.period == 2
      assert ep.repeats == 6
    end

    test "a non-repeating sequence yields no episodes" do
      assert %{count: 0, max_repeats: 0} = LoopStats.action_cycles(Enum.to_list(1..50))
    end
  end

  describe "report/2 and aggregate/1" do
    test "rates are per minute of scored frames" do
      # 3600 frames = exactly 1 minute; two taunt entries
      actions =
        List.duplicate(14, 1000) ++
          [264] ++ List.duplicate(14, 1000) ++ [264] ++ List.duplicate(14, 1598)

      controllers = List.duplicate(ctrl(), 3600)
      r = LoopStats.report(%{actions: actions, controllers: controllers, n: 3600})

      assert_in_delta r.minutes, 1.0, 0.001
      assert_in_delta r.summary.taunts_per_min, 2.0, 0.001
    end

    test "aggregate reports mean and range per key" do
      mk = fn n ->
        LoopStats.report(%{
          actions: List.duplicate(264, n) ++ List.duplicate(14, 3600 - n),
          controllers: List.duplicate(ctrl(), 3600),
          n: 3600
        })
      end

      agg = LoopStats.aggregate([mk.(1), mk.(1)])

      assert agg.n == 2
      assert agg.stats.taunts_per_min.min == agg.stats.taunts_per_min.max
      assert length(agg.stats.taunts_per_min.values) == 2
    end

    test "aggregate of an empty list is empty, not a crash" do
      assert %{n: 0} = LoopStats.aggregate([])
    end
  end
end
