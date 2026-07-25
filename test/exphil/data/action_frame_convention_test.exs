defmodule ExPhil.Data.ActionFrameConventionTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.ActionFrameConvention, as: AFC
  alias ExPhil.Eval.StateStreamDiff

  doctest ExPhil.Data.ActionFrameConvention

  @dir "test/fixtures/statestream"

  describe "the table is the measurement" do
    test "static deltas match a fresh derivation from the committed pairs" do
      # THE anti-drift test. AFC hardcodes the table so it is usable at
      # runtime without parsing fixtures; this proves the hardcoded copy still
      # equals what the recordings actually say. If someone edits @deltas by
      # hand, or a Peppi/libmelee upgrade moves the convention, this fails
      # here rather than silently mis-normalizing every policy.
      derived =
        for name <- ["fox_ms_frame1", "fox_ms_float"], reduce: %{} do
          acc ->
            {:ok, report} =
              StateStreamDiff.diff("#{@dir}/#{name}.slp", "#{@dir}/#{name}.live-trace.log")

            Map.merge(acc, Map.new(report.mapping, fn {action, m} -> {action, m.delta} end))
        end

      assert derived == AFC.deltas()
    end

    test "covers exactly the 9 measured actions" do
      assert AFC.coverage() == 9
      assert Map.keys(AFC.deltas()) |> Enum.sort() == [24, 25, 29, 42, 323, 360, 361, 365, 366]
    end
  end

  describe "live_to_parsed/2" do
    test "subtracts the measured delta" do
      assert AFC.live_to_parsed(24, 1) == 0
      assert AFC.live_to_parsed(24, 3) == 2
      assert AFC.live_to_parsed(366, 1) == 0
      assert AFC.live_to_parsed(361, 2) == 1
    end

    test "leaves the agreeing actions alone" do
      assert AFC.live_to_parsed(360, 2) == 2
      assert AFC.live_to_parsed(365, 3) == 3
    end

    test "round-trips with parsed_to_live/2" do
      for {action, _delta} <- AFC.deltas(), af <- 0..5 do
        assert AFC.live_to_parsed(action, AFC.parsed_to_live(action, af)) == af
      end
    end
  end

  describe "coverage limits are explicit, not silent" do
    test "unmeasured actions pass through unchanged" do
      # The deltas are NOT extrapolable (mostly 1, but 360/365 are 0), so an
      # unmeasured action must not be guessed at.
      assert AFC.live_to_parsed(14, 5) == 5
      assert AFC.live_to_parsed(252, 0) == 0
      refute AFC.known?(14)
    end

    test "negative action_frame sentinels are never adjusted" do
      # af == -1 means "no action frame" (seen on 322/324), not a counter.
      assert AFC.live_to_parsed(24, -1) == -1
    end

    test "nil action or af passes through" do
      assert AFC.live_to_parsed(nil, 3) == 3
      assert AFC.live_to_parsed(24, nil) == nil
    end

    test "unknown_actions/1 sizes the gap for a workload" do
      assert AFC.unknown_actions([24, 14, 365, 252, 14]) == [14, 252]
      assert AFC.unknown_actions([24, 365]) == []
    end

    test "known?/1 is honest about what was measured" do
      assert AFC.known?(24)
      assert AFC.known?(365)
      refute AFC.known?(0)
      refute AFC.known?(nil)
    end
  end

  describe "normalize_player/1" do
    test "shifts action_frame in place" do
      p = %{action: 24, action_frame: 1, y: 3.0}
      assert AFC.normalize_player(p) == %{action: 24, action_frame: 0, y: 3.0}
    end

    test "tolerates float action ids (Peppi/bridge both emit floats)" do
      assert AFC.normalize_player(%{action: 24.0, action_frame: 3}).action_frame == 2
    end

    test "passes through nil and structs missing the keys" do
      assert AFC.normalize_player(nil) == nil
      assert AFC.normalize_player(%{action: nil, action_frame: 4}).action_frame == 4
    end
  end
end
