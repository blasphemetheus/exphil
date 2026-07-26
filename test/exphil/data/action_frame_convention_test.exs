defmodule ExPhil.Data.ActionFrameConventionTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.ActionFrameConvention, as: AFC
  alias ExPhil.Eval.StateStreamDiff

  doctest ExPhil.Data.ActionFrameConvention

  @dir "test/fixtures/statestream"

  describe "the table is the measurement" do
    test "committed pairs never CONTRADICT the table" do
      # THE anti-drift test. The table is broader than the committed pairs
      # (most of it comes from a Mewtwo recording too large to vendor), so
      # this is a consistency check rather than equality: every action these
      # fixtures measure must match what the table claims. A Peppi/libmelee
      # upgrade that moves the convention fails here instead of silently
      # mis-normalizing every policy.
      derived =
        for name <- ["fox_ms_frame1", "fox_ms_float"], reduce: %{} do
          acc ->
            {:ok, report} =
              StateStreamDiff.diff("#{@dir}/#{name}.slp", "#{@dir}/#{name}.live-trace.log")

            Map.merge(acc, Map.new(report.mapping, fn {action, m} -> {action, m.delta} end))
        end

      refute Enum.empty?(derived)

      for {action, delta} <- derived do
        assert AFC.deltas()[action] == delta,
               "fixture says action #{action} has delta #{delta}, table says " <>
                 "#{inspect(AFC.deltas()[action])}"
      end
    end

    test "table matches the pinned mapping fixture exactly" do
      # action_frame_map.json is the recorded measurement, regenerable with
      # scripts/diff_state_streams.exs --json. Hand-editing @deltas without
      # re-measuring fails here.
      pinned =
        "#{@dir}/action_frame_map.json"
        |> File.read!()
        |> Jason.decode!()
        |> Map.fetch!("deltas")
        |> Map.new(fn {k, v} -> {String.to_integer(k), v} end)

      assert pinned == AFC.deltas()
    end

    test "covers the 77 measured actions" do
      assert AFC.coverage() == 77

      # The Fox-derived originals must survive every later merge.
      for action <- [24, 25, 29, 42, 323, 360, 361, 365, 366] do
        assert AFC.known?(action)
      end
    end

    test "deltas are only ever 0 or 1" do
      assert AFC.deltas() |> Map.values() |> Enum.uniq() |> Enum.sort() == [0, 1]
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
      # The deltas are NOT extrapolable (a majority are 1, but many — the
      # shine states 360/365, most hitstun/tumble — are 0), so an unmeasured
      # action must not be guessed at. 252/253 are cliff catch/wait, which no
      # recorded pair has covered yet.
      assert AFC.live_to_parsed(252, 5) == 5
      assert AFC.live_to_parsed(253, 0) == 0
      refute AFC.known?(252)
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
      assert AFC.unknown_actions([24, 253, 365, 252, 253]) == [252, 253]
      assert AFC.unknown_actions([24, 365, 14]) == []
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
