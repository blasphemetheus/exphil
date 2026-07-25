defmodule ExPhil.Eval.StateStreamDiffTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.StateStreamDiff

  @dir "test/fixtures/statestream"
  @frame1 "#{@dir}/fox_ms_frame1"
  @float "#{@dir}/fox_ms_float"

  # The parsed<->live action_frame convention (task #8 / GOTCHAS #81).
  #
  # This is the artifact task #8 phase 1 exists to produce. It is pinned so
  # the convention can never silently drift again: a Peppi upgrade, a
  # libmelee upgrade, or a recorder change that moves any of these numbers
  # breaks this test instead of silently collapsing a trained policy on live
  # states (which no training loss can detect — the loss lives entirely in
  # parsed space).
  #
  # delta == live_af - parsed_af.
  @expected_delta %{
    24 => 1,
    25 => 1,
    29 => 1,
    42 => 1,
    323 => 1,
    360 => 0,
    361 => 1,
    365 => 0,
    366 => 1
  }

  setup_all do
    {:ok, frame1} = StateStreamDiff.diff("#{@frame1}.slp", "#{@frame1}.live-trace.log")
    {:ok, float} = StateStreamDiff.diff("#{@float}.slp", "#{@float}.live-trace.log")
    %{frame1: frame1, float: float}
  end

  describe "alignment" do
    test "anchors on the first jumpsquat entry, not frame 0", ctx do
      # Both pairs: the replay starts at -123 (countdown), the trace at f0.
      # Assuming f0 == parsed frame 0 would misalign by 123 frames.
      assert ctx.frame1.offset == -123
      assert ctx.float.offset == -123
    end

    test "aligned frames agree perfectly on action, on_ground and y", ctx do
      for report <- [ctx.frame1, ctx.float] do
        assert report.agreement.action == 1.0
        assert report.agreement.on_ground == 1.0
        assert report.agreement.y == 1.0
      end
    end

    test "compares the full trace", ctx do
      assert ctx.frame1.frames_compared == 300
      assert ctx.float.frames_compared == 300
    end

    test "align/3 reports a missing anchor rather than guessing", ctx do
      _ = ctx
      {:ok, parsed} = StateStreamDiff.parse_replay("#{@frame1}.slp", 1)
      {:ok, trace} = StateStreamDiff.parse_trace("#{@frame1}.live-trace.log")

      assert {:error, {:anchor_not_found, _side, 998}} =
               StateStreamDiff.align(parsed, trace, anchor_action: 998)
    end
  end

  describe "action_frame mapping" do
    test "matches the pinned convention in both pairs", ctx do
      for report <- [ctx.frame1, ctx.float] do
        for {action, mapped} <- report.mapping do
          assert Map.has_key?(@expected_delta, action),
                 "action #{action} appeared with delta #{inspect(mapped.delta)} but is not pinned"

          assert mapped.delta == @expected_delta[action],
                 "action #{action}: expected delta #{@expected_delta[action]}, got #{inspect(mapped.delta)}"
        end
      end
    end

    test "every action has a single constant offset", ctx do
      for report <- [ctx.frame1, ctx.float] do
        assert report.inconsistent_actions == []

        for {_action, mapped} <- report.mapping do
          assert mapped.consistent?
          assert length(mapped.deltas) == 1
        end
      end
    end

    test "the two pairs never disagree about a shared action", ctx do
      shared =
        MapSet.intersection(
          MapSet.new(Map.keys(ctx.frame1.mapping)),
          MapSet.new(Map.keys(ctx.float.mapping))
        )

      refute Enum.empty?(shared)

      for action <- shared do
        assert ctx.frame1.mapping[action].delta == ctx.float.mapping[action].delta,
               "action #{action} has a run-dependent offset — it is not a fixed convention"
      end
    end

    test "reproduces the specific shifts recorded in GOTCHAS #81", ctx do
      # "parsed gives jumpsquat af 0,1,2 / 366 af 0 / 361 af 1,2 while live
      #  libmelee gives 1,2,3 / 1 / 2,3 (365 happens to agree)"
      js = ctx.frame1.mapping[24]
      assert js.parsed_af == 0..2
      assert js.live_af == 1..3

      a366 = ctx.frame1.mapping[366]
      assert a366.parsed_af == 0..0
      assert a366.live_af == 1..1

      a361 = ctx.frame1.mapping[361]
      assert a361.parsed_af == 1..2
      assert a361.live_af == 2..3

      a365 = ctx.frame1.mapping[365]
      assert a365.parsed_af == a365.live_af, "365 is documented to agree"
      assert a365.delta == 0
    end

    test "action_frame is the ONLY field that shifts", ctx do
      # If a future change starts shifting y or on_ground too, the embedding
      # boundary fix (phase 2 option 1) would be incomplete — catch it here.
      assert ctx.frame1.shifted_fields == [:action_frame]
      assert ctx.float.shifted_fields == [:action_frame]
    end

    test "the shift is large enough to matter", ctx do
      # Sanity on the premise: if af mostly agreed, this whole investigation
      # would be chasing noise. It disagrees on most frames.
      assert ctx.frame1.agreement.action_frame < 0.5
      assert ctx.float.agreement.action_frame < 0.5
    end
  end

  describe "to_live_af/3" do
    test "applies the per-action delta", ctx do
      m = ctx.frame1.mapping
      assert StateStreamDiff.to_live_af(m, 24, 0) == 1
      assert StateStreamDiff.to_live_af(m, 24, 2) == 3
      assert StateStreamDiff.to_live_af(m, 365, 1) == 1
    end

    test "passes unknown actions through unchanged", ctx do
      assert StateStreamDiff.to_live_af(ctx.frame1.mapping, 9999, 7) == 7
    end
  end

  describe "parsing" do
    test "reads only [trace] lines out of raw recorder stdout" do
      # The logs are full recorder stdout and contain build noise; the parser
      # must ignore everything that is not a trace line.
      {:ok, rows} = StateStreamDiff.parse_trace("#{@frame1}.live-trace.log")

      assert length(rows) == 300
      assert hd(rows).f == 0
      assert List.last(rows).f == 299
      assert Enum.all?(rows, &is_integer(&1.act))
    end

    test "surfaces a file with no trace lines as an error" do
      assert {:error, {:no_trace_lines, _}} = StateStreamDiff.parse_trace("mix.exs")
    end

    test "parse_replay/2 yields one row per frame with the fields we diff" do
      {:ok, rows} = StateStreamDiff.parse_replay("#{@frame1}.slp", 1)

      assert length(rows) == 1199
      row = hd(rows)
      assert row.f == -123
      assert Map.has_key?(row, :act)
      assert Map.has_key?(row, :af)
      assert Map.has_key?(row, :gnd)
      assert Map.has_key?(row, :y)
    end
  end
end
