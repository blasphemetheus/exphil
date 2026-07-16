defmodule ExPhil.Eval.ScenarioScanTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.ScenarioScan

  # A committed scenario source replay (probe game: Mewtwo P1 policy vs
  # tech_random Fox P2 on FD) — also referenced by scenarios/manifest.json.
  @fixture "scenarios/replays/Game_20260715T195318.slp"

  @wait 14
  @dash 20
  @down_bound_u 183
  @tech_in_place 199

  defp player(overrides) do
    Map.merge(
      %{x: 0.0, y: 0.0, action: @wait, facing: 1, on_ground: true, stock: 4},
      overrides
    )
  end

  defp frame(n, p1_over, p2_over) do
    %{frame: n, p1: player(p1_over), p2: player(p2_over)}
  end

  defp frames(n, p1_over, p2_over \\ %{}) do
    Enum.map(1..n, &frame(&1, p1_over, p2_over))
  end

  describe "opponent_behind/2" do
    test "detects a persistent behind situation at its onset" do
      fs =
        frames(50, %{facing: 1}, %{x: 60.0}) ++
          Enum.map(51..80, &frame(&1, %{facing: 1}, %{x: -15.0}))

      assert [%{type: :opponent_behind, frame: 51}] = ScenarioScan.opponent_behind(fs)
    end

    test "ignores brief crossups shorter than the persistence bound" do
      fs =
        frames(50, %{facing: 1}, %{x: 60.0}) ++
          Enum.map(51..55, &frame(&1, %{facing: 1}, %{x: -15.0})) ++
          Enum.map(56..90, &frame(&1, %{facing: 1}, %{x: 60.0}))

      assert ScenarioScan.opponent_behind(fs) == []
    end

    test "opponent in front does not trigger" do
      fs = frames(100, %{facing: 1}, %{x: 15.0})
      assert ScenarioScan.opponent_behind(fs) == []
    end

    test "airborne opponent does not trigger" do
      fs = frames(100, %{facing: 1}, %{x: -15.0, on_ground: false})
      assert ScenarioScan.opponent_behind(fs) == []
    end
  end

  describe "tech_chase/1" do
    test "detects P2 entering knockdown from outside the lifecycle" do
      fs =
        frames(20, %{}, %{action: @wait}) ++
          (frames(10, %{}, %{action: @down_bound_u}) |> reframe(21))

      assert [%{type: :tech_chase, frame: 21, note: note}] = ScenarioScan.tech_chase(fs)
      assert note =~ "missed_bound_u"
    end

    test "tech entries are detected with their class" do
      fs =
        frames(20, %{}, %{action: 88}) ++
          (frames(10, %{}, %{action: @tech_in_place}) |> reframe(21))

      assert [%{frame: 21, note: note}] = ScenarioScan.tech_chase(fs)
      assert note =~ "tech_in_place"
    end

    test "no entry from within the lifecycle (bound -> wait is one episode)" do
      fs =
        frames(10, %{}, %{action: @down_bound_u}) ++
          (frames(10, %{}, %{action: 184}) |> reframe(11))

      assert ScenarioScan.tech_chase(fs) == []
    end
  end

  describe "edgeguard/2" do
    test "detects P2 offstage with P1 onstage" do
      fs =
        frames(30, %{x: 40.0}, %{x: 60.0}) ++
          (frames(30, %{x: 40.0}, %{x: 95.0, on_ground: false}) |> reframe(31))

      assert [%{type: :edgeguard, frame: 31}] = ScenarioScan.edgeguard(fs)
    end

    test "no candidate when P1 is offstage too" do
      fs = frames(60, %{x: -95.0, y: -20.0}, %{x: 95.0})
      assert ScenarioScan.edgeguard(fs) == []
    end

    test "below-stage P2 counts as offstage even inside |x| bounds" do
      fs =
        frames(30, %{x: 0.0}, %{x: 50.0}) ++
          (frames(30, %{x: 0.0}, %{x: 50.0, y: -30.0}) |> reframe(31))

      assert [%{frame: 31}] = ScenarioScan.edgeguard(fs)
    end
  end

  describe "getup/1" do
    test "detects P1 entering missed-tech knockdown" do
      fs =
        frames(20, %{action: 88}) ++
          (frames(10, %{action: @down_bound_u}) |> reframe(21))

      assert [%{type: :getup, frame: 21}] = ScenarioScan.getup(fs)
    end

    test "a tech is not a getup scenario (no decision pending)" do
      fs =
        frames(20, %{action: 88}) ++
          (frames(10, %{action: @tech_in_place}) |> reframe(21))

      assert ScenarioScan.getup(fs) == []
    end
  end

  describe "idle_deadlock/2" do
    test "requires 90 consecutive mutual-Wait frames; candidate 90 frames in" do
      fs =
        frames(10, %{action: @dash}, %{action: @dash}) ++
          (frames(150, %{action: @wait}, %{action: @wait}) |> reframe(11))

      assert [%{type: :idle_deadlock, frame: 100, note: note}] = ScenarioScan.idle_deadlock(fs)
      assert note =~ "run=150f"
    end

    test "89 frames is not a deadlock" do
      fs =
        frames(89, %{action: @wait}, %{action: @wait}) ++
          (frames(30, %{action: @dash}, %{action: @wait}) |> reframe(90))

      assert ScenarioScan.idle_deadlock(fs) == []
    end
  end

  describe "scan/2 bounds and spacing" do
    test "drops candidates before min_frame and after max_frac of the game" do
      # Behind situation living at frames 1..80 (too early) and one at 400+
      fs =
        Enum.map(1..80, &frame(&1, %{facing: 1}, %{x: -15.0})) ++
          Enum.map(81..399, &frame(&1, %{facing: 1}, %{x: 60.0})) ++
          Enum.map(400..500, &frame(&1, %{facing: 1}, %{x: -15.0})) ++
          Enum.map(501..900, &frame(&1, %{facing: 1}, %{x: 60.0}))

      cands = ScenarioScan.scan(fs, types: [:opponent_behind])
      assert Enum.map(cands, & &1.frame) == [400]
    end

    test "same-type candidates are spaced by the gap" do
      # Two behind onsets 100 frames apart; gap 240 keeps only the first
      fs =
        Enum.map(1..400, &frame(&1, %{facing: 1}, %{x: 60.0})) ++
          Enum.map(401..430, &frame(&1, %{facing: 1}, %{x: -15.0})) ++
          Enum.map(431..500, &frame(&1, %{facing: 1}, %{x: 60.0})) ++
          Enum.map(501..530, &frame(&1, %{facing: 1}, %{x: -15.0})) ++
          Enum.map(531..2000, &frame(&1, %{facing: 1}, %{x: 60.0}))

      cands = ScenarioScan.scan(fs, types: [:opponent_behind])
      assert Enum.map(cands, & &1.frame) == [401]

      cands = ScenarioScan.scan(fs, types: [:opponent_behind], gap: 50)
      assert Enum.map(cands, & &1.frame) == [401, 501]
    end
  end

  describe "load/1 + scan/2 (fixture integration)" do
    @tag :fixture
    test "loads a committed scenario replay and finds the curated moments" do
      assert {:ok, %{frames: frames}} = ScenarioScan.load(@fixture)
      assert length(frames) > 5_000

      # The manifest entries curated from this file must keep being found.
      cands = ScenarioScan.scan(frames)
      by_type = Enum.group_by(cands, & &1.type, & &1.frame)

      assert 880 in by_type[:opponent_behind]
      assert 1157 in by_type[:tech_chase]
      assert 1453 in by_type[:edgeguard]
      assert 4970 in by_type[:idle_deadlock]
    end

    @tag :fixture
    test "load reports unparseable files as errors" do
      assert {:error, _} = ScenarioScan.load("mix.exs")
    end
  end

  # Renumber a frame block so appended segments carry consecutive frame ids.
  defp reframe(fs, start) do
    Enum.with_index(fs, fn f, i -> %{f | frame: start + i} end)
  end
end
