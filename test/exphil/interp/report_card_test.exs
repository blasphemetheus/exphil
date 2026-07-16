defmodule ExPhil.Interp.ReportCardTest do
  use ExUnit.Case, async: true

  alias ExPhil.Interp.ReportCard

  @fixture "test/fixtures/replays/mewtwo_fair_chains.slp"

  @wait 14
  @jumpsquat 24
  @jump_f 25
  @fall 29
  @dj_f 27
  @shield_hold 179
  @escape_air 235

  defp idle_controller, do: %{button_x: false, button_y: false}

  describe "pct/2 (percentile index convention)" do
    test "empty list is nil" do
      assert ReportCard.pct([], 0.5) == nil
    end

    test "floor(p*len) capped at the last index" do
      assert ReportCard.pct([10, 20], 0.5) == 20
      assert ReportCard.pct([10, 20, 30, 40], 0.5) == 30
      assert ReportCard.pct([10], 0.95) == 10
      assert ReportCard.pct([1, 2, 3], 1.0) == 3
    end
  end

  describe "jump_rate/1" do
    test "counts jumpsquat entries per 100 grounded frames" do
      # 1 jumpsquat entry; airborne frames (24..34) excluded from grounded.
      # grounded = 98 wait frames; rate = 1 * 100 / 98
      actions = List.duplicate(@wait, 50) ++ [@jumpsquat, @jump_f, @fall] ++ List.duplicate(@wait, 48)
      assert_in_delta ReportCard.jump_rate(actions), 100 / 98, 1.0e-9
    end

    test "staying in jumpsquat is one entry, not many" do
      actions = [@wait, @jumpsquat, @jumpsquat, @jumpsquat, @wait]
      # entries = 1, grounded = 2 (the two waits)
      assert ReportCard.jump_rate(actions) == 50.0
    end

    test "zero jumps" do
      assert ReportCard.jump_rate(List.duplicate(@wait, 100)) == 0.0
    end
  end

  describe "dj_stats/1" do
    test "no double jump: 0% and nil p50" do
      actions = [@wait, @jump_f, @fall, @fall, @wait]
      assert ReportCard.dj_stats(actions) == {0.0, nil}
    end

    test "metronome DJ: every stint DJs, immediately" do
      # Two air stints, each rising then DJ at index 1 of the stint
      stint = [@jump_f, @dj_f, @fall]
      actions = [@wait] ++ stint ++ [@wait] ++ stint ++ [@wait]
      assert {100.0, 1} = ReportCard.dj_stats(actions)
    end

    test "patient DJ: p50 measures frames from liftoff" do
      # DJ at stint index 12 (>= 8 passes the gate)
      stint = [@jump_f] ++ List.duplicate(@fall, 11) ++ [@dj_f, @fall]
      actions = [@wait] ++ stint ++ [@wait]
      assert {100.0, 12} = ReportCard.dj_stats(actions)
    end

    test "stints not starting with a rising state are ignored" do
      # Falling-only stint (e.g. walked off a platform)
      actions = [@wait, @fall, @fall, @wait]
      assert ReportCard.dj_stats(actions) == {0.0, nil}
    end
  end

  describe "oos_idle_pct/1" do
    test "shield exit into a jump within 20f is not idle" do
      actions = [@shield_hold, @shield_hold, @wait, @wait, @jumpsquat, @jump_f] ++ List.duplicate(@wait, 20)
      assert ReportCard.oos_idle_pct(actions) == 0.0
    end

    test "shield exit with no follow-up is idle" do
      actions = [@shield_hold, @shield_hold] ++ List.duplicate(@wait, 30)
      assert ReportCard.oos_idle_pct(actions) == 100.0
    end

    test "escape option counts as a follow-up" do
      actions = [@shield_hold, @wait, @escape_air] ++ List.duplicate(@wait, 25)
      assert ReportCard.oos_idle_pct(actions) == 0.0
    end

    test "mixed exits average out" do
      idle_exit = [@shield_hold, @shield_hold] ++ List.duplicate(@wait, 25)
      jump_exit = [@shield_hold, @shield_hold, @jumpsquat, @jump_f] ++ List.duplicate(@wait, 25)
      assert ReportCard.oos_idle_pct(idle_exit ++ jump_exit) == 50.0
    end

    test "no shielding at all: 0.0, no crash" do
      assert ReportCard.oos_idle_pct(List.duplicate(@wait, 50)) == 0.0
    end
  end

  describe "xy_press_p50/1" do
    test "nil when X/Y never pressed" do
      assert ReportCard.xy_press_p50(List.duplicate(idle_controller(), 10)) == nil
    end

    test "short-hop-capable presses measured in frames" do
      controllers =
        List.duplicate(idle_controller(), 3) ++
          List.duplicate(%{button_x: true, button_y: false}, 2) ++
          List.duplicate(idle_controller(), 3) ++
          List.duplicate(%{button_x: false, button_y: true}, 4) ++
          List.duplicate(idle_controller(), 3)

      # press runs: X 2f, Y 4f -> sorted [2,4], p50 index floor(0.5*2)=1 -> 4
      assert ReportCard.xy_press_p50(controllers) == 4
    end

    test "nil controllers (dropped frames) do not crash or count" do
      controllers = [nil, %{button_x: true}, nil]
      assert ReportCard.xy_press_p50(controllers) == 1
    end
  end

  describe "evaluate/1 (gate vector)" do
    test "a disciplined synthetic game passes all 8 gates" do
      # Mostly neutral movement, one patient jump, brief shield with a jump
      # OOS, short X presses, no idle deadlock (break waits with dash 20)
      block = List.duplicate(@wait, 200) ++ [20] ++ List.duplicate(@wait, 200)

      # 3 air stints, exactly one with a DJ (33% <= the 35% gate)
      actions =
        block ++
          [@jumpsquat, @jump_f] ++ List.duplicate(@fall, 10) ++ [@dj_f] ++ List.duplicate(@fall, 3) ++
          block ++
          List.duplicate(@shield_hold, 10) ++ [@jumpsquat, @jump_f, @fall] ++
          block ++
          [@jumpsquat, @jump_f] ++ List.duplicate(@fall, 6) ++
          block

      controllers =
        List.duplicate(idle_controller(), 100) ++
          List.duplicate(%{button_x: true, button_y: false}, 3) ++
          List.duplicate(idle_controller(), length(actions) - 103)

      result = ReportCard.evaluate(%{p1: %{actions: actions, controllers: controllers}})

      failed = result.gates |> Enum.reject(& &1.pass) |> Enum.map(& &1.name)
      assert failed == []
      assert result.passed == 8
      assert result.total == 8
    end

    test "a 215-frame shield run fails exactly the shield gates" do
      base = List.duplicate(@wait, 250) ++ [20] ++ List.duplicate(@wait, 250)
      actions = base ++ List.duplicate(@shield_hold, 215) ++ [@jumpsquat, @jump_f, @fall] ++ base
      controllers = List.duplicate(idle_controller(), length(actions))

      result = ReportCard.evaluate(%{p1: %{actions: actions, controllers: controllers}})
      failed = result.gates |> Enum.reject(& &1.pass) |> Enum.map(& &1.name)

      assert "shield run p95 (f)" in failed
      # 215/931 frames ~ 23% occupancy also trips the occupancy gate
      assert "shield occupancy %" in failed
      refute "max idle streak (f)" in failed
      refute "jumps/100 grounded" in failed
    end

    test "a 301-frame idle streak trips the deadlock detector" do
      base = List.duplicate(@wait, 200) ++ [20] ++ List.duplicate(@wait, 200)
      actions = base ++ List.duplicate(@wait, 301) ++ [20] ++ base
      controllers = List.duplicate(idle_controller(), length(actions))

      result = ReportCard.evaluate(%{p1: %{actions: actions, controllers: controllers}})
      failed = result.gates |> Enum.reject(& &1.pass) |> Enum.map(& &1.name)
      assert failed == ["max idle streak (f)"]
    end

    test "jump metronome trips the jump gates" do
      # Constant jumpsquat->jump->DJ cycling, tiny grounded time between
      cycle = [@wait, @jumpsquat, @jump_f, @dj_f, @fall]
      actions = List.duplicate(cycle, 60) |> List.flatten()
      controllers = List.duplicate(idle_controller(), length(actions))

      result = ReportCard.evaluate(%{p1: %{actions: actions, controllers: controllers}})
      failed = result.gates |> Enum.reject(& &1.pass) |> Enum.map(& &1.name)

      assert "jumps/100 grounded" in failed
      assert "DJ per air stint %" in failed
      assert "liftoff->DJ p50 (f)" in failed
    end
  end

  describe "evaluate_path/1 + score/1 (fixture integration)" do
    @tag :fixture
    test "runs end-to-end on a real replay" do
      result = ReportCard.evaluate_path(@fixture)
      assert result.total == 8
      assert length(result.gates) == 8
      assert result.passed in 0..8
      assert ReportCard.score(@fixture) == result.passed
    end
  end
end
