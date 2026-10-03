defmodule ExPhil.Melee.CheckmateTest do
  use ExUnit.Case, async: true

  alias ExPhil.Melee.Checkmate

  @fd 32
  @bf 31
  @edge 85.5657

  defp fox(over), do: Map.merge(%{stage: @fd, x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, jumps_left: 1, up_b: true, side_b: true, air_dodge: true, wall_jump: true}, over)

  describe "obviously back" do
    test "just past the ledge at ledge height grabs it with nothing" do
      r = Checkmate.analyze(fox(%{x: @edge + 8, y: -5.0, jumps_left: 0, up_b: false, side_b: false, air_dodge: false, wall_jump: false}))
      refute r.checkmate?
      assert r.outcome == :ledge
      assert r.plan == []
    end

    test "above the stage lands on it" do
      r = Checkmate.analyze(fox(%{x: 30.0, y: 40.0, jumps_left: 0, up_b: false, side_b: false, air_dodge: false}))
      refute r.checkmate?
      assert r.outcome == :stage
    end

    test "above a Battlefield side platform lands on the platform" do
      r = Checkmate.analyze(%{stage: @bf, x: 40.0, y: 60.0, jumps_left: 0, up_b: false, side_b: false, air_dodge: false, wall_jump: false})
      refute r.checkmate?
      assert r.outcome in [:platform, :stage]
    end
  end

  describe "resources decide it" do
    test "deep below the ledge: up-B alone falls short, jump then up-B makes it" do
      deep = fox(%{x: @edge + 20, y: -100.0, side_b: false, air_dodge: false, wall_jump: false})
      assert Checkmate.checkmate?(%{deep | jumps_left: 0})
      r = Checkmate.analyze(%{deep | jumps_left: 1})
      refute r.checkmate?
      assert Enum.any?(r.plan, &match?({:jump, _}, &1))
      assert Enum.any?(r.plan, &match?({:up_b, _, _}, &1))
    end

    test "far out at ledge height: nothing is checkmate, side-B brings it back" do
      far = fox(%{x: @edge + 40, y: 0.0, jumps_left: 0, up_b: false, side_b: false, air_dodge: false, wall_jump: false})
      assert Checkmate.checkmate?(far)
      r = Checkmate.analyze(%{far | side_b: true})
      refute r.checkmate?
    end

    test "past the side blast zone is dead" do
      assert Checkmate.checkmate?(fox(%{x: 250.0, y: 0.0}))
    end

    test "below the bottom blast zone is dead whatever is left" do
      assert Checkmate.checkmate?(fox(%{x: 0.0, y: -150.0}))
    end

    # Pokemon Stadium: the short vertical wall at x 73.8 (y -15 to -17.5)
    # under the slanted lip (StageCollision). Below the grab band and inside
    # the ledge, only a wall jump gets Fox up and onto the stage.
    test "a wall jump under the Stadium lip is the only way back" do
      at_wall = fox(%{stage: 3, x: 75.0, y: -16.5, vx: -0.8, jumps_left: 0, up_b: false, side_b: false, air_dodge: false})
      refute Checkmate.checkmate?(at_wall)
      assert Checkmate.checkmate?(%{at_wall | wall_jump: false})
    end
  end

  describe "the state, not the story" do
    test "the same position is the same answer however it was reached" do
      a = fox(%{x: @edge + 60, y: -90.0, jumps_left: 0, side_b: false, air_dodge: false})
      assert Checkmate.analyze(a) == Checkmate.analyze(Map.put(a, :cause, :air_dodged_off))
    end
  end

  describe "the mirrored side" do
    test "left and right are symmetric" do
      right = fox(%{x: @edge + 40, y: -60.0, jumps_left: 0, side_b: false, air_dodge: false})
      left = %{right | x: -right.x}
      assert Checkmate.checkmate?(right) == Checkmate.checkmate?(left)
    end
  end

  describe "search sanity" do
    test "with every resource, anywhere inside the blast zones near the stage is not checkmate" do
      for x <- [@edge + 10, @edge + 40, @edge + 80], y <- [0.0, -40.0, -80.0] do
        refute Checkmate.checkmate?(fox(%{x: x, y: y})), "#{x}, #{y}"
      end
    end

    test "with nothing, anything below the ledge box and past the wall is checkmate" do
      for x <- [@edge + 20, @edge + 60], y <- [-40.0, -80.0] do
        assert Checkmate.checkmate?(fox(%{x: x, y: y, jumps_left: 0, up_b: false, side_b: false, air_dodge: false, wall_jump: false})), "#{x}, #{y}"
      end
    end
  end
end

defmodule ExPhil.Melee.CheckmateCalibrationTest do
  use ExUnit.Case, async: true

  alias ExPhil.Melee.Checkmate

  # Ground truth from the sim, not the eye: scripts/recovery_probe.exs seeds
  # the replay bit-exact at the frame and runs 473 scripted recoveries
  # (jump / Fire Fox at 10 angles and 8 timings / Illusion / air dodge).
  # Coach review 170200_v1 (Battlefield), port 1's last stock (dies f12455).

  test "f12370 (jump already spent, 0/473 recover) is checkmate" do
    assert Checkmate.checkmate?(%{stage: 31, x: -207.7, y: -40.9, vx: 0.55, vy: 3.25, jumps_left: 0})
  end

  # Hitstun ends at f12338 in tumble with a jump left and knockback still
  # carrying Fox up-left (replay speed_x/y_attack). Only jump-now + Fire Fox
  # at 22.5 deg grabs ledge: 2/527 in the sim.
  test "f12338 (first actionable, jump left, 2/527 recover) is not checkmate" do
    refute Checkmate.checkmate?(%{stage: 31, x: -205.5, y: -9.6, vx: 0.04, vy: -2.8, kb_vx: -1.247, kb_vy: 1.204, jumps_left: 1})
  end

  # f8397, port 1's other death: launched high with a jump; 263/473 recover.
  test "f8397 (launched, 263/473 recover) is not checkmate" do
    refute Checkmate.checkmate?(%{stage: 31, x: 166.5, y: 41.5, vx: 0.0, vy: -1.5, jumps_left: 1})
  end
end

defmodule ExPhil.Melee.CheckmateGeometryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Melee.Checkmate

  test "the stage shell is solid inside and open outside" do
    assert Checkmate.solid?(32, 0.0, -20.0)
    refute Checkmate.solid?(32, 0.0, 5.0)
    refute Checkmate.solid?(32, 80.0, -30.0)
    assert Checkmate.solid?(3, 0.0, -100.0)
    refute Checkmate.solid?(31, 0.0, -45.0)
  end

  test "straight below Battlefield, rising, Fox hits the underside instead of passing through" do
    # Directly under the stage centre with only an up-special: the ceiling
    # stops a straight-up path, so the way out has to angle to a ledge.
    r = Checkmate.analyze(%{stage: 31, x: 0.0, y: -60.0, jumps_left: 0, side_b: false, air_dodge: false, wall_jump: false})
    refute r.checkmate?
    assert r.outcome in [:ledge, :stage]
  end

  test "ledges come from the collision data" do
    assert Enum.sort(Enum.map(Checkmate.geometry(31).ledges, &Float.round(elem(&1, 0), 1))) == [-68.4, 68.4]
  end
end

defmodule ExPhil.Melee.CheckmateAirDodgeTest do
  use ExUnit.Case, async: true

  alias ExPhil.Melee.Checkmate

  # melee-sim-light reports/triage/fox_airdodge_ledge.c (2026-09-23): Fox at
  # (-120, 40) off Final Destination's left ledge, facing the stage, with only
  # an air dodge, catches the ledge when dodging toward the stage (0, -20, +20
  # degrees): the dodge lasts 49 f with no grab, and he grabs on the frame it
  # ends or a few frames into the helpless fall.
  test "air dodge so the dodge ends falling into the ledge" do
    r = Checkmate.analyze(%{stage: 32, x: -120.0, y: 40.0, facing: 1, jumps_left: 0, up_b: false, side_b: false, air_dodge: true, wall_jump: false})
    refute r.checkmate?
    assert r.outcome == :ledge
    assert [{:air_dodge, _, _}] = r.plan
    assert Checkmate.checkmate?(%{stage: 32, x: -120.0, y: 40.0, facing: 1, jumps_left: 0, up_b: false, side_b: false, air_dodge: false, wall_jump: false})
  end
end

defmodule ExPhil.Melee.CheckmateRoutesTest do
  use ExUnit.Case, async: true

  alias ExPhil.Melee.Checkmate

  @fd 32
  @edge 85.5657
  @stripped %{stage: @fd, vx: 0.0, vy: 0.0, jumps_left: 0, up_b: false, side_b: false, air_dodge: false, wall_jump: false}

  test "no resources far out: zero routes is checkmate, and agrees with analyze" do
    st = Map.merge(@stripped, %{x: @edge + 40, y: 0.0})
    r = Checkmate.routes(st)
    assert r == %{count: 0, verdict: :checkmate, routes: []}
    assert Checkmate.checkmate?(st)
  end

  test "only side-B to the ledge: one route is forced, timing variants collapse into it" do
    r = Checkmate.routes(Map.merge(@stripped, %{x: @edge + 40, y: 0.0, side_b: true}))
    assert r.verdict == :forced
    assert [%{means: [:side_b], outcome: :ledge, plans: plans, delays: {lo, hi}}] = r.routes
    assert plans > 1
    assert lo <= hi
  end

  test "jump and up-B both in hand near the ledge: a mixup" do
    r = Checkmate.routes(Map.merge(@stripped, %{x: @edge + 20, y: -30.0, jumps_left: 1, up_b: true}))
    assert r.verdict == :mixup
    assert r.count >= 2
    means = Enum.map(r.routes, & &1.means)
    assert [:jump] in means or [:jump, :up_b] in means
    assert Enum.any?(means, &(:up_b in &1))
    assert r.routes == Enum.sort_by(r.routes, &{length(&1.means), &1.fastest})
  end

  test "drift alone arriving is a route with no start delay" do
    r = Checkmate.routes(Map.merge(@stripped, %{x: 30.0, y: 40.0}))
    assert %{means: [], outcome: :stage, delays: nil} = hd(r.routes)
  end
end

defmodule ExPhil.Melee.CheckmateTopBlastTest do
  use ExUnit.Case, async: true

  alias ExPhil.Melee.Checkmate

  # ftCo_800D3158: the top blast zone kills an airborne fighter only while
  # upward knockback exceeds 2.4/f. Sweep row Game_20260918T171514 p1 f10143
  # (Battlefield, y 202.6, knockback up 1.61): the sim recovers 252/527.
  test "above the top blast zone with little knockback left is not death" do
    refute Checkmate.checkmate?(%{stage: 31, x: -88.6, y: 202.6, kb_vx: -0.17, kb_vy: 1.61, jumps_left: 0})
    assert Checkmate.checkmate?(%{stage: 31, x: -88.6, y: 202.6, kb_vx: 0.0, kb_vy: 3.0, jumps_left: 0})
  end
end
