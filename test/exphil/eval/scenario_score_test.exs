defmodule ExPhil.Eval.ScenarioScoreTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.ScenarioScore

  # Action-state constants (mirror the module under test so semantic drift
  # breaks loudly here rather than silently re-deriving).
  @wait 14
  @dash 20
  @jab 44
  @fair 66
  @grab 212
  @down_bound_u 183
  @down_wait_u 184
  @down_stand_u 186
  @down_attack_u 187
  @tech_in_place 199

  defp player(overrides \\ %{}) do
    Map.merge(
      %{x: 0.0, y: 0.0, action: @wait, facing: 1, on_ground: true, stock: 4},
      overrides
    )
  end

  defp frame(n, p1_over, p2_over) do
    %{frame: n, p1: player(p1_over), p2: player(p2_over)}
  end

  # Window of `n` frames where both players idle in place.
  defp static_window(n, p1_over \\ %{}, p2_over \\ %{}) do
    Enum.map(1..n, &frame(&1, p1_over, p2_over))
  end

  describe "opponent_behind" do
    # Fox 20 units behind Mewtwo (P1 faces right, Fox at x=-20)
    setup do
      %{handoff: %{p1: player(%{x: 0.0, facing: 1}), p2: player(%{x: -20.0})}}
    end

    test "pass: turns to face the opponent within 60f", %{handoff: handoff} do
      window =
        static_window(10, %{facing: 1}, %{x: -20.0}) ++
          static_window(50, %{facing: -1}, %{x: -20.0})

      r = ScenarioScore.score(:opponent_behind, handoff, window)

      assert r.pass
      assert r.score > 0.9
      assert r.details.frames_to_response == 10
      assert r.details.opponent_side == -1
    end

    test "pass: up/down smash counts without turning", %{handoff: handoff} do
      window =
        static_window(5, %{facing: 1}, %{x: -20.0}) ++
          static_window(20, %{facing: 1, action: 64}, %{x: -20.0})

      r = ScenarioScore.score(:opponent_behind, handoff, window)
      assert r.pass
    end

    test "fail: never turns", %{handoff: handoff} do
      window = static_window(300, %{facing: 1}, %{x: -20.0})
      r = ScenarioScore.score(:opponent_behind, handoff, window)

      refute r.pass
      assert r.score == 0.0
      assert r.details.frames_to_response == nil
    end

    test "late turn scores partial credit without passing", %{handoff: handoff} do
      window =
        static_window(100, %{facing: 1}, %{x: -20.0}) ++
          static_window(50, %{facing: -1}, %{x: -20.0})

      r = ScenarioScore.score(:opponent_behind, handoff, window)
      refute r.pass
      assert r.score == 0.25
    end
  end

  describe "tech_chase" do
    setup do
      %{handoff: %{p1: player(%{x: 0.0}), p2: player(%{x: 30.0, action: @down_bound_u})}}
    end

    test "pass: close at P2's first actionable frame, attacks within 30f", %{handoff: handoff} do
      window =
        # P2 down for 20 frames while P1 closes in
        Enum.map(1..20, fn n ->
          frame(n, %{x: n * 1.0, action: @dash}, %{x: 30.0, action: @down_wait_u})
        end) ++
          # P2 stands (actionable) at index 20; P1 at x=20, dist=10
          static_window(5, %{x: 20.0}, %{x: 30.0, action: @wait}) ++
          static_window(20, %{x: 22.0, action: @grab}, %{x: 30.0})

      r = ScenarioScore.score(:tech_chase, handoff, window)

      assert r.pass
      assert r.score == 1.0
      assert r.details.p2_actionable_at == 20
      assert r.details.distance_at_actionable <= 25.0
      assert r.details.frames_to_attack <= 30
    end

    test "fail: too far away and never attacks", %{handoff: handoff} do
      window =
        static_window(30, %{x: -60.0}, %{x: 30.0, action: @down_wait_u}) ++
          static_window(60, %{x: -60.0}, %{x: 30.0, action: @wait})

      r = ScenarioScore.score(:tech_chase, handoff, window)

      refute r.pass
      assert r.score == 0.0
    end

    test "half credit for proximity without an attack", %{handoff: handoff} do
      window =
        static_window(30, %{x: 25.0}, %{x: 30.0, action: @down_wait_u}) ++
          static_window(60, %{x: 25.0}, %{x: 30.0, action: @wait})

      r = ScenarioScore.score(:tech_chase, handoff, window)

      refute r.pass
      assert r.score == 0.5
    end

    test "P2 never actionable: judged by closest approach + any attack", %{handoff: handoff} do
      window = static_window(50, %{x: 28.0, action: @jab}, %{x: 30.0, action: @down_wait_u})
      r = ScenarioScore.score(:tech_chase, handoff, window)

      assert r.pass
      assert r.details.p2_actionable_at == nil
    end
  end

  describe "edgeguard" do
    setup do
      # Fox offstage right at the ledge, P1 mid-stage
      %{handoff: %{p1: player(%{x: 20.0}), p2: player(%{x: 95.0, y: -20.0, on_ground: false})}}
    end

    test "pass: moves toward the ledge within 60f", %{handoff: handoff} do
      window =
        Enum.map(1..60, fn n -> frame(n, %{x: 20.0 + n * 0.5, action: @dash}, %{x: 95.0}) end)

      r = ScenarioScore.score(:edgeguard, handoff, window)

      assert r.pass
      assert r.details.max_toward_disp >= 10.0
      assert r.details.ledge_side == 1
      refute r.details.p2_stock_lost
      assert r.score == 0.7
    end

    test "stock loss in window adds the bonus", %{handoff: handoff} do
      window =
        Enum.map(1..60, fn n -> frame(n, %{x: 20.0 + n * 0.5}, %{x: 95.0}) end) ++
          [frame(61, %{x: 50.0}, %{x: 95.0, stock: 3})]

      r = ScenarioScore.score(:edgeguard, handoff, window)

      assert r.pass
      assert r.details.p2_stock_lost
      assert r.score == 1.0
    end

    test "fail: retreats from the ledge", %{handoff: handoff} do
      window = Enum.map(1..90, fn n -> frame(n, %{x: 20.0 - n * 0.3}, %{x: 95.0}) end)
      r = ScenarioScore.score(:edgeguard, handoff, window)

      refute r.pass
      assert r.score == 0.0
    end
  end

  describe "getup" do
    setup do
      %{handoff: %{p1: player(%{x: -40.0, action: @down_bound_u}), p2: player(%{x: 0.0})}}
    end

    test "pass: getup option shortly after down-wait begins", %{handoff: handoff} do
      window =
        static_window(5, %{action: @down_bound_u}) ++
          static_window(11, %{action: @down_wait_u}) ++
          static_window(30, %{action: @down_stand_u})

      r = ScenarioScore.score(:getup, handoff, window)

      assert r.pass
      assert r.details.frames_to_act == 11
      assert r.details.response_kind == :getup
      assert r.details.down_wait_runs == 1
      assert r.score > 0.8
    end

    test "pass: bound straight into a getup option (zero down-wait frames)", %{handoff: handoff} do
      window =
        static_window(8, %{action: @down_bound_u}) ++
          static_window(30, %{action: @down_attack_u})

      r = ScenarioScore.score(:getup, handoff, window)

      assert r.pass
      assert r.details.frames_to_act == 0
      assert r.details.response_kind == :getup
      assert r.details.down_wait_runs == 0
    end

    test "pass: teching counts as an instant action", %{handoff: handoff} do
      window = static_window(3, %{action: @down_bound_u}) ++ static_window(30, %{action: @tech_in_place})
      r = ScenarioScore.score(:getup, handoff, window)

      assert r.pass
      assert r.details.frames_to_act == 0
      assert r.details.response_kind == :tech
    end

    test "fail: lies in down-wait past the deadline", %{handoff: handoff} do
      window =
        static_window(5, %{action: @down_bound_u}) ++
          static_window(60, %{action: @down_wait_u}) ++
          static_window(30, %{action: @down_attack_u})

      r = ScenarioScore.score(:getup, handoff, window)

      refute r.pass
      assert r.details.frames_to_act == 60
      assert r.score == 0.25
    end

    test "fail: never reaches an actionable down state", %{handoff: handoff} do
      window = static_window(100, %{action: @down_bound_u})
      r = ScenarioScore.score(:getup, handoff, window)

      refute r.pass
      assert r.score == 0.0
      assert r.details.response_kind == :never_actionable
    end
  end

  describe "idle_deadlock" do
    setup do
      %{handoff: %{p1: player(%{x: -10.0}), p2: player(%{x: 10.0})}}
    end

    test "pass: leaves Wait and closes distance", %{handoff: handoff} do
      window =
        static_window(30, %{x: -10.0}) ++
          Enum.map(1..100, fn n -> frame(30 + n, %{x: -10.0 + n * 0.2, action: @dash}, %{x: 10.0}) end)

      r = ScenarioScore.score(:idle_deadlock, handoff, window)

      assert r.pass
      assert r.details.frames_to_leave_idle == 30
      assert r.details.max_toward_disp >= 5.0
      assert r.score == 1.0
    end

    test "half credit: leaves idle but retreats", %{handoff: handoff} do
      window =
        static_window(10, %{x: -10.0}) ++
          Enum.map(1..100, fn n -> frame(10 + n, %{x: -10.0 - n * 0.2, action: @dash}, %{x: 10.0}) end)

      r = ScenarioScore.score(:idle_deadlock, handoff, window)

      refute r.pass
      assert r.score == 0.5
    end

    test "fail: stands in Wait the whole window", %{handoff: handoff} do
      window = static_window(320, %{x: -10.0})
      r = ScenarioScore.score(:idle_deadlock, handoff, window)

      refute r.pass
      assert r.score == 0.0
      assert r.details.frames_to_leave_idle == nil
    end

    test "attacking in place (fair toward P2) counts as leaving idle but not moving", %{handoff: handoff} do
      window = static_window(300, %{action: @fair, x: -10.0})
      r = ScenarioScore.score(:idle_deadlock, handoff, window)

      refute r.pass
      assert r.score == 0.5
      assert r.details.frames_to_leave_idle == 0
    end
  end

  test "all scores stay in [0, 1] and carry the result shape" do
    handoff = %{p1: player(), p2: player(%{x: 15.0})}
    window = static_window(300)

    for type <- [:opponent_behind, :tech_chase, :edgeguard, :getup, :idle_deadlock] do
      r = ScenarioScore.score(type, handoff, window)
      assert r.score >= 0.0 and r.score <= 1.0
      assert is_boolean(r.pass)
      assert is_map(r.details)
    end
  end
end
