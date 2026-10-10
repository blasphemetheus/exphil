defmodule ExPhil.Eval.RecoveryMeansTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.RecoveryMeans

  @edge 85.5

  defp p(over) do
    Map.merge(%{x: 0.0, y: 0.0, on_ground: true, jumps_left: 1, action: 14, percent: 0.0, stock: 4, hitstun_frames_left: 0}, over)
  end

  defp frame(own), do: %{own: own, opp: p(%{}), controller: nil}

  # offstage trip: launched (hitstun) -> actionable at (x, y) -> path actions -> returned or died
  defp trip(x, y, jumps, path, outcome) do
    launched = frame(p(%{x: x, y: y, on_ground: false, jumps_left: jumps, hitstun_frames_left: 5, percent: 40.0}))
    first = frame(p(%{x: x, y: y, on_ground: false, jumps_left: jumps, percent: 40.0}))

    middle =
      Enum.map(path, fn {action, j} ->
        frame(p(%{x: x, y: y, on_ground: false, jumps_left: j, action: action, percent: 40.0}))
      end)

    last =
      case outcome do
        :returned -> frame(p(%{x: 0.0, y: 0.0, on_ground: true, percent: 40.0}))
        :died -> frame(p(%{x: x, y: -200.0, on_ground: false, percent: 40.0, stock: 3}))
      end

    [frame(p(%{})), launched, first] ++ middle ++ [last, frame(p(%{stock: if(outcome == :died, do: 3, else: 4)}))]
  end

  test "episodes: situation at the decision frame, first means, outcome" do
    frames =
      trip(110.0, -40.0, 1, [{29, 1}, {350, 1}], :died) ++
        trip(100.0, 10.0, 1, [{29, 0}, {353, 0}], :returned) ++
        trip(150.0, -10.0, 0, [{236, 0}], :died)

    eps = RecoveryMeans.episodes(frames, @edge)
    assert length(eps) == 3

    [a, b, c] = eps
    assert {a.height, a.dist, a.jumps, a.first, a.outcome} == {:low, :mid, 1, :side_b, :died}
    assert {b.height, b.dist, b.jumps, b.first, b.seq, b.outcome} == {:high, :near, 1, :jump, "jump>up_b", :returned}
    assert {c.height, c.dist, c.jumps, c.first, c.outcome} == {:ledge, :far, 0, :airdodge, :died}
    assert RecoveryMeans.bucket(a) == "low/mid/j1"
  end

  test "score: mismatch against a reference that never side-Bs low, named defects, split-half" do
    expert = for _ <- 1..30, do: trip(110.0, -40.0, 1, [{29, 0}, {353, 0}], :returned)
    expert_eps = RecoveryMeans.episodes(Enum.concat(expert), @edge)
    table = RecoveryMeans.table(expert_eps)
    assert table["buckets"]["low/mid/j1"]["jump"] == 30

    bot_eps =
      RecoveryMeans.episodes(
        Enum.concat([trip(110.0, -40.0, 1, [{350, 1}], :died), trip(110.0, -40.0, 1, [{29, 0}], :returned)]),
        @edge
      )

    s = RecoveryMeans.score(bot_eps, table)
    assert s["episodes"] == 2
    assert s["mismatch_rate"] == 0.5
    assert s["side_b_low"] == 0.5
    assert s["return_rate"] == 0.5
    assert s["means_js"] > 0.0

    floor = RecoveryMeans.split_half(expert_eps)
    assert floor["mismatch_rate"] == 0.0
  end

  test "the dead body below the blast zone is not an offstage trip" do
    # sim/replay order: stock drops on crossing, then DEAD_DOWN (action 0) sits at
    # y ~ -141 for ~59 frames, then the revival platform
    before = trip(110.0, -40.0, 1, [{350, 1}], :died)
    dead = for _ <- 1..59, do: frame(p(%{x: 110.0, y: -141.0, on_ground: false, action: 0, stock: 3}))
    rebirth = frame(p(%{x: 0.0, y: 40.0, on_ground: false, action: 12, stock: 3}))
    eps = RecoveryMeans.episodes(before ++ dead ++ [rebirth, frame(p(%{stock: 3}))], @edge)
    assert Enum.map(eps, &{&1.height, &1.first, &1.outcome}) == [{:low, :side_b, :died}]
  end

  test "pre_trace keeps the 90 frames before the decision frame (every 3rd, oldest first, newest last); the approach fields read the last 30" do
    # 100 grounded frames at the edge; the double jump is spent 50 frames before the trip (jumps 1 -> 0 at an
    # aerial action), then a carried-off side-B trip with no jump
    idle = for i <- 0..49, do: frame(p(%{x: 80.0 + i * 0.1, action: 14}))
    spent = for _ <- 0..49, do: frame(p(%{x: 85.0, y: 10.0, on_ground: false, jumps_left: 0, action: 27}))
    launched = frame(p(%{x: 90.0, y: 5.0, on_ground: false, jumps_left: 0, action: 350, percent: 0.0}))
    path = for _ <- 1..5, do: frame(p(%{x: 120.0, y: -30.0, on_ground: false, jumps_left: 0, action: 350}))
    dead = frame(p(%{x: 120.0, y: -200.0, on_ground: false, jumps_left: 0, stock: 3}))
    [ep] = RecoveryMeans.episodes(idle ++ spent ++ [launched] ++ path ++ [dead, frame(p(%{stock: 3}))], @edge)
    assert ep.jumps == 0 and ep.first == :side_b and ep.first_at == 0
    assert length(ep.pre_trace) == 30
    # row = [action, x, y, speed_y, stick_x, stick_y, b, jump, jumps_left]; the newest row is the frame before the decision frame
    assert List.last(ep.pre_trace) |> Enum.at(0) == 27
    assert List.last(ep.pre_trace) |> Enum.at(8) == 0
    # the oldest rows are the grounded approach with the jump still in hand
    assert List.first(ep.pre_trace) |> Enum.at(0) == 14
    assert List.first(ep.pre_trace) |> Enum.at(8) == 1
    # the approach fields still read the last 30 frames only
    assert ep.pre_actions == [27]
    # a trip with a short history has a short pre_trace
    [short] = RecoveryMeans.episodes(trip(110.0, -40.0, 1, [{29, 1}, {350, 1}], :died), @edge)
    assert length(short.pre_trace) <= 2
  end

  test "an episode with no actionable frame (died in stun) is not scored" do
    stun = frame(p(%{x: 120.0, y: -30.0, on_ground: false, hitstun_frames_left: 9, percent: 90.0}))
    dead = frame(p(%{x: 120.0, y: -200.0, on_ground: false, percent: 90.0, stock: 3}))
    assert RecoveryMeans.episodes([frame(p(%{})), stun, stun, dead, frame(p(%{stock: 3}))], @edge) == []
  end
end
