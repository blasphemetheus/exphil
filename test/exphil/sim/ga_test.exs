defmodule ExPhil.Sim.GATest do
  @moduledoc """
  The static-checkmate term of the GA fitness (M4, 2026-09-25): `checkmate_setup/4` asks
  `ExPhil.Melee.Checkmate` once, at P2's first actionable offstage frame after the chain.
  Pure functions on synthetic states — no sim, no GPU.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Sim.GA

  @fd 32
  @edge 85.5

  defp player(over) do
    Map.merge(
      %{
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        action: 14,
        facing: 1,
        on_ground: true,
        hitstun_frames_left: 0,
        jumps_left: 1,
        speed_air_x_self: 0.0,
        speed_y_self: 0.0,
        speed_x_attack: 0.0,
        speed_y_attack: 0.0
      },
      over
    )
  end

  defp state(frame, p1, p2), do: %{frame: frame, players: %{1 => player(p1), 2 => player(p2)}}

  # P1 stands on stage; P2 is hit on frame 10 (percent rises), then is airborne offstage at `p2`
  # from frame 11 on, in hitstun for `stun` frames after the hit.
  defp rollout(p2, opts) do
    stun = Keyword.get(opts, :stun, 5)
    frames = Keyword.get(opts, :frames, 60)

    for f <- 0..frames do
      cond do
        f < 10 -> state(f, %{}, %{})
        f == 10 -> state(f, %{}, Map.merge(%{percent: 40.0, on_ground: false, hitstun_frames_left: stun}, p2))
        true -> state(f, %{}, Map.merge(%{percent: 40.0, on_ground: false, hitstun_frames_left: max(0, stun - (f - 10))}, p2))
      end
    end
  end

  describe "checkmate_state/2" do
    test "splits self and knockback velocity and keeps a real facing" do
      p = player(%{x: 120.0, y: -30.0, speed_air_x_self: 0.5, speed_y_self: -1.0, speed_x_attack: 2.0, speed_y_attack: 1.5, jumps_left: 0, facing: -1})
      st = GA.checkmate_state(p, @fd)
      assert st == %{stage: @fd, x: 120.0, y: -30.0, vx: 0.5, vy: -1.0, kb_vx: 2.0, kb_vy: 1.5, jumps_left: 0, facing: -1}
    end

    test "a missing or zero facing becomes nil so the model faces the stage" do
      assert GA.checkmate_state(player(%{facing: 0}), @fd).facing == nil
      assert GA.checkmate_state(Map.delete(player(%{}), :facing), @fd).facing == nil
      assert GA.checkmate_state(player(%{speed_y_self: nil, jumps_left: nil}), @fd) |> Map.take([:vy, :jumps_left]) == %{vy: 0.0, jumps_left: 0}
    end
  end

  describe "checkmate_setup/4" do
    test "far offstage with nothing to recover with is checkmate at the first actionable frame" do
      states = rollout(%{x: 250.0, y: 0.0, jumps_left: 0}, stun: 5)
      r = GA.checkmate_setup(states, 10, @edge, @fd)
      assert r.checkmate?
      # hit on frame 10 with 5 f of hitstun -> first actionable frame is 15, not 11
      assert r.frame == 15
      assert r.state.x == 250.0
    end

    test "just past the ledge at ledge height is not checkmate (the ledge is reachable)" do
      states = rollout(%{x: @edge + 8.0, y: -5.0, jumps_left: 0}, stun: 1)
      r = GA.checkmate_setup(states, 10, @edge, @fd)
      refute r.checkmate?
      assert r.frame == 11
      assert r.outcome in [:ledge, :stage, :platform]
    end

    test "evaluates once, at the FIRST actionable frame, even if P2 is checkmated later" do
      # frames 11..30: recoverable just past the ledge; frames 31+: way out. The model is asked at
      # frame 11 only — drifting away afterwards is the defender's choice, not the setup.
      near = %{x: @edge + 8.0, y: -5.0, jumps_left: 0}
      states = rollout(near, stun: 1) |> Enum.map(fn s -> if s.frame > 30, do: put_in(s, [:players, 2, :x], 250.0), else: s end)
      r = GA.checkmate_setup(states, 10, @edge, @fd)
      refute r.checkmate?
      assert r.frame == 11
    end

    test "no recent hit means no evaluation: a defender who jumps off on its own is not a setup" do
      states = for f <- 0..60, do: state(f, %{}, %{x: 250.0, y: 0.0, on_ground: false, jumps_left: 0})
      assert GA.checkmate_setup(states, 10, @edge, @fd) == %{checkmate?: false, frame: nil, outcome: nil, state: nil}
    end

    test "still in hitstun through the whole window means no evaluation" do
      states = rollout(%{x: 250.0, y: 0.0, jumps_left: 0}, stun: 200)
      assert GA.checkmate_setup(states, 10, @edge, @fd).frame == nil
    end

    test "on-stage airborne P2 is not offstage and is not evaluated" do
      states = rollout(%{x: 20.0, y: 30.0, jumps_left: 0}, stun: 0)
      assert GA.checkmate_setup(states, 10, @edge, @fd).frame == nil
    end

    test "without a chain the last 45 frames are searched" do
      states = rollout(%{x: 250.0, y: 0.0, jumps_left: 0}, stun: 0, frames: 200)
      r = GA.checkmate_setup(states, nil, @edge, @fd)
      assert r.checkmate?
      # start = 201 - 45 = 156
      assert r.frame == 156
    end
  end
end
