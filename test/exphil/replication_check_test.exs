defmodule ExPhil.Test.ReplicationCheckTest do
  @moduledoc """
  Pure unit tests for the overfit-replication checker. No model, no GPU — this
  verifies the strictness logic and the fixture/checker agreement so the next
  session can trust piece 2 before wiring the training harness (piece 3).
  """
  use ExUnit.Case, async: true

  import ExPhil.Test.ReplayFixtures
  alias ExPhil.Test.ReplicationCheck
  alias ExPhil.Bridge.ControllerState

  # Ground-truth controllers from the synthetic multishine fixture.
  defp expected_controllers(opts \\ []) do
    tech_fixture(:multishine, opts) |> Enum.map(&elem(&1, 1))
  end

  describe "fixture <-> checker agreement" do
    test "the multishine fixture emits one shine per period" do
      frames = 64
      period = 8
      controllers = expected_controllers(frames: frames, period: period)

      idx = ReplicationCheck.shine_indices(controllers)
      assert length(idx) == div(frames, period)
      # Shines land on phase 0 → indices 0, period, 2*period, ...
      assert idx == Enum.map(0..(div(frames, period) - 1), &(&1 * period))
    end

    test "the fixture state varies with shine phase (test-validity guard)" do
      # If states were constant, no correct model could reproduce a periodic
      # output — this asserts the fixture avoids that trap.
      states = tech_fixture(:multishine, frames: 16, period: 8) |> Enum.map(&elem(&1, 0))
      fox_actions = Enum.map(states, fn gs -> gs.players[2].action end)
      assert Enum.uniq(fox_actions) |> length() > 1, "Fox action-state must vary across the shine cycle"
    end
  end

  describe ":exact" do
    test "identical sequence passes" do
      exp = expected_controllers()
      assert {:ok, %{pass: true}} = ReplicationCheck.check(exp, exp, strictness: :exact)
    end

    test "a single flipped button fails with the offending frame" do
      exp = expected_controllers()
      # Flip button_a on frame 3.
      actual = List.update_at(exp, 3, fn c -> %ControllerState{c | button_a: not c.button_a} end)

      assert {:error, %{pass: false, first_mismatch_frame: 3}} =
               ReplicationCheck.check(exp, actual, strictness: :exact)
    end

    test "length mismatch fails" do
      exp = expected_controllers()
      assert {:error, %{message: "length mismatch" <> _}} =
               ReplicationCheck.check(exp, Enum.drop(exp, 1), strictness: :exact)
    end
  end

  describe ":periodic" do
    test "a one-frame global shift still passes (robust to jitter)" do
      exp = expected_controllers()
      # Rotate left by one frame: every shine event moves by 1, within tolerance.
      actual = tl(exp) ++ [hd(exp)]

      assert {:ok, %{pass: true}} =
               ReplicationCheck.check(exp, actual, strictness: :periodic, period_tolerance: 1)
    end

    test "emitting zero shines fails" do
      exp = expected_controllers()
      actual = Enum.map(exp, fn _ -> neutral_controller() end)

      assert {:error, %{pass: false, actual_shines: []}} =
               ReplicationCheck.check(exp, actual, strictness: :periodic)
    end

    test "a two-frame shift exceeds a one-frame tolerance" do
      exp = expected_controllers()
      actual = Enum.drop(exp, 2) ++ Enum.take(exp, 2)

      assert {:error, %{pass: false}} =
               ReplicationCheck.check(exp, actual, strictness: :periodic, period_tolerance: 1)
    end
  end

  describe ":loose" do
    test "same shine count passes" do
      exp = expected_controllers()
      actual = tl(exp) ++ [hd(exp)]
      assert {:ok, %{pass: true}} = ReplicationCheck.check(exp, actual, strictness: :loose)
    end

    test "far-off shine count fails" do
      exp = expected_controllers()
      actual = Enum.map(exp, fn _ -> neutral_controller() end)
      assert {:error, %{pass: false}} =
               ReplicationCheck.check(exp, actual, strictness: :loose, count_tolerance: 1)
    end
  end

  describe "infer_period/1" do
    test "median gap between shines" do
      assert ReplicationCheck.infer_period([0, 8, 16, 24]) == 8
      assert ReplicationCheck.infer_period([]) == nil
      assert ReplicationCheck.infer_period([5]) == nil
    end
  end
end
