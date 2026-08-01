defmodule ExPhil.Agents.LiveQueueTest do
  @moduledoc """
  Live-side queue-as-input invariants (2026-08-01, first Direct games).

  Two live-only corruptions the sync harness structurally cannot catch
  (calls == frames and ports are static there):

    * async runs >1 inference per game frame, so per-DECISION ring pushes
      warp slot k into "k calls ago" instead of "k frames ago"
    * Slippi Online assigns in-game ports per session — the static --port
      guess ego-swaps the embedding when the bot draws the other port
      (observed: policy saw the human as "self", degenerated into hold-B)
  """
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent

  # Distinguishable stand-in "controllers" — the ring never inspects them
  defp c(n), do: {:ctrl, n}

  describe "update_controller_queue/5 (frame-gated ring)" do
    test "fresh ring seeds with the single new controller" do
      assert Agent.update_controller_queue([], c(1), 1, false, 4) == [c(1)]
    end

    test "frame_delta 1 is a plain push, newest first, truncated to depth" do
      q = Agent.update_controller_queue([c(3), c(2), c(1)], c(4), 1, false, 3)
      assert q == [c(4), c(3), c(2)]
    end

    test "frame_delta 0 (same-frame re-inference) replaces the head" do
      q = Agent.update_controller_queue([c(3), c(2), c(1)], c(4), 0, false, 4)
      assert q == [c(4), c(2), c(1)]
    end

    test "repeated same-frame re-inference never grows the ring" do
      q0 = [c(2), c(1)]

      q =
        Enum.reduce(3..10, q0, fn n, acc ->
          Agent.update_controller_queue(acc, c(n), 0, false, 4)
        end)

      # Only the head churned; frame history depth is unchanged
      assert q == [c(10), c(1)]
    end

    test "frame gap fills the skipped slots with the HELD previous input" do
      # 3 frames passed since the last decision: the game held c(3) for the
      # 2 undecided frames, then c(4) lands on the current frame
      q = Agent.update_controller_queue([c(3), c(2), c(1)], c(4), 3, false, 4)
      assert q == [c(4), c(3), c(3), c(3)]
    end

    test "huge gap saturates at depth without raising" do
      q = Agent.update_controller_queue([c(1)], c(2), 100, false, 4)
      assert q == [c(2), c(1), c(1), c(1)]
    end

    test "game reset drops the old game's ring entirely" do
      q = Agent.update_controller_queue([c(3), c(2), c(1)], c(4), 1, true, 4)
      assert q == [c(4)]
    end
  end

  describe "own-port detection plumbing" do
    test "GameState carries own_port from the bridge payload (nil offline)" do
      assert %ExPhil.Bridge.GameState{}.own_port == nil
      assert %ExPhil.Bridge.GameState{own_port: 2}.own_port == 2
    end

    test "Player carries connect_code" do
      assert %ExPhil.Bridge.Player{connect_code: "EXPH#288"}.connect_code == "EXPH#288"
    end
  end
end
