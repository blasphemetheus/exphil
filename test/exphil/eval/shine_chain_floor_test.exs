defmodule ExPhil.Eval.ShineChainFloorTest do
  @moduledoc """
  The assertion GOTCHAS #79 says every metric needs, applied to the metric
  that taught us the lesson: does `ShineChain` score a REAL multishine
  highly?

  `shine_chain_test.exs` covers the semantics on synthetic action lists.
  Nothing before this asserted the metric against a recording of the actual
  technique — which is exactly the gap that let v2 ship while scoring every
  correct execution as "max chain 1".
  """
  use ExUnit.Case, async: true

  alias ExPhil.Eval.ShineChain
  alias ExPhil.Test.MetricFloor

  @good "test/fixtures/replays/fox_multishine_closed.slp"

  # The technique's real frame shape (#79's actual instruction: write it
  # down, then check the metric against it — not against its own
  # assumptions). Measured in MULTISHINE_TRACE dumps: each cycle is a
  # ~9-frame loop of 2 grounded reflector frames + a 3-frame jumpsquat +
  # an airborne reflector, i.e. only ~1/3 of frames are grounded. Any
  # metric demanding mostly-grounded frames rejects every correct attempt.
  @shape "~9-frame cycle: 2 grounded reflector frames + 3f jumpsquat + airborne reflector (~1/3 grounded)"

  defp own_actions(path) do
    path
    |> MetricFloor.frames!()
    |> Enum.map(& &1.game_state.players[1].action)
  end

  describe "known-good execution scores highly (the #79 floor)" do
    test "the canonical fixture chains well past the B0 gate" do
      chains = @good |> own_actions() |> ShineChain.chains()
      best = Enum.max(chains, fn -> 0 end)

      # GOALS.md gate B0 asks >= 50; the fixture measured 186 when it was
      # recorded. The floor is set at the GATE, not at the measurement, so
      # a real regression trips it but ordinary recording variance does not.
      MetricFloor.assert_floor(best, 50,
        metric: "ShineChain.chains max",
        subject: "the canonical multishine fixture (#{Path.basename(@good)})",
        shape: @shape
      )
    end

    test "grounded fraction sits in the technique's real band, not near 1.0" do
      frames = MetricFloor.frames!(@good)

      grounded =
        Enum.count(frames, fn f -> f.game_state.players[1].on_ground end) / max(length(frames), 1)

      # The v2 metric implicitly demanded >= 0.9 here. Pinning the real band
      # documents WHY that target was unreachable, so nobody re-derives it.
      assert grounded < 0.9,
             "grounded fraction #{Float.round(grounded, 3)} — if this is ever ~0.9+, the " <>
               "fixture is not a multishine (see GOTCHAS #79/#78)"
    end
  end

  describe "the metric discriminates" do
    test "a full-jump loop scores far below the real technique" do
      real = @good |> own_actions() |> ShineChain.chains() |> Enum.max(fn -> 0 end)

      # Synthetic sloppy loop: grounded reflector -> FULL jump (long air gap,
      # no aerial shine) -> land. This is the 2026-07-23 fixture's shape, the
      # thing v2 was tuned to reject; v3 must still reject it.
      ground_reflect = ExPhil.Constants.reflector_ground() |> Enum.at(0)
      jumpsquat = ExPhil.Constants.jumpsquat()
      aerial_jump = ExPhil.Constants.aerial_jump()

      sloppy =
        List.duplicate(
          [ground_reflect, ground_reflect, jumpsquat] ++ List.duplicate(aerial_jump, 20),
          8
        )
        |> List.flatten()

      sloppy_best = sloppy |> ShineChain.chains() |> Enum.max(fn -> 0 end)

      MetricFloor.assert_discriminates(real, sloppy_best,
        min_ratio: 5.0,
        metric: "ShineChain.chains max (real fixture vs full-jump loop)"
      )
    end
  end
end
