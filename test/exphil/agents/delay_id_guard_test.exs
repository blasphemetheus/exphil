defmodule ExPhil.Agents.DelayIdGuardTest do
  # Untrained delay-id guard (2026-08-24 crown-decider trap): bare
  # --frame-delay 4 silently deployed a delay-conditioned policy at
  # id4 — untrained — and collapsed chaining for three netplay games.
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent

  test "rejects an id outside the checkpoint's trained set" do
    assert Agent.untrained_delay_id?(4, [0, 2, 3], true, false)
    refute Agent.untrained_delay_id?(3, [0, 2, 3], true, false)
    refute Agent.untrained_delay_id?(0, [0, 2, 3], true, false)
  end

  test "nil train_delays falls back to the champion line's known set {0,2,3}" do
    assert Agent.untrained_delay_id?(4, nil, true, false)
    refute Agent.untrained_delay_id?(2, nil, true, false)
  end

  test "explicit override (--delay-id-override) bypasses the guard" do
    refute Agent.untrained_delay_id?(4, [0, 2, 3], true, true)
  end

  test "non-delay-conditioned policies are never guarded" do
    refute Agent.untrained_delay_id?(9, [0], false, false)
  end
end
