defmodule ExPhil.Eval.PortCheckTest do
  @moduledoc """
  Pins the dummy-port decision (GOTCHAS #57 / #57b) — the check that gates
  every CPU-dummy eval block and had no tests until 2026-08-03, despite the
  bug it guards having silently invalidated the entire combo-drill era of
  recordings (five of six on 2026-07-26 came up HUMAN).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Eval.PortCheck

  defp cpu(port, level), do: %{"port" => port, "type" => 1, "cpu_level" => level}
  defp human(port), do: %{"port" => port, "type" => 0, "cpu_level" => nil}

  describe "verify/2" do
    test "passes when the requested port is a CPU at the requested level" do
      assert PortCheck.verify([human(1), cpu(2, 9)], expect_cpu: 2, expect_level: 9) == :ok
    end

    test "catches THE bug: a HUMAN port where a CPU was requested" do
      assert {:error, :not_cpu, msg} =
               PortCheck.verify([human(1), human(2)], expect_cpu: 2)

      assert msg =~ "NOT a CPU"
      assert msg =~ "#57", "the message must name the gotcha so the reader finds the cause"
    end

    test "catches the autostart race: CPU present but at Melee's default level 1" do
      assert {:error, :wrong_level, msg} =
               PortCheck.verify([human(1), cpu(2, 1)], expect_cpu: 2, expect_level: 9)

      assert msg =~ "level 1"
      assert msg =~ "expected 9"
    end

    test "catches an absent port" do
      assert {:error, :absent, _} = PortCheck.verify([human(1)], expect_cpu: 2)
    end

    test "tolerates a build that does not report cpu_level" do
      # Level is only checked when the replay reports one — some builds
      # return nil, and failing there would reject good recordings.
      assert PortCheck.verify([human(1), cpu(2, nil)], expect_cpu: 2, expect_level: 9) == :ok
    end

    test "no expectation means no check (the default eval path)" do
      assert PortCheck.verify([human(1), human(2)]) == :ok
    end
  end

  describe "type_name/1" do
    test "names every Slippi port type" do
      assert PortCheck.type_name(0) == "HUMAN"
      assert PortCheck.type_name(1) == "CPU"
      assert PortCheck.type_name(2) == "DEMO"
      assert PortCheck.type_name(3) == "empty"
      assert PortCheck.type_name(nil) =~ "type="
    end
  end
end

defmodule ExPhil.Test.AnalogReleaseEdgeTest do
  @moduledoc """
  GOTCHAS #66: an equivalence check is only as strong as its tolerance
  classes. The scenario drift check treated the shield family 178-182 as
  equivalent, so a broken analog RELEASE (EXI inputs latch neutral) scored
  "15/15 exact" while P2 rode shield to break and dizzy every stock.

  `ReplicationCheck` had the same blindness in a different place — it
  compared the digital l/r bits and ignored the analog shoulder axis
  entirely. This pins that a release EDGE is now visible to `:exact`.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Test.ReplicationCheck

  defp ctrl(shoulder) do
    %ControllerState{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: shoulder,
      r_shoulder: 0.0,
      button_a: false,
      button_b: false,
      button_x: false,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false,
      button_d_up: false
    }
  end

  test "a latched analog release is NOT scored as exact" do
    # expected: press then RELEASE. actual: press then LATCH (never releases)
    expected = [ctrl(1.0), ctrl(1.0), ctrl(0.0)]
    latched = [ctrl(1.0), ctrl(1.0), ctrl(1.0)]

    assert {:error, diag} = ReplicationCheck.check(expected, latched, strictness: :exact)

    refute diag.pass,
           "a latched analog release must fail :exact — this is the #66 blindness that " <>
             "turned 'release is broken' into 15/15 exact"
  end

  test "an identical analog stream still passes" do
    stream = [ctrl(1.0), ctrl(1.0), ctrl(0.0)]
    assert {:ok, diag} = ReplicationCheck.check(stream, stream, strictness: :exact)
    assert diag.pass
  end
end
