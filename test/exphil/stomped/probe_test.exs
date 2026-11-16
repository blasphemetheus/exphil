defmodule ExPhil.Stomped.ProbeTest do
  use ExUnit.Case, async: true

  alias ExPhil.Stomped.Bots.Probe
  alias ExPhil.Stomped.Pad

  test "a pad encodes its target frame and decodes to offset 0 at that frame" do
    for t <- [-123, 0, 49, 50, 1234, 99_999] do
      pad = Probe.pad(t)
      # The echo is the pad after the game's stick processing and our inverse of it.
      echo = Pad.from_controller(controller(pad))
      assert Probe.offset(echo, t) == 0
      assert Probe.offset(echo, t + 1) == 1
      assert Probe.offset(echo, t + 7) == 7
    end
  end

  test "neutral and foreign sticks are told apart from late pads" do
    assert Probe.offset(%{stick_x: 128}, 10) == :neutral
    assert Probe.offset(%{stick_x: 128 + 10}, 10) == :other
    assert Probe.offset(%{stick_x: 128 + 80}, 10) == :other
  end

  test "50 consecutive targets have 50 distinct codes" do
    codes = for t <- 1000..1049, do: Probe.pad(t).stick_x
    assert length(Enum.uniq(codes)) == 50
    assert Enum.all?(codes, &(&1 - 128 >= 24 and &1 - 128 <= 73))
  end

  test "with every field encoded, each field decodes to its own offset" do
    for t <- [-120, 7, 1234] do
      pad = Probe.pad(t, :all)
      assert Probe.offsets(pad, t) == %{main: 0, c: 0, button: 0, trigger: 0}
      assert Probe.offsets(pad, t + 1) == %{main: 1, c: 1, button: 1, trigger: 1}
      assert Probe.offsets(pad, t + 2).button == 0
      assert Probe.offsets(pad, t + 2).main == 2
    end

    # A main-only pad reads neutral on the other fields.
    assert %{c: :neutral, trigger: :neutral} =
             Probe.offsets(Map.merge(%{c_x: 128, buttons: 0, l: 0}, Probe.pad(5)), 5)
  end

  # The controller the decoder would hand back for a pad the game read.
  defp controller(%{stick_x: x, stick_y: y}) do
    %ExPhil.Bridge.ControllerState{
      main_stick: %{x: (x - 128) / 80 / 2 + 0.5, y: (y - 128) / 80 / 2 + 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: 0.0,
      r_shoulder: 0.0
    }
  end
end
