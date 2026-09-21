defmodule ExPhil.Sim.SearchStyleTest do
  use ExUnit.Case, async: true
  alias ExPhil.Sim.Search

  defp gs(action, y, on_ground), do: %{players: %{1 => %{action: action, y: y, on_ground: on_ground, x: 0.0}}}

  test "counts a full hop but not a short hop; counts grab starts and shield frames" do
    short = [gs(24, 0.0, true), gs(25, 3.0, false), gs(29, 10.0, false), gs(30, 14.0, false), gs(30, 8.0, false), gs(42, 0.0, true)]
    full = [gs(24, 0.0, true), gs(25, 5.0, false), gs(29, 20.0, false), gs(30, 35.0, false), gs(30, 41.0, false), gs(30, 30.0, false), gs(42, 0.0, true)]
    grab_shield = [gs(14, 0.0, true), gs(212, 0.0, true), gs(213, 0.0, true), gs(14, 0.0, true), gs(179, 0.0, true), gs(179, 0.0, true), gs(180, 0.0, true)]
    assert Search.style_counts(short) == %{full_hops: 0, grabs: 0, shield_frames: 0}
    assert Search.style_counts(full) == %{full_hops: 1, grabs: 0, shield_frames: 0}
    assert Search.style_counts(grab_shield) == %{full_hops: 0, grabs: 1, shield_frames: 3}
  end
end
