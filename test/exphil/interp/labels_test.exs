defmodule ExPhil.Interp.LabelsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Interp.Labels

  # Synthetic 3-frame replay: WAIT, WAIT, DASH. The dash input (full-X) is
  # recorded on the DASH frame (index 2) — Slippi's convention (GOTCHA #113).
  defp bridge_ctrl(x), do: %{main_stick: %{x: x, y: 0.5}, button_a: false, button_b: false, button_x: false, button_y: false, button_z: false, button_l: false, button_r: false, button_d_up: false}
  defp raw_ctrl(x), do: %{main_stick_x: x, main_stick_y: 0.5, button_a: false, button_b: false, button_x: false, button_y: false, button_z: false, button_l: false, button_r: false, button_d_up: false}

  defp training, do: [
    %{game_state: %{players: %{1 => %{action: 14}}}, controller: bridge_ctrl(0.5)},
    %{game_state: %{players: %{1 => %{action: 14}}}, controller: bridge_ctrl(0.5)},
    %{game_state: %{players: %{1 => %{action: 20}}}, controller: bridge_ctrl(1.0)}
  ]

  defp raw, do: [
    %{players: %{2 => %{action: 14, controller: raw_ctrl(0.5)}}},
    %{players: %{2 => %{action: 14, controller: raw_ctrl(0.5)}}},
    %{players: %{2 => %{action: 20, controller: raw_ctrl(1.0)}}}
  ]

  test "issued_input/2 returns the successor frame's controller (training shape)" do
    # From the last WAIT frame (1), the issued input is the dash recorded on frame 2.
    assert Labels.issued_input(training(), 1) |> Labels.full_x?()
    # From frame 0 the issued input is frame 1's neutral controller.
    refute Labels.issued_input(training(), 0) |> Labels.full_x?()
    # Past the end: nil.
    assert Labels.issued_input(training(), 2) == nil
  end

  test "issued_input/3 works on raw NIF frames keyed by port" do
    assert Labels.issued_input(raw(), 1, 2) |> Labels.full_x?()
    refute Labels.issued_input(raw(), 0, 2) |> Labels.full_x?()
    assert Labels.issued_input(raw(), 1, 1) == nil
  end

  test "producing_input/2 is the SAME-frame controller — the leaky one" do
    # On the DASH frame the producing input is the dash; on the WAIT frames it is neutral.
    assert Labels.producing_input(training(), 2) |> Labels.full_x?()
    refute Labels.producing_input(training(), 1) |> Labels.full_x?()
  end

  test "works on :array frames too" do
    arr = :array.from_list(training())
    assert Labels.issued_input(arr, 1) |> Labels.full_x?()
    assert Labels.issued_input(arr, 2) == nil
  end

  test "predicates are shape-agnostic" do
    assert Labels.stick_magnitude(bridge_ctrl(1.0)) == 1.0
    assert Labels.stick_magnitude(raw_ctrl(0.0)) == 1.0
    assert Labels.kind(bridge_ctrl(0.6)) == :stick_dead
    assert Labels.kind(bridge_ctrl(0.75)) == :stick_mid
    assert Labels.kind(bridge_ctrl(0.95)) == :stick_full
    assert Labels.kind(%{bridge_ctrl(0.5) | button_a: true}) == :button
    assert Labels.kind(%{bridge_ctrl(0.95) | button_a: true}) == :both
    assert Labels.kind(nil) == :none
    refute Labels.non_neutral?(bridge_ctrl(0.55))
    assert Labels.non_neutral?(bridge_ctrl(0.7))
  end
end
